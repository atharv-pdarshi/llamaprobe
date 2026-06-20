#include "hook_engine.hpp"
#include "types.hpp"
#include "tui/app.hpp"
#include "session.hpp"

#include "llama.h"
#include "ggml.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <thread>
#include <atomic>

// ── CLI argument parsing ──────────────────────────────────────────────────────

struct Args {
    std::string model_path;
    std::string prompt       = "Explain the transformer attention mechanism.";
    int         n_predict    = 128;
    int         n_ctx        = 2048;
    int         n_threads    = 4;
    bool        tui_enabled  = true;   // false = just dump packets to stdout (debug)
    std::string replay_path;           // --replay <session.json>
    float       thresh_max_activation = 6.0f;
    float       thresh_sparsity       = 0.80f;
    float       thresh_latency_mult   = 3.0f;
};

static Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--model") == 0 && i + 1 < argc)
            a.model_path = argv[++i];
        else if (std::strcmp(argv[i], "--prompt") == 0 && i + 1 < argc)
            a.prompt = argv[++i];
        else if (std::strcmp(argv[i], "--n-predict") == 0 && i + 1 < argc)
            a.n_predict = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--threads") == 0 && i + 1 < argc)
            a.n_threads = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--no-tui") == 0)
            a.tui_enabled = false;
        else if (std::strcmp(argv[i], "--replay") == 0 && i + 1 < argc) {
            a.replay_path = argv[++i];
            a.tui_enabled = true;
        }
        else if (std::strcmp(argv[i], "--max-activation") == 0 && i + 1 < argc)
            a.thresh_max_activation = static_cast<float>(std::atof(argv[++i]));
        else if (std::strcmp(argv[i], "--sparsity-threshold") == 0 && i + 1 < argc)
            a.thresh_sparsity = static_cast<float>(std::atof(argv[++i]));
        else if (std::strcmp(argv[i], "--latency-mult") == 0 && i + 1 < argc)
            a.thresh_latency_mult = static_cast<float>(std::atof(argv[++i]));
    }
    return a;
}

// ── Debug dump (--no-tui mode) ────────────────────────────────────────────────

static void dump_packets(HookEngine& engine, std::atomic<bool>& running) {
    uint32_t last = 0;
    while (running) {
        auto snap = engine.packets.snapshot();
        for (size_t i = last; i < snap.size(); ++i) {
            const auto& p = snap[i];
            std::printf("[%6u] %8.3f ms  %-14s  %-12s  sparsity=%.1f%%  max=%.3f\n",
                p.id,
                static_cast<double>(p.latency_ms),
                layer_type_str(p.type),
                device_str(p.device),
                static_cast<double>(p.sparsity * 100.f),
                static_cast<double>(p.max_val));
        }
        last = static_cast<uint32_t>(snap.size());
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
}

// ── Main ──────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);

    if (args.model_path.empty() && args.replay_path.empty()) {
        std::fprintf(stderr,
            "Usage: llamaprobe --model <path.gguf> [--prompt <text>]\n"
            "                  [--n-predict N] [--threads N] [--no-tui]\n"
            "                  [--max-activation F] [--sparsity-threshold F] [--latency-mult F]\n"
            "       llamaprobe --replay <session.json>\n");
        return 1;
    }

    // ── Setup hook engine ─────────────────────────────────────────────────────
    HookEngine engine;
    engine.anomaly_cfg.max_activation = args.thresh_max_activation;
    engine.anomaly_cfg.sparsity       = args.thresh_sparsity;
    engine.anomaly_cfg.latency_mult   = args.thresh_latency_mult;

    if (!args.replay_path.empty()) {
        auto packets = Session::load(args.replay_path);
        if (packets.empty()) {
            std::fprintf(stderr, "Failed to load session: %s\n",
                         args.replay_path.c_str());
            return 1;
        }

        // Feed packets into the engine on a background thread, then run TUI
        std::thread feeder([&] {
            uint64_t prev_ts = 0;
            for (const auto& pkt : packets) {
                if (prev_ts > 0 && pkt.timestamp_us > prev_ts) {
                    uint64_t delay_us = pkt.timestamp_us - prev_ts;
                    // Cap per-packet sleep to 50 ms so replay doesn't feel sluggish
                    uint64_t sleep_us = std::min(delay_us, uint64_t(50'000));
                    std::this_thread::sleep_for(
                        std::chrono::microseconds(sleep_us));
                }
                prev_ts = pkt.timestamp_us;
                engine.packets.push(pkt);
                engine.packet_count.fetch_add(1);
                if (pkt.is_anomaly)
                    engine.anomaly_count.fetch_add(1);
            }
        });

        run_tui(engine);
        feeder.join();
        return 0;
    }

    // ── Load model ────────────────────────────────────────────────────────────
    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;   // CPU-only

    llama_model* model = llama_model_load_from_file(args.model_path.c_str(), mparams);
    if (!model) {
        std::fprintf(stderr, "Failed to load model: %s\n", args.model_path.c_str());
        return 1;
    }

    // ── Create context with hook installed ────────────────────────────────────
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx      = args.n_ctx;
    cparams.n_threads  = args.n_threads;
    HookEngine::install(cparams, &engine);

    llama_context* ctx = llama_new_context_with_model(model, cparams);
    if (!ctx) {
        std::fprintf(stderr, "Failed to create context.\n");
        llama_model_free(model);
        return 1;
    }

    // ── Tokenize prompt ───────────────────────────────────────────────────────
    const llama_vocab* vocab = llama_model_get_vocab(model);
    std::vector<llama_token> tokens_ids(args.prompt.size() + 16);
    int n_tokens = llama_tokenize(vocab,
                                   args.prompt.c_str(),
                                   static_cast<int>(args.prompt.size()),
                                   tokens_ids.data(),
                                   static_cast<int>(tokens_ids.size()),
                                   /*add_special=*/true,
                                   /*parse_special=*/true);
    if (n_tokens < 0) {
        tokens_ids.resize(-n_tokens);
        n_tokens = llama_tokenize(vocab,
                                   args.prompt.c_str(),
                                   static_cast<int>(args.prompt.size()),
                                   tokens_ids.data(),
                                   static_cast<int>(tokens_ids.size()),
                                   true, true);
    }
    tokens_ids.resize(n_tokens);

    // Pass decoded token strings to hook engine for attention labeling
    std::vector<std::string> token_strs;
    token_strs.reserve(n_tokens);
    char buf[64];
    for (auto tid : tokens_ids) {
        llama_token_to_piece(vocab, tid, buf, sizeof(buf), 0, true);
        token_strs.push_back(buf);
    }
    engine.set_tokens(token_strs);

    // ── Launch TUI or debug dump ──────────────────────────────────────────────
    std::atomic<bool> running{true};
    std::thread ui_thread;

    if (!args.tui_enabled) {
        ui_thread = std::thread([&]{ dump_packets(engine, running); });
    } else {
        ui_thread = std::thread([&]{ run_tui(engine); });
    }

    // ── Run inference ─────────────────────────────────────────────────────────
    engine.begin_inference();

    llama_batch batch = llama_batch_get_one(tokens_ids.data(), n_tokens);
    if (llama_decode(ctx, batch) != 0) {
        std::fprintf(stderr, "llama_decode failed on prompt.\n");
    }

    // Generate tokens
    for (int i = 0; i < args.n_predict; ++i) {
        llama_token new_tok = llama_sampler_sample(
            llama_sampler_chain_init(llama_sampler_chain_default_params()),
            ctx, -1);

        if (llama_vocab_is_eog(vocab, new_tok)) break;

        llama_token_to_piece(vocab, new_tok, buf, sizeof(buf), 0, true);
        if (!args.tui_enabled) std::printf("%s", buf);

        batch = llama_batch_get_one(&new_tok, 1);
        if (llama_decode(ctx, batch) != 0) break;
        engine.token_count.fetch_add(1);
    }
    if (!args.tui_enabled) std::printf("\n");

    // ── Shutdown ──────────────────────────────────────────────────────────────
    running = false;
    if (ui_thread.joinable()) ui_thread.join();

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    std::printf("\nCaptured %u packets, %u anomalies.\n",
        engine.packet_count.load(), engine.anomaly_count.load());
    return 0;
}
