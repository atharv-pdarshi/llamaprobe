#include "hook_engine.hpp"
#include "types.hpp"

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
            "       llamaprobe --replay <session.json>\n");
        return 1;
    }

    // ── Setup hook engine ─────────────────────────────────────────────────────
    HookEngine engine;

    if (!args.replay_path.empty()) {
        // TODO(Phase 4): implement replay from session JSON
        std::fprintf(stderr, "Replay not yet implemented.\n");
        return 1;
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
        // TODO(Phase 1): launch TUI — placeholder print for now
        std::printf("TUI coming in next phase. Run with --no-tui to see raw packets.\n");
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
