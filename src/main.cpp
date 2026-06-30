#include "hook_engine.hpp"
#include "types.hpp"
#include "tui/app.hpp"
#include "session.hpp"
#include "config.hpp"

#include "llama.h"
#include "ggml.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>

// ── CLI argument parsing ──────────────────────────────────────────────────────

struct Args {
    std::string model_path;
    std::string prompt       = "Explain the transformer attention mechanism.";
    int         n_predict    = 128;
    int         n_ctx        = 2048;
    int         n_threads    = 4;
    int         n_gpu_layers = 0;
    bool        tui_enabled  = true;   // false = just dump packets to stdout (debug)
    std::string replay_path;           // --replay <session.json>
    std::string record_path;           // --record <path>  (empty = no auto-record)
    std::string config_path;           // --config <path>  (default: llamaprobe.toml)
    float       thresh_max_activation = 6.0f;
    float       thresh_sparsity       = 0.80f;
    float       thresh_latency_mult   = 3.0f;
};

static Args parse_args(int argc, char** argv, const Config& cfg) {
    // Start with config-file defaults (CLI overrides these below)
    Args a;
    if (cfg.has("model", "path"))
        a.model_path = cfg.get_string("model", "path");
    if (cfg.has("model", "prompt"))
        a.prompt = cfg.get_string("model", "prompt");
    if (cfg.has("model", "n_predict"))
        a.n_predict = cfg.get_int("model", "n_predict", 128);
    if (cfg.has("model", "threads"))
        a.n_threads = cfg.get_int("model", "threads", 4);
    if (cfg.has("model", "n_gpu_layers"))
        a.n_gpu_layers = cfg.get_int("model", "n_gpu_layers", 0);
    if (cfg.has("anomaly", "max_activation"))
        a.thresh_max_activation = cfg.get_float("anomaly", "max_activation", 6.0f);
    if (cfg.has("anomaly", "sparsity_threshold"))
        a.thresh_sparsity = cfg.get_float("anomaly", "sparsity_threshold", 0.80f);
    if (cfg.has("anomaly", "latency_mult"))
        a.thresh_latency_mult = cfg.get_float("anomaly", "latency_mult", 3.0f);
    if (cfg.has("record", "path"))
        a.record_path = cfg.get_string("record", "path");

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--model") == 0 && i + 1 < argc)
            a.model_path = argv[++i];
        else if (std::strcmp(argv[i], "--prompt") == 0 && i + 1 < argc)
            a.prompt = argv[++i];
        else if (std::strcmp(argv[i], "--n-predict") == 0 && i + 1 < argc)
            a.n_predict = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--threads") == 0 && i + 1 < argc)
            a.n_threads = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--n-gpu-layers") == 0 && i + 1 < argc)
            a.n_gpu_layers = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--no-tui") == 0)
            a.tui_enabled = false;
        else if (std::strcmp(argv[i], "--replay") == 0 && i + 1 < argc) {
            a.replay_path = argv[++i];
            a.tui_enabled = true;
        }
        else if (std::strcmp(argv[i], "--record") == 0 && i + 1 < argc)
            a.record_path = argv[++i];
        else if (std::strcmp(argv[i], "--config") == 0 && i + 1 < argc)
            a.config_path = argv[++i];
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
    // Determine config path from CLI first pass (look for --config before full parse)
    std::string config_path = "llamaprobe.toml";
    for (int i = 1; i < argc - 1; ++i) {
        if (std::strcmp(argv[i], "--config") == 0) {
            config_path = argv[i + 1];
            break;
        }
    }

    // Load config (silently ignore if not found)
    Config cfg;
    cfg.load(config_path);

    Args args = parse_args(argc, argv, cfg);

    if (args.model_path.empty() && args.replay_path.empty()) {
        std::fprintf(stderr,
            "Usage: llamaprobe --model <path.gguf> [--prompt <text>]\n"
            "                  [--n-predict N] [--threads N] [--no-tui]\n"
            "                  [--n-gpu-layers N]\n"
            "                  [--record <path.json>]\n"
            "                  [--config <path.toml>]\n"
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
        auto sdata = Session::load(args.replay_path);
        if (sdata.packets.empty()) {
            std::fprintf(stderr, "Failed to load session (or file is empty): %s\n",
                         args.replay_path.c_str());
            return 1;
        }

        // Build a merged timeline of all four event types sorted by timestamp.
        // This lets panels 3/5/6 appear at the right moment during replay.
        struct ReplayEv {
            uint64_t ts;
            int      kind; // 0=packet, 1=attention, 2=timing, 3=anomaly
            size_t   idx;
        };
        std::vector<ReplayEv> timeline;
        timeline.reserve(sdata.packets.size()
                       + sdata.attention_captures.size()
                       + sdata.token_timings.size()
                       + sdata.anomaly_events.size());

        for (size_t i = 0; i < sdata.packets.size(); ++i)
            timeline.push_back({sdata.packets[i].timestamp_us, 0, i});
        for (size_t i = 0; i < sdata.attention_captures.size(); ++i)
            timeline.push_back({sdata.attention_captures[i].timestamp_us, 1, i});
        for (size_t i = 0; i < sdata.token_timings.size(); ++i)
            timeline.push_back({sdata.token_timings[i].timestamp_us, 2, i});
        for (size_t i = 0; i < sdata.anomaly_events.size(); ++i)
            timeline.push_back({sdata.anomaly_events[i].timestamp_us, 3, i});

        std::stable_sort(timeline.begin(), timeline.end(),
                         [](const ReplayEv& a, const ReplayEv& b){
                             return a.ts < b.ts; });

        engine.inference_start = std::chrono::steady_clock::now();

        std::thread feeder([&sdata, &timeline, &engine] {
            uint64_t prev_ts = 0;
            for (const auto& ev : timeline) {
                if (prev_ts > 0 && ev.ts > prev_ts) {
                    uint64_t delay_us = ev.ts - prev_ts;
                    std::this_thread::sleep_for(std::chrono::microseconds(
                        std::min(delay_us, uint64_t(50'000))));
                }
                prev_ts = ev.ts;

                switch (ev.kind) {
                    case 0: {
                        const auto& pkt = sdata.packets[ev.idx];
                        engine.packets.push(pkt);
                        engine.packet_count.fetch_add(1);
                        if (pkt.is_anomaly)
                            engine.anomaly_count.fetch_add(1);
                        break;
                    }
                    case 1:
                        engine.attention.push(sdata.attention_captures[ev.idx]);
                        break;
                    case 2:
                        engine.token_timings.push(sdata.token_timings[ev.idx]);
                        engine.token_count.fetch_add(1);
                        break;
                    case 3:
                        engine.anomalies.push(sdata.anomaly_events[ev.idx]);
                        break;
                }
            }
        });

        run_tui(engine, "");   // no recording during replay
        feeder.join();
        return 0;
    }

    // ── Load model ────────────────────────────────────────────────────────────
    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = args.n_gpu_layers;

    llama_model* model = llama_model_load_from_file(args.model_path.c_str(), mparams);
    if (!model) {
        std::fprintf(stderr, "Failed to load model: %s\n", args.model_path.c_str());
        return 1;
    }

    // ── Create context with hook installed ────────────────────────────────────
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx        = args.n_ctx;
    cparams.n_threads    = args.n_threads;
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;  // keep softmax tensor visible for Panel 3
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
        ui_thread = std::thread([&]{ run_tui(engine, args.record_path); });
    }

    // ── Run inference ─────────────────────────────────────────────────────────
    engine.begin_inference();
    auto inference_start = engine.inference_start;  // capture for token timing

    llama_batch batch = llama_batch_get_one(tokens_ids.data(), n_tokens);
    if (llama_decode(ctx, batch) != 0) {
        std::fprintf(stderr, "llama_decode failed on prompt.\n");
    }

    // Generate tokens
    llama_sampler* smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.8f));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(42));

    for (int i = 0; i < args.n_predict; ++i) {
        llama_token new_tok = llama_sampler_sample(smpl, ctx, -1);

        if (llama_vocab_is_eog(vocab, new_tok)) break;

        llama_token_to_piece(vocab, new_tok, buf, sizeof(buf), 0, true);
        if (!args.tui_enabled) std::printf("%s", buf);

        batch = llama_batch_get_one(&new_tok, 1);

        // Measure per-token decode time
        auto tok_start = std::chrono::steady_clock::now();
        if (llama_decode(ctx, batch) != 0) break;
        auto tok_end = std::chrono::steady_clock::now();

        float tok_ms = std::chrono::duration<float, std::milli>(tok_end - tok_start).count();

        TokenTiming tt;
        tt.token_idx   = static_cast<uint32_t>(i);
        tt.token_str   = buf;
        tt.duration_ms = tok_ms;
        tt.timestamp_us = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(
                tok_start - inference_start).count());
        engine.token_timings.push(tt);
        if (engine.session_ptr && engine.recording.load())
            engine.session_ptr->append_timing(tt);

        engine.token_count.fetch_add(1);
    }
    if (!args.tui_enabled) std::printf("\n");
    llama_sampler_free(smpl);

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
