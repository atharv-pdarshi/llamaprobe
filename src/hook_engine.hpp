#pragma once

#include "types.hpp"
#include "ring_buffer.hpp"
#include "metrics.hpp"
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <string>
#include <vector>
#include <mutex>
#include <atomic>

class HookEngine {
public:
    // ── Shared buffers (TUI reads via snapshot()) ─────────────────────────────
    RingBuffer<LayerPacket,     512> packets;
    RingBuffer<AttentionCapture, 32> attention;
    RingBuffer<AnomalyEvent,    128> anomalies;

    // ── Setup ─────────────────────────────────────────────────────────────────

    // Call this before llama_new_context_with_model to register the callback
    static void install(llama_context_params& params, HookEngine* engine);

    // Set decoded token strings so attention panels can label axes
    void set_tokens(std::vector<std::string> toks);

    // Call before each llama_decode() call to reset timing
    void begin_inference();

    // ── Stats (read by TUI status bar) ────────────────────────────────────────
    std::atomic<uint32_t> packet_count{0};
    std::atomic<uint32_t> anomaly_count{0};

    // ── Callback entry point (public so the static trampoline can call it) ────
    void on_tensor(const struct ggml_tensor* t);

private:
    MetricsComputer          metrics_;
    std::vector<std::string> tokens_;
    std::mutex               tokens_mtx_;

    // Anomaly thresholds (configurable later via config file)
    float thresh_max_activation = 6.0f;
    float thresh_sparsity       = 0.80f;
    float thresh_latency_mult   = 3.0f;   // spike if latency > N * rolling avg
    float rolling_avg_latency_  = 0.f;

    LayerType     classify_layer(const char* name) const;
    ComputeDevice detect_device(const struct ggml_tensor* t) const;
    bool          should_capture(const struct ggml_tensor* t) const;
    bool          is_attention_weights(const char* name) const;
    void          check_anomalies(const LayerPacket& pkt,
                                  const MetricsComputer::Stats& stats);
};
