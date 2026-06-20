#include "hook_engine.hpp"
#include "session.hpp"
#include <cstring>
#include <cmath>
#include <algorithm>

// ── Static trampoline registered with llama.cpp ───────────────────────────────

static bool eval_callback(struct ggml_tensor* t, bool ask, void* user_data) {
    if (ask) return true;   // always allow evaluation
    static_cast<HookEngine*>(user_data)->on_tensor(t);
    return true;
}

// ── Public ────────────────────────────────────────────────────────────────────

void HookEngine::install(llama_context_params& params, HookEngine* engine) {
    params.cb_eval           = eval_callback;
    params.cb_eval_user_data = engine;
}

void HookEngine::set_tokens(std::vector<std::string> toks) {
    std::lock_guard<std::mutex> lk(tokens_mtx_);
    tokens_ = std::move(toks);
}

void HookEngine::begin_inference() {
    inference_start = std::chrono::steady_clock::now();
    metrics_.reset();
    packet_count  = 0;
    anomaly_count = 0;
}

// ── Core capture ──────────────────────────────────────────────────────────────

void HookEngine::on_tensor(const struct ggml_tensor* t) {
    if (!should_capture(t)) return;

    const char* name = ggml_get_name(t);

    LayerPacket pkt;
    pkt.id           = packet_count.fetch_add(1);
    pkt.timestamp_us = metrics_.elapsed_us();
    pkt.layer_name   = name ? name : "<unnamed>";
    pkt.type         = classify_layer(name);
    pkt.device       = detect_device(t);
    pkt.dtype        = MetricsComputer::dtype_of(t);
    pkt.latency_ms   = metrics_.latency_delta_ms(pkt.layer_name);

    // Shape
    for (int i = ggml_n_dims(t) - 1; i >= 0; --i)
        pkt.shape.push_back(t->ne[i]);

    // Activation stats (sample up to 4096 elements for speed)
    auto stats    = metrics_.compute(t);
    pkt.sparsity  = stats.sparsity;
    pkt.mean      = stats.mean;
    pkt.max_val   = stats.max_val;
    pkt.is_anomaly = false;

    check_anomalies(pkt, stats);
    packets.push(pkt);

    if (session_ptr && recording.load()) {
        session_ptr->append(pkt);
    }

    // Capture attention weight matrix separately if this is an attention output
    if (is_attention_weights(name) && t->data &&
        (t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_F16)) {

        // Dims: [seq_len, seq_len, n_heads, ...]
        int64_t seq  = t->ne[0];
        int64_t heads = t->ne[2] > 0 ? t->ne[2] : 1;

        std::lock_guard<std::mutex> lk(tokens_mtx_);
        for (int64_t h = 0; h < std::min(heads, (int64_t)8); ++h) {
            AttentionCapture cap;
            cap.layer_idx   = pkt.id;
            cap.head_idx    = static_cast<uint32_t>(h);
            cap.total_heads = static_cast<uint32_t>(heads);
            cap.timestamp_us = pkt.timestamp_us;
            cap.tokens      = tokens_;

            cap.matrix.resize(seq, std::vector<float>(seq, 0.f));
            for (int64_t r = 0; r < seq; ++r) {
                for (int64_t c = 0; c < seq; ++c) {
                    size_t idx = h * seq * seq + r * seq + c;
                    if (t->type == GGML_TYPE_F32) {
                        cap.matrix[r][c] = static_cast<const float*>(t->data)[idx];
                    } else {
                        cap.matrix[r][c] = ggml_fp16_to_fp32(
                            static_cast<const ggml_fp16_t*>(t->data)[idx]);
                    }
                }
            }
            attention.push(std::move(cap));
        }
    }
}

// ── Private helpers ───────────────────────────────────────────────────────────

bool HookEngine::should_capture(const struct ggml_tensor* t) const {
    if (!t) return false;
    const char* name = ggml_get_name(t);
    if (!name || name[0] == '\0') return false;
    // Skip weight tensors — we only want activation outputs
    if (std::strstr(name, ".weight")) return false;
    if (std::strstr(name, ".bias"))   return false;
    // Only capture named layer outputs, not intermediate scratch tensors
    return std::strstr(name, "blk.")      ||
           std::strstr(name, "token_embd") ||
           std::strstr(name, "result_norm") ||
           std::strstr(name, "output");
}

LayerType HookEngine::classify_layer(const char* name) const {
    if (!name) return LayerType::Other;
    if (std::strstr(name, "token_embd"))               return LayerType::Embedding;
    if (std::strstr(name, "attn_norm") ||
        std::strstr(name, "ffn_norm")  ||
        std::strstr(name, "result_norm")) return LayerType::LayerNorm;
    if (std::strstr(name, "attn"))                     return LayerType::Attention;
    if (std::strstr(name, "ffn") || std::strstr(name, "mlp")) return LayerType::MLP;
    if (std::strstr(name, "output"))                   return LayerType::Output;
    return LayerType::Other;
}

ComputeDevice HookEngine::detect_device(const struct ggml_tensor* t) const {
    if (!t || !t->buffer) return ComputeDevice::CPU;
    const char* bname = ggml_backend_buffer_name(t->buffer);
    if (!bname) return ComputeDevice::CPU;
    if (std::strstr(bname, "CUDA") || std::strstr(bname, "cuda")) {
        if (std::strstr(bname, "1")) return ComputeDevice::CUDA_1;
        if (std::strstr(bname, "2")) return ComputeDevice::CUDA_2;
        if (std::strstr(bname, "3")) return ComputeDevice::CUDA_3;
        return ComputeDevice::CUDA_0;
    }
    return ComputeDevice::CPU;
}

bool HookEngine::is_attention_weights(const char* name) const {
    if (!name) return false;
    // Captures the softmax output of attention (the actual weight matrix)
    return std::strstr(name, "attn_weight") ||
           std::strstr(name, "kq_soft_max") ||
           std::strstr(name, "attn_softmax");
}

void HookEngine::check_anomalies(const LayerPacket& pkt,
                                  const MetricsComputer::Stats& stats) {
    AnomalyEvent ev;
    ev.timestamp_us = pkt.timestamp_us;
    ev.layer_name   = pkt.layer_name;

    auto emit = [&](AnomalySeverity sev, std::string desc) {
        ev.severity    = sev;
        ev.description = std::move(desc);
        anomalies.push(ev);
        anomaly_count.fetch_add(1);
        // Mark the packet we just built — caller will see this
        const_cast<LayerPacket&>(pkt).is_anomaly = true;
    };

    if (stats.has_nan || stats.has_inf)
        emit(AnomalySeverity::Critical, "NaN or Inf detected in activation");

    if (pkt.max_val > anomaly_cfg.max_activation)
        emit(AnomalySeverity::Warning,
             "Outlier activation: max=" + std::to_string(pkt.max_val) +
             " > threshold=" + std::to_string(anomaly_cfg.max_activation));

    if (pkt.sparsity > anomaly_cfg.sparsity)
        emit(AnomalySeverity::Info,
             "High sparsity: " + std::to_string(int(pkt.sparsity * 100)) + "%");

    // Latency spike detection using rolling average
    if (rolling_avg_latency_ > 0.f &&
        pkt.latency_ms > anomaly_cfg.latency_mult * rolling_avg_latency_)
        emit(AnomalySeverity::Warning,
             "Slow layer: " + std::to_string(pkt.latency_ms) + "ms (avg=" +
             std::to_string(rolling_avg_latency_) + "ms)");

    // Update rolling average (exponential moving average, alpha=0.1)
    if (pkt.latency_ms > 0.f)
        rolling_avg_latency_ = rolling_avg_latency_ == 0.f
            ? pkt.latency_ms
            : 0.9f * rolling_avg_latency_ + 0.1f * pkt.latency_ms;
}
