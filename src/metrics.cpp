#include "metrics.hpp"
#include "ggml.h"
#include <algorithm>
#include <numeric>
#include <chrono>

using Clock = std::chrono::steady_clock;

// ── Public ────────────────────────────────────────────────────────────────────

MetricsComputer::Stats MetricsComputer::compute(const struct ggml_tensor* t,
                                                 size_t max_elements) const {
    if (!t || !t->data) return {};

    const size_t total = ggml_nelements(t);
    const size_t n     = std::min(total, max_elements);

    switch (t->type) {
        case GGML_TYPE_F32:
            return compute_f32(static_cast<const float*>(t->data), n);
        case GGML_TYPE_F16:
            return compute_f16(t->data, n);
        default:
            return {};
    }
}

DType MetricsComputer::dtype_of(const struct ggml_tensor* t) {
    if (!t) return DType::Other;
    switch (t->type) {
        case GGML_TYPE_F32:  return DType::F32;
        case GGML_TYPE_F16:  return DType::F16;
        case GGML_TYPE_Q8_0: return DType::Q8_0;
        case GGML_TYPE_Q4_0: return DType::Q4_0;
        case GGML_TYPE_Q4_1: return DType::Q4_1;
        default:             return DType::Other;
    }
}

float MetricsComputer::latency_delta_ms(const std::string& layer_name) {
    auto now = Clock::now();
    float delta_ms = 0.f;

    auto it = last_seen_.find(layer_name);
    if (it != last_seen_.end()) {
        delta_ms = std::chrono::duration<float, std::milli>(now - it->second).count();
        it->second = now;
    } else {
        last_seen_[layer_name] = now;
    }
    return delta_ms;
}

void MetricsComputer::reset() {
    start_ = Clock::now();
    last_seen_.clear();
}

uint64_t MetricsComputer::elapsed_us() const {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            Clock::now() - start_).count());
}

// ── Private ───────────────────────────────────────────────────────────────────

MetricsComputer::Stats MetricsComputer::compute_f32(const float* data, size_t n) const {
    Stats s;
    size_t sparse_count = 0;
    float  sum = 0.f;

    for (size_t i = 0; i < n; ++i) {
        float v = data[i];
        if (std::isnan(v)) { s.has_nan = true; continue; }
        if (std::isinf(v)) { s.has_inf = true; continue; }
        if (std::fabs(v) < kSparseEpsilon) ++sparse_count;
        sum      += v;
        s.max_val = std::max(s.max_val, std::fabs(v));
    }
    s.mean     = n > 0 ? sum / static_cast<float>(n) : 0.f;
    s.sparsity = n > 0 ? static_cast<float>(sparse_count) / static_cast<float>(n) : 0.f;
    return s;
}

MetricsComputer::Stats MetricsComputer::compute_f16(const void* data, size_t n) const {
    // Convert fp16 → fp32 on the fly using ggml helper
    const ggml_fp16_t* src = static_cast<const ggml_fp16_t*>(data);
    Stats s;
    size_t sparse_count = 0;
    float  sum = 0.f;

    for (size_t i = 0; i < n; ++i) {
        float v = ggml_fp16_to_fp32(src[i]);
        if (std::isnan(v)) { s.has_nan = true; continue; }
        if (std::isinf(v)) { s.has_inf = true; continue; }
        if (std::fabs(v) < kSparseEpsilon) ++sparse_count;
        sum      += v;
        s.max_val = std::max(s.max_val, std::fabs(v));
    }
    s.mean     = n > 0 ? sum / static_cast<float>(n) : 0.f;
    s.sparsity = n > 0 ? static_cast<float>(sparse_count) / static_cast<float>(n) : 0.f;
    return s;
}
