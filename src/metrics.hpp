#pragma once

#include "types.hpp"
#include "ggml.h"
#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <string>
#include <chrono>

// Computes activation statistics from a raw ggml tensor.
// All methods are safe to call from the hook callback thread.
class MetricsComputer {
public:
    struct Stats {
        float sparsity = 0.f;   // fraction of elements where |x| < epsilon
        float mean     = 0.f;
        float max_val  = 0.f;
        bool  has_nan  = false;
        bool  has_inf  = false;
    };

    // Compute stats from tensor data. Returns zeroed Stats on unsupported dtype
    // or null data. Samples at most max_elements for speed.
    Stats compute(const struct ggml_tensor* t, size_t max_elements = 4096) const;

    // Convert ggml enum to our DType
    static DType dtype_of(const struct ggml_tensor* t);

    // Latency tracking: returns ms since last call for this layer name
    float latency_delta_ms(const std::string& layer_name);

    // Mark inference start (call before llama_decode)
    void reset();

    // Microseconds since reset()
    uint64_t elapsed_us() const;

private:
    static constexpr float kSparseEpsilon = 1e-6f;

    mutable std::unordered_map<std::string,
        std::chrono::steady_clock::time_point> last_seen_;
    std::chrono::steady_clock::time_point start_;

    Stats compute_f32(const float* data, size_t n) const;
    Stats compute_f16(const void* data,  size_t n) const;
};
