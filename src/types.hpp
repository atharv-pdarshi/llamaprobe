#pragma once

#include <string>
#include <vector>
#include <cstdint>

// ── Enumerations ─────────────────────────────────────────────────────────────

enum class LayerType {
    Embedding,
    Attention,
    MLP,
    LayerNorm,
    Output,
    Other
};

enum class ComputeDevice {
    CPU,
    CUDA_0,
    CUDA_1,
    CUDA_2,
    CUDA_3,
    Unknown
};

enum class DType {
    F32,
    F16,
    Q8_0,
    Q4_0,
    Q4_1,
    Other
};

enum class AnomalySeverity {
    Info,
    Warning,
    Critical
};

// ── Core packet emitted by HookEngine for every captured tensor ───────────────

struct LayerPacket {
    uint32_t    id;
    uint64_t    timestamp_us;       // microseconds since inference start
    std::string layer_name;         // e.g. "layers.1.attn_norm"
    LayerType   type;
    ComputeDevice device;
    std::vector<int64_t> shape;     // tensor dimensions
    DType       dtype;
    float       sparsity;           // fraction of |x| < epsilon values
    float       mean;
    float       max_val;
    float       latency_ms;         // delta from previous packet in same layer path
    bool        is_anomaly;
};

// ── Attention weight matrix for a single head ─────────────────────────────────

struct AttentionCapture {
    uint32_t    layer_idx;
    uint32_t    head_idx;
    uint32_t    total_heads;
    std::vector<std::string>        tokens;
    std::vector<std::vector<float>> matrix;  // [seq_len x seq_len], row = query token
    uint64_t    timestamp_us;
};

// ── Per-token timing (populated in main generation loop) ─────────────────────

struct TokenTiming {
    uint32_t    token_idx;
    std::string token_str;    // decoded text of the token
    float       duration_ms;  // wall time for this token's decode call
    uint64_t    timestamp_us; // microseconds since inference start
};

// ── Anomaly event written by AnomalyDetector ─────────────────────────────────

struct AnomalyEvent {
    uint64_t        timestamp_us;
    AnomalySeverity severity;
    std::string     layer_name;
    std::string     description;    // human-readable, e.g. "Max activation 7.2 > threshold 6.0"
};

// ── Helper: convert enums to display strings ─────────────────────────────────

inline const char* layer_type_str(LayerType t) {
    switch (t) {
        case LayerType::Embedding:  return "Embedding";
        case LayerType::Attention:  return "Attn (Self)";
        case LayerType::MLP:        return "MLP (SwiGLU)";
        case LayerType::LayerNorm:  return "LayerNorm";
        case LayerType::Output:     return "Output";
        default:                    return "Other";
    }
}

inline const char* device_str(ComputeDevice d) {
    switch (d) {
        case ComputeDevice::CPU:    return "CPU";
        case ComputeDevice::CUDA_0: return "CUDA [GPU 0]";
        case ComputeDevice::CUDA_1: return "CUDA [GPU 1]";
        case ComputeDevice::CUDA_2: return "CUDA [GPU 2]";
        case ComputeDevice::CUDA_3: return "CUDA [GPU 3]";
        default:                    return "Unknown";
    }
}

inline const char* dtype_str(DType d) {
    switch (d) {
        case DType::F32:   return "float32";
        case DType::F16:   return "float16";
        case DType::Q8_0:  return "int8 (Q8_0)";
        case DType::Q4_0:  return "int4 (Q4_0)";
        case DType::Q4_1:  return "int4 (Q4_1)";
        default:           return "other";
    }
}

inline const char* severity_icon(AnomalySeverity s) {
    switch (s) {
        case AnomalySeverity::Info:     return "ℹ";
        case AnomalySeverity::Warning:  return "⚠";
        case AnomalySeverity::Critical: return "✖";
        default:                        return "?";
    }
}
