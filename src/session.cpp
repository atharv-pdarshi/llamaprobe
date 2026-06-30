#include "session.hpp"
#include "json.hpp"
#include <cmath>
#include <cstdio>
#include <fstream>

using json = nlohmann::json;

// ── LayerPacket ───────────────────────────────────────────────────────────────

static float safe_f(float v) { return std::isfinite(v) ? v : 0.f; }

static json packet_to_json(const LayerPacket& p) {
    return {
        {"id",           p.id},
        {"timestamp_us", p.timestamp_us},
        {"layer_name",   p.layer_name},
        {"type",         static_cast<int>(p.type)},
        {"type_str",     layer_type_str(p.type)},
        {"device",       static_cast<int>(p.device)},
        {"device_str",   device_str(p.device)},
        {"shape",        p.shape},
        {"dtype",        static_cast<int>(p.dtype)},
        {"dtype_str",    dtype_str(p.dtype)},
        {"sparsity",     safe_f(p.sparsity)},
        {"mean",         safe_f(p.mean)},
        {"max_val",      safe_f(p.max_val)},
        {"latency_ms",   safe_f(p.latency_ms)},
        {"is_anomaly",   p.is_anomaly},
    };
}

static LayerPacket json_to_packet(const json& j) {
    LayerPacket p;
    p.id           = j.at("id").get<uint32_t>();
    p.timestamp_us = j.at("timestamp_us").get<uint64_t>();
    p.layer_name   = j.at("layer_name").get<std::string>();
    p.type         = static_cast<LayerType>(j.at("type").get<int>());
    p.device       = static_cast<ComputeDevice>(j.at("device").get<int>());
    p.shape        = j.at("shape").get<std::vector<int64_t>>();
    p.dtype        = static_cast<DType>(j.at("dtype").get<int>());
    p.sparsity     = j.at("sparsity").get<float>();
    p.mean         = j.at("mean").get<float>();
    p.max_val      = j.at("max_val").get<float>();
    p.latency_ms   = j.at("latency_ms").get<float>();
    p.is_anomaly   = j.at("is_anomaly").get<bool>();
    return p;
}

// ── TokenTiming ───────────────────────────────────────────────────────────────

static json timing_to_json(const TokenTiming& t) {
    return {
        {"token_idx",    t.token_idx},
        {"token_str",    t.token_str},
        {"duration_ms",  safe_f(t.duration_ms)},
        {"timestamp_us", t.timestamp_us},
    };
}

static TokenTiming json_to_timing(const json& j) {
    TokenTiming t;
    t.token_idx    = j.at("token_idx").get<uint32_t>();
    t.token_str    = j.at("token_str").get<std::string>();
    t.duration_ms  = j.at("duration_ms").get<float>();
    t.timestamp_us = j.at("timestamp_us").get<uint64_t>();
    return t;
}

// ── AttentionCapture ──────────────────────────────────────────────────────────

static json attention_to_json(const AttentionCapture& c) {
    // Flatten the seq×seq matrix to a 1-D array for compact storage.
    int64_t seq = static_cast<int64_t>(c.matrix.size());
    std::vector<float> flat;
    flat.reserve(seq * seq);
    for (const auto& row : c.matrix)
        for (float v : row)
            flat.push_back(safe_f(v));
    return {
        {"layer_idx",    c.layer_idx},
        {"head_idx",     c.head_idx},
        {"total_heads",  c.total_heads},
        {"tokens",       c.tokens},
        {"seq_len",      seq},
        {"matrix_flat",  std::move(flat)},
        {"timestamp_us", c.timestamp_us},
    };
}

static AttentionCapture json_to_attention(const json& j) {
    AttentionCapture c;
    c.layer_idx    = j.at("layer_idx").get<uint32_t>();
    c.head_idx     = j.at("head_idx").get<uint32_t>();
    c.total_heads  = j.at("total_heads").get<uint32_t>();
    c.tokens       = j.at("tokens").get<std::vector<std::string>>();
    c.timestamp_us = j.at("timestamp_us").get<uint64_t>();
    int64_t seq    = j.at("seq_len").get<int64_t>();
    auto flat      = j.at("matrix_flat").get<std::vector<float>>();
    c.matrix.assign(seq, std::vector<float>(seq, 0.f));
    for (int64_t r = 0; r < seq; ++r)
        for (int64_t col = 0; col < seq; ++col)
            c.matrix[r][col] = flat[r * seq + col];
    return c;
}

// ── AnomalyEvent ──────────────────────────────────────────────────────────────

static json anomaly_to_json(const AnomalyEvent& e) {
    return {
        {"timestamp_us", e.timestamp_us},
        {"severity",     static_cast<int>(e.severity)},
        {"layer_name",   e.layer_name},
        {"description",  e.description},
    };
}

static AnomalyEvent json_to_anomaly(const json& j) {
    AnomalyEvent e;
    e.timestamp_us = j.at("timestamp_us").get<uint64_t>();
    e.severity     = static_cast<AnomalySeverity>(j.at("severity").get<int>());
    e.layer_name   = j.at("layer_name").get<std::string>();
    e.description  = j.at("description").get<std::string>();
    return e;
}

// ── Session methods ───────────────────────────────────────────────────────────

bool Session::start_recording(const std::string& path) {
    path_ = path;
    buffer_pkts_.clear();
    buffer_timings_.clear();
    buffer_attention_.clear();
    buffer_anomalies_.clear();
    recording_ = true;
    return true;
}

void Session::append(const LayerPacket& pkt) {
    if (!recording_) return;
    buffer_pkts_.push_back(pkt);
}

void Session::append_attention(const AttentionCapture& cap) {
    if (!recording_) return;
    buffer_attention_.push_back(cap);
}

void Session::append_timing(const TokenTiming& tt) {
    if (!recording_) return;
    buffer_timings_.push_back(tt);
}

void Session::append_anomaly(const AnomalyEvent& ev) {
    if (!recording_) return;
    buffer_anomalies_.push_back(ev);
}

void Session::stop_recording() {
    if (!recording_) return;
    recording_ = false;

    json pkts = json::array();
    for (const auto& p : buffer_pkts_)
        pkts.push_back(packet_to_json(p));

    json timings = json::array();
    for (const auto& t : buffer_timings_)
        timings.push_back(timing_to_json(t));

    json attentions = json::array();
    for (const auto& c : buffer_attention_)
        attentions.push_back(attention_to_json(c));

    json anomalies_arr = json::array();
    for (const auto& e : buffer_anomalies_)
        anomalies_arr.push_back(anomaly_to_json(e));

    json root = {
        {"version",            2},
        {"packets",            std::move(pkts)},
        {"token_timings",      std::move(timings)},
        {"attention_captures", std::move(attentions)},
        {"anomaly_events",     std::move(anomalies_arr)},
    };

    // Write to a temp file first so a crash never leaves the target truncated.
    std::string tmp_path = path_ + ".tmp";
    {
        std::ofstream f(tmp_path);
        if (!f) return;
        f << root.dump(2, ' ', false, nlohmann::json::error_handler_t::replace) << "\n";
    }
    std::rename(tmp_path.c_str(), path_.c_str());

    buffer_pkts_.clear();
    buffer_timings_.clear();
    buffer_attention_.clear();
    buffer_anomalies_.clear();
}

SessionData Session::load(const std::string& path) {
    std::ifstream f(path);
    if (!f) return {};

    json j;
    try {
        f >> j;
    } catch (...) {
        return {};
    }

    SessionData out;

    // Legacy format: plain JSON array of LayerPackets.
    if (j.is_array()) {
        out.packets.reserve(j.size());
        for (const auto& item : j)
            out.packets.push_back(json_to_packet(item));
        return out;
    }

    // Current format: versioned object with all four arrays.
    if (j.contains("packets"))
        for (const auto& item : j.at("packets"))
            out.packets.push_back(json_to_packet(item));

    if (j.contains("token_timings"))
        for (const auto& item : j.at("token_timings"))
            out.token_timings.push_back(json_to_timing(item));

    if (j.contains("attention_captures"))
        for (const auto& item : j.at("attention_captures"))
            out.attention_captures.push_back(json_to_attention(item));

    if (j.contains("anomaly_events"))
        for (const auto& item : j.at("anomaly_events"))
            out.anomaly_events.push_back(json_to_anomaly(item));

    return out;
}

// ── Export helpers ────────────────────────────────────────────────────────────

void Session::export_perfetto(const std::vector<LayerPacket>& packets,
                               const std::string& path) {
    json events = json::array();
    for (const auto& p : packets) {
        events.push_back({
            {"name", p.layer_name},
            {"ph",   "X"},
            {"ts",   p.timestamp_us},
            {"dur",  static_cast<uint64_t>(p.latency_ms * 1000.0f)},
            {"pid",  0},
            {"tid",  0},
            {"args", {
                {"sparsity", p.sparsity},
                {"max",      p.max_val},
            }},
        });
    }
    json root;
    root["traceEvents"] = std::move(events);

    std::ofstream f(path);
    if (!f) return;
    f << root.dump(2, ' ', false, nlohmann::json::error_handler_t::replace) << "\n";
}

void Session::export_csv(const std::vector<LayerPacket>& packets,
                          const std::string& path) {
    std::ofstream f(path);
    if (!f) return;
    f << "id,timestamp_us,layer_name,type,device,dtype,"
         "sparsity,mean,max_val,latency_ms,is_anomaly\n";
    for (const auto& p : packets) {
        f << p.id                       << ","
          << p.timestamp_us             << ","
          << p.layer_name               << ","
          << layer_type_str(p.type)     << ","
          << device_str(p.device)       << ","
          << dtype_str(p.dtype)         << ","
          << p.sparsity                 << ","
          << p.mean                     << ","
          << p.max_val                  << ","
          << p.latency_ms               << ","
          << (p.is_anomaly ? "1" : "0") << "\n";
    }
}
