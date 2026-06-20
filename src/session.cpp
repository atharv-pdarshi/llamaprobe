#include "session.hpp"
#include "json.hpp"
#include <fstream>
#include <stdexcept>

using json = nlohmann::json;

// Serialize one LayerPacket to a json object
static json packet_to_json(const LayerPacket& p) {
    return {
        {"id",           p.id},
        {"timestamp_us", p.timestamp_us},
        {"layer_name",   p.layer_name},
        {"type",         static_cast<int>(p.type)},
        {"device",       static_cast<int>(p.device)},
        {"shape",        p.shape},
        {"dtype",        static_cast<int>(p.dtype)},
        {"sparsity",     p.sparsity},
        {"mean",         p.mean},
        {"max_val",      p.max_val},
        {"latency_ms",   p.latency_ms},
        {"is_anomaly",   p.is_anomaly},
    };
}

// Deserialize one json object back to a LayerPacket
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

bool Session::start_recording(const std::string& path) {
    path_      = path;
    buffer_.clear();
    recording_ = true;
    return true;
}

void Session::append(const LayerPacket& pkt) {
    if (!recording_) return;
    buffer_.push_back(pkt);
}

void Session::stop_recording() {
    if (!recording_) return;
    recording_ = false;

    json arr = json::array();
    for (const auto& p : buffer_)
        arr.push_back(packet_to_json(p));

    std::ofstream f(path_);
    if (!f) return;   // silently skip if path is bad
    f << arr.dump(2) << "\n";
    buffer_.clear();
}

std::vector<LayerPacket> Session::load(const std::string& path) {
    std::ifstream f(path);
    if (!f) return {};

    json arr;
    try {
        f >> arr;
    } catch (...) {
        return {};
    }

    std::vector<LayerPacket> out;
    out.reserve(arr.size());
    for (const auto& j : arr)
        out.push_back(json_to_packet(j));
    return out;
}
