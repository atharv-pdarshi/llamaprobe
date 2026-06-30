#pragma once

#include "types.hpp"
#include <string>
#include <vector>

// All data captured in one session, returned by Session::load().
struct SessionData {
    std::vector<LayerPacket>      packets;
    std::vector<TokenTiming>      token_timings;
    std::vector<AttentionCapture> attention_captures;
    std::vector<AnomalyEvent>     anomaly_events;
};

class Session {
public:
    bool start_recording(const std::string& path);
    void append          (const LayerPacket&      pkt);
    void append_attention(const AttentionCapture& cap);
    void append_timing   (const TokenTiming&      tt);
    void append_anomaly  (const AnomalyEvent&     ev);
    void stop_recording();
    bool is_recording() const { return recording_; }

    // Load a saved session. Handles both the legacy (plain-array) format and
    // the current (versioned object) format.
    static SessionData load(const std::string& path);

    // Export helpers (packets only — for CSV / Perfetto consumers)
    static void export_perfetto(const std::vector<LayerPacket>& packets,
                                const std::string& path);
    static void export_csv     (const std::vector<LayerPacket>& packets,
                                const std::string& path);

private:
    bool recording_ = false;
    std::string path_;
    std::vector<LayerPacket>      buffer_pkts_;
    std::vector<TokenTiming>      buffer_timings_;
    std::vector<AttentionCapture> buffer_attention_;
    std::vector<AnomalyEvent>     buffer_anomalies_;
};
