#pragma once

#include "types.hpp"
#include <string>
#include <vector>

// Phase 4: record and replay captured sessions as JSON.

class Session {
public:
    // Start writing packets to path. Call stop() to finish.
    bool start_recording(const std::string& path);
    void append(const LayerPacket& pkt);
    void stop_recording();
    bool is_recording() const { return recording_; }

    // Load a saved session and return all packets in order.
    static std::vector<LayerPacket> load(const std::string& path);

private:
    bool        recording_ = false;
    std::string path_;
};
