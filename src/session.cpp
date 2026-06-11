#include "session.hpp"
// TODO(Phase 4): implement JSON serialization via nlohmann/json

bool Session::start_recording(const std::string& path) {
    path_      = path;
    recording_ = true;
    return true;
}

void Session::append(const LayerPacket& /*pkt*/) {
    // TODO(Phase 4)
}

void Session::stop_recording() {
    recording_ = false;
}

std::vector<LayerPacket> Session::load(const std::string& /*path*/) {
    // TODO(Phase 4)
    return {};
}
