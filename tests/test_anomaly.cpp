#include "../src/types.hpp"
#include "../src/ring_buffer.hpp"
#include <cassert>
#include <cstring>

// Anomaly logic lives in HookEngine; test the data structures it uses.

static void test_anomaly_event_fields() {
    AnomalyEvent ev;
    ev.timestamp_us = 123456;
    ev.severity     = AnomalySeverity::Warning;
    ev.layer_name   = "blk.0.attn";
    ev.description  = "Max activation 7.1 > threshold 6.0";

    assert(ev.severity == AnomalySeverity::Warning);
    assert(ev.layer_name == "blk.0.attn");
    assert(std::strcmp(severity_icon(ev.severity), "⚠") == 0);
}

static void test_anomaly_ring_buffer() {
    RingBuffer<AnomalyEvent, 8> log;

    for (int i = 0; i < 10; ++i) {
        AnomalyEvent ev;
        ev.timestamp_us = static_cast<uint64_t>(i * 1000);
        ev.severity     = AnomalySeverity::Info;
        ev.description  = "event " + std::to_string(i);
        log.push(ev);
    }

    // Buffer holds only 8 — oldest 2 should be gone
    auto snap = log.snapshot();
    assert(snap.size() == 8);
    assert(snap[0].description == "event 2");
    assert(snap[7].description == "event 9");
}

static void test_severity_icons() {
    assert(std::strcmp(severity_icon(AnomalySeverity::Info),     "ℹ") == 0);
    assert(std::strcmp(severity_icon(AnomalySeverity::Warning),  "⚠") == 0);
    assert(std::strcmp(severity_icon(AnomalySeverity::Critical), "✖") == 0);
}

int main() {
    test_anomaly_event_fields();
    test_anomaly_ring_buffer();
    test_severity_icons();
    return 0;
}
