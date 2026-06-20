#include "../src/hook_engine.hpp"
#include "../src/session.hpp"
#include <cassert>
#include <cstdio>
#include <filesystem>

// ── Helpers ───────────────────────────────────────────────────────────────────

// Build a minimal LayerPacket for testing
static LayerPacket make_packet(uint32_t id, const std::string& name,
                                float max_val, float sparsity,
                                float latency_ms, bool is_anomaly = false) {
    LayerPacket p;
    p.id           = id;
    p.timestamp_us = id * 1000;
    p.layer_name   = name;
    p.type         = LayerType::Attention;
    p.device       = ComputeDevice::CPU;
    p.dtype        = DType::F16;
    p.shape        = {1, 32, 2048};
    p.sparsity     = sparsity;
    p.mean         = 0.f;
    p.max_val      = max_val;
    p.latency_ms   = latency_ms;
    p.is_anomaly   = is_anomaly;
    return p;
}

// ── Test 1: session round-trip ─────────────────────────────────────────────

static void test_session_round_trip() {
    const std::string path = "llamaprobe_test_session.json";

    Session s;
    assert(s.start_recording(path));

    auto p0 = make_packet(0, "blk.0.attn", 4.5f, 0.2f, 1.1f);
    auto p1 = make_packet(1, "blk.1.attn", 7.2f, 0.9f, 2.3f, true);
    auto p2 = make_packet(2, "blk.0.ffn",  3.1f, 0.1f, 0.8f);

    s.append(p0);
    s.append(p1);
    s.append(p2);
    s.stop_recording();

    // File must exist
    assert(std::filesystem::exists(path));

    auto loaded = Session::load(path);
    assert(loaded.size() == 3);

    assert(loaded[0].id         == 0);
    assert(loaded[0].layer_name == "blk.0.attn");
    assert(loaded[0].is_anomaly == false);

    assert(loaded[1].id         == 1);
    assert(loaded[1].is_anomaly == true);
    assert(loaded[1].max_val    > 7.f);

    assert(loaded[2].layer_name == "blk.0.ffn");

    std::remove(path.c_str());
    std::printf("  test_session_round_trip   PASS\n");
}

// ── Test 2: session load on missing file returns empty ─────────────────────

static void test_session_load_missing() {
    auto packets = Session::load("llamaprobe_nonexistent_99999.json");
    assert(packets.empty());
    std::printf("  test_session_load_missing PASS\n");
}

// ── Test 3: anomaly config thresholds are respected ────────────────────────
// We can't run real inference in a unit test, but we CAN verify that
// AnomalyConfig fields round-trip correctly and that the struct defaults match
// the documented values.

static void test_anomaly_config_defaults() {
    AnomalyConfig cfg;
    assert(cfg.max_activation == 6.0f);
    assert(cfg.sparsity       == 0.80f);
    assert(cfg.latency_mult   == 3.0f);
    std::printf("  test_anomaly_config       PASS\n");
}

// ── Test 4: replay ordering is preserved ──────────────────────────────────

static void test_session_ordering() {
    const std::string path = "llamaprobe_test_order.json";

    Session s;
    s.start_recording(path);
    for (int i = 0; i < 20; ++i)
        s.append(make_packet(i, "blk." + std::to_string(i) + ".attn",
                              1.f, 0.f, 0.5f));
    s.stop_recording();

    auto loaded = Session::load(path);
    assert(loaded.size() == 20);
    for (int i = 0; i < 20; ++i)
        assert(loaded[i].id == static_cast<uint32_t>(i));

    std::remove(path.c_str());
    std::printf("  test_session_ordering     PASS\n");
}

int main() {
    std::printf("Integration tests:\n");
    test_session_round_trip();
    test_session_load_missing();
    test_anomaly_config_defaults();
    test_session_ordering();
    std::printf("All integration tests passed.\n");
    return 0;
}
