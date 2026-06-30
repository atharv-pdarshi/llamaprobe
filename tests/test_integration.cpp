#include "../src/hook_engine.hpp"
#include "../src/session.hpp"
#include <cassert>
#include <cstdio>
#include <filesystem>
#include <fstream>

// ── Helpers ───────────────────────────────────────────────────────────────────

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

    assert(std::filesystem::exists(path));

    auto data = Session::load(path);
    assert(data.packets.size() == 3);

    assert(data.packets[0].id         == 0);
    assert(data.packets[0].layer_name == "blk.0.attn");
    assert(data.packets[0].is_anomaly == false);

    assert(data.packets[1].id         == 1);
    assert(data.packets[1].is_anomaly == true);
    assert(data.packets[1].max_val    > 7.f);

    assert(data.packets[2].layer_name == "blk.0.ffn");

    std::remove(path.c_str());
    std::printf("  test_session_round_trip   PASS\n");
}

// ── Test 2: session load on missing file returns empty ─────────────────────

static void test_session_load_missing() {
    auto data = Session::load("llamaprobe_nonexistent_99999.json");
    assert(data.packets.empty());
    std::printf("  test_session_load_missing PASS\n");
}

// ── Test 3: anomaly config thresholds are respected ────────────────────────

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

    auto data = Session::load(path);
    assert(data.packets.size() == 20);
    for (int i = 0; i < 20; ++i)
        assert(data.packets[i].id == static_cast<uint32_t>(i));

    std::remove(path.c_str());
    std::printf("  test_session_ordering     PASS\n");
}

// ── Test 5: token timings round-trip ──────────────────────────────────────

static void test_token_timing_round_trip() {
    const std::string path = "llamaprobe_test_timings.json";

    TokenTiming tt;
    tt.token_idx    = 3;
    tt.token_str    = " hello";
    tt.duration_ms  = 12.5f;
    tt.timestamp_us = 9000;

    Session s;
    s.start_recording(path);
    s.append_timing(tt);
    s.stop_recording();

    auto data = Session::load(path);
    assert(data.token_timings.size() == 1);
    assert(data.token_timings[0].token_idx   == 3);
    assert(data.token_timings[0].token_str   == " hello");
    assert(data.token_timings[0].duration_ms == 12.5f);

    std::remove(path.c_str());
    std::printf("  test_token_timing         PASS\n");
}

// ── Test 6: anomaly events round-trip ─────────────────────────────────────

static void test_anomaly_event_round_trip() {
    const std::string path = "llamaprobe_test_anomalies.json";

    AnomalyEvent ev;
    ev.timestamp_us = 5000;
    ev.severity     = AnomalySeverity::Warning;
    ev.layer_name   = "blk.2.attn";
    ev.description  = "Outlier activation: max=7.5";

    Session s;
    s.start_recording(path);
    s.append_anomaly(ev);
    s.stop_recording();

    auto data = Session::load(path);
    assert(data.anomaly_events.size() == 1);
    assert(data.anomaly_events[0].severity   == AnomalySeverity::Warning);
    assert(data.anomaly_events[0].layer_name == "blk.2.attn");
    assert(data.anomaly_events[0].description == "Outlier activation: max=7.5");

    std::remove(path.c_str());
    std::printf("  test_anomaly_event        PASS\n");
}

// ── Test 7: attention capture round-trip ──────────────────────────────────

static void test_attention_round_trip() {
    const std::string path = "llamaprobe_test_attention.json";

    AttentionCapture cap;
    cap.layer_idx    = 0;
    cap.head_idx     = 1;
    cap.total_heads  = 8;
    cap.timestamp_us = 3000;
    cap.tokens       = {"The", " cat"};
    cap.matrix       = {{0.8f, 0.2f}, {0.4f, 0.6f}};

    Session s;
    s.start_recording(path);
    s.append_attention(cap);
    s.stop_recording();

    auto data = Session::load(path);
    assert(data.attention_captures.size() == 1);
    const auto& c = data.attention_captures[0];
    assert(c.head_idx     == 1);
    assert(c.total_heads  == 8);
    assert(c.tokens[0]    == "The");
    assert(c.matrix[0][0] == 0.8f);
    assert(c.matrix[1][1] == 0.6f);

    std::remove(path.c_str());
    std::printf("  test_attention_round_trip PASS\n");
}

// ── Test 8: legacy plain-array format is still loadable ───────────────────

static void test_legacy_format() {
    const std::string path = "llamaprobe_test_legacy.json";

    // Write old-style plain array
    {
        std::ofstream f(path);
        f << R"([{"id":0,"timestamp_us":1000,"layer_name":"blk.0.attn",)"
             R"("type":1,"device":0,"shape":[32],"dtype":0,)"
             R"("sparsity":0.1,"mean":0.0,"max_val":2.0,)"
             R"("latency_ms":1.5,"is_anomaly":false}])";
    }

    auto data = Session::load(path);
    assert(data.packets.size() == 1);
    assert(data.packets[0].layer_name == "blk.0.attn");
    assert(data.token_timings.empty());
    assert(data.attention_captures.empty());
    assert(data.anomaly_events.empty());

    std::remove(path.c_str());
    std::printf("  test_legacy_format        PASS\n");
}

int main() {
    std::printf("Integration tests:\n");
    test_session_round_trip();
    test_session_load_missing();
    test_anomaly_config_defaults();
    test_session_ordering();
    test_token_timing_round_trip();
    test_anomaly_event_round_trip();
    test_attention_round_trip();
    test_legacy_format();
    std::printf("All integration tests passed.\n");
    return 0;
}
