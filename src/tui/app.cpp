#include "app.hpp"
#include "session.hpp"
#include "topology_panel.hpp"
#include "stream_panel.hpp"
#include "attention_panel.hpp"
#include "metrics_panel.hpp"
#include "anomaly_panel.hpp"
#include "timeline_panel.hpp"

#include "ftxui/component/component.hpp"
#include "ftxui/component/screen_interactive.hpp"
#include "ftxui/dom/elements.hpp"
#include "ftxui/screen/color.hpp"

#include <memory>
#include <optional>
#include <thread>
#include <chrono>
#include <atomic>
#include <iomanip>
#include <sstream>

using namespace ftxui;

static Element make_help_overlay() {
    auto row = [](const char* key, const char* action) {
        return hbox({
            text(key)    | color(Color::Yellow) | size(WIDTH, EQUAL, 18),
            text(action) | color(Color::White),
        });
    };
    return vbox({
        text("  Keybindings  ") | bold | center,
        separator(),
        row(" Tab / Shift+Tab", " Cycle panel focus"),
        row(" j / k",           " Navigate up / down in panel"),
        row(" h / l",           " Pan attention matrix left / right"),
        row(" Space",           " Select layer as capture target"),
        row(" F",               " Toggle attention matrix fullscreen"),
        row(" + / -",           " Increase / decrease contrast"),
        row(" H / L",           " Cycle attention head"),
        row(" R",               " Toggle session recording"),
        row(" E",               " Export buffer to JSON + CSV + Perfetto"),
        row(" C",               " Snapshot layer for comparison in Panel 4"),
        row(" P",               " Pause / resume live capture"),
        row(" /",               " Filter Panel 2 by layer name"),
        row(" Q",               " Quit"),
        row(" ?",               " Toggle this help"),
        separator(),
        text(" Press ? or Esc to close ") | center | color(Color::GrayDark),
    }) | border | color(Color::Blue);
}

static void export_snapshot(HookEngine& engine, const std::string& base_path) {
    auto snap_pkts = engine.packets.snapshot();
    auto snap_attn = engine.attention.snapshot();
    auto snap_tt   = engine.token_timings.snapshot();
    auto snap_anom = engine.anomalies.snapshot();

    {
        Session tmp;
        tmp.start_recording(base_path + ".json");
        for (const auto& pkt : snap_pkts) tmp.append(pkt);
        for (const auto& cap : snap_attn) tmp.append_attention(cap);
        for (const auto& tt  : snap_tt)   tmp.append_timing(tt);
        for (const auto& ev  : snap_anom) tmp.append_anomaly(ev);
        tmp.stop_recording();
    }

    Session::export_csv(snap_pkts, base_path + ".csv");
    Session::export_perfetto(snap_pkts, base_path + "_perfetto.json");
}

void run_tui(HookEngine& engine, const std::string& cli_record_path) {
    // ── Shared state ──────────────────────────────────────────────────────────
    Session session;
    std::string record_path = cli_record_path.empty() ? "session.json" : cli_record_path;
    engine.session_ptr = &session;

    if (!cli_record_path.empty()) {
        session.start_recording(record_path);
        engine.recording.store(true);
    }

    int         focused        = 0;   // 0-5 for panels 1-6
    std::string selected_layer;       // set by Panel 1 Space key
    bool        paused         = false;
    bool        show_help      = false;

    // Per-panel focus flags — passed by ref into each panel
    std::array<bool, 6> panel_focused = {true, false, false, false, false, false};
    auto update_focus = [&] {
        for (int i = 0; i < 6; ++i) panel_focused[i] = (i == focused);
    };

    // Comparison snapshot for Panel 4 (C key)
    auto compare_pkt = std::make_shared<std::optional<LayerPacket>>();

    // ── Build panels ──────────────────────────────────────────────────────────
    auto p1 = make_topology_panel (engine, panel_focused[0], selected_layer);
    auto p2 = make_stream_panel   (engine, panel_focused[1]);
    auto p3 = make_attention_panel(engine, panel_focused[2]);
    auto p4 = make_metrics_panel  (engine, panel_focused[3], selected_layer, compare_pkt);
    auto p5 = make_anomaly_panel  (engine, panel_focused[4]);
    auto p6 = make_timeline_panel (engine, panel_focused[5]);

    // ── Root layout ───────────────────────────────────────────────────────────
    // Row 1: Topology (left 30%) | Live Stream (right 70%)
    // Row 2: Attention Matrix (full width)
    // Row 3: Runtime Metrics (left 50%) | Anomaly Ledger (right 50%)
    // Row 4: Per-Token Timeline (full width, compact)

    auto root = Renderer([&](bool) -> Element {
        // Status bar
        std::ostringstream sb;
        sb << " llamaprobe ";

        // tokens/sec
        uint32_t toks = engine.token_count.load();
        if (toks > 0) {
            auto now     = std::chrono::steady_clock::now();
            double secs  = std::chrono::duration<double>(
                               now - engine.inference_start).count();
            double tps   = secs > 0.0 ? toks / secs : 0.0;
            sb << "| " << std::fixed << std::setprecision(1) << tps << " tok/s  ";
        }

        sb << "| packets: " << engine.packet_count.load()
           << "  anomalies: " << engine.anomaly_count.load();

        if (engine.recording.load())
            sb << "  ● REC";
        if (paused)
            sb << "  [PAUSED]";

        std::string keyhint =
            " [Tab] Cycle Focus  [Space] Select Layer  "
            "[C] Compare  [P] Pause  [Q] Quit  [?] Help";

        Element status = hbox({
            text(sb.str())   | color(Color::Green) | bold,
            filler(),
            text(keyhint)    | color(Color::GrayDark),
        });

        Element row1 = hbox({
            p1->Render() | size(WIDTH, LESS_THAN, 36),
            p2->Render() | flex,
        });

        Element row2 = p3->Render();

        Element row3 = hbox({
            p4->Render() | flex,
            p5->Render() | flex,
        });

        Element row4 = p6->Render() | size(HEIGHT, LESS_THAN, 10);

        Element main = vbox({
            status,
            separator(),
            row1 | size(HEIGHT, LESS_THAN, 18),
            row2 | size(HEIGHT, LESS_THAN, 14),
            row3 | flex,
            row4,
        });

        if (show_help) {
            return dbox({
                main,
                make_help_overlay() | clear_under | center,
            });
        }
        return main;
    });

    // ── Global keyboard handler ───────────────────────────────────────────────
    auto screen = ScreenInteractive::Fullscreen();

    auto app = CatchEvent(root, [&](Event e) -> bool {
        // Q — quit
        if (e == Event::Character('q') || e == Event::Character('Q')) {
            screen.ExitLoopClosure()();
            return true;
        }
        // Tab — cycle panel focus
        if (e == Event::Tab || e == Event::Character('\t')) {
            focused = (focused + 1) % 6;
            update_focus();
            return true;
        }
        // Shift+Tab — reverse cycle
        if (e == Event::TabReverse) {
            focused = (focused + 5) % 6;
            update_focus();
            return true;
        }
        // ? — toggle help overlay
        if (e == Event::Character('?')) {
            show_help = !show_help;
            return true;
        }
        // Esc — close help overlay if open
        if (show_help && e == Event::Escape) {
            show_help = false;
            return true;
        }
        // P — pause/resume capture
        if (e == Event::Character('p') || e == Event::Character('P')) {
            paused = !paused;
            return true;
        }
        // R — toggle recording
        if (e == Event::Character('r') || e == Event::Character('R')) {
            bool was = engine.recording.load();
            bool next_state = !was;
            engine.recording.store(next_state);
            if (next_state) {
                session.start_recording(record_path);
            } else {
                session.stop_recording();
            }
            return true;
        }
        // E — export current packet buffer to JSON + CSV + Perfetto
        if (e == Event::Character('e') || e == Event::Character('E')) {
            try { export_snapshot(engine, "session_export"); } catch (...) {}
            return true;
        }
        // C — snapshot current layer for comparison in Panel 4
        if (e == Event::Character('c') || e == Event::Character('C')) {
            if (!selected_layer.empty()) {
                auto snap = engine.packets.snapshot();
                for (int i = static_cast<int>(snap.size()) - 1; i >= 0; --i) {
                    if (snap[i].layer_name == selected_layer) {
                        *compare_pkt = snap[i];
                        break;
                    }
                }
            }
            return true;
        }

        // Dispatch j/k/h/l/Space/H/L/+/-// to the focused panel
        switch (focused) {
            case 0: return p1->OnEvent(e);
            case 1: return p2->OnEvent(e);
            case 2: return p3->OnEvent(e);
            case 3: return p4->OnEvent(e);
            case 4: return p5->OnEvent(e);
            case 5: return p6->OnEvent(e);
        }
        return false;
    });

    // ── Refresh thread: trigger re-render every 100ms ─────────────────────────
    std::atomic<bool> running{true};
    std::thread refresh([&] {
        while (running) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            screen.PostEvent(Event::Custom);
        }
    });

    screen.Loop(app);

    // Stop recording and clear pointers before session goes out of scope.
    // If we don't, the inference thread (main) can still call on_tensor with
    // recording=true and session_ptr pointing to a destroyed stack object.
    engine.recording.store(false);
    engine.session_ptr = nullptr;
    session.stop_recording();

    running = false;
    refresh.join();
}
