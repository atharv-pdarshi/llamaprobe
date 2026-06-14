#include "app.hpp"
#include "topology_panel.hpp"
#include "stream_panel.hpp"
#include "attention_panel.hpp"
#include "metrics_panel.hpp"
#include "anomaly_panel.hpp"

#include "ftxui/component/component.hpp"
#include "ftxui/component/screen_interactive.hpp"
#include "ftxui/dom/elements.hpp"
#include "ftxui/screen/color.hpp"

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
        row(" P",               " Pause / resume live capture"),
        row(" Q",               " Quit"),
        row(" ?",               " Toggle this help"),
        separator(),
        text(" Press ? or Esc to close ") | center | color(Color::GrayDark),
    }) | border | color(Color::Blue);
}

void run_tui(HookEngine& engine) {
    // ── Shared state ──────────────────────────────────────────────────────────
    int         focused        = 0;   // 0-4 for panels 1-5
    std::string selected_layer;       // set by Panel 1 Space key
    bool        paused         = false;
    bool        show_help      = false;

    // Per-panel focus flags — passed by ref into each panel
    std::array<bool, 5> panel_focused = {true, false, false, false, false};
    auto update_focus = [&] {
        for (int i = 0; i < 5; ++i) panel_focused[i] = (i == focused);
    };

    // ── Build panels ──────────────────────────────────────────────────────────
    auto p1 = make_topology_panel (engine, panel_focused[0], selected_layer);
    auto p2 = make_stream_panel   (engine, panel_focused[1]);
    auto p3 = make_attention_panel(engine, panel_focused[2]);
    auto p4 = make_metrics_panel  (engine, panel_focused[3], selected_layer);
    auto p5 = make_anomaly_panel  (engine, panel_focused[4]);

    // ── Root layout ───────────────────────────────────────────────────────────
    // Row 1: Topology (left 30%) | Live Stream (right 70%)
    // Row 2: Attention Matrix (full width)
    // Row 3: Runtime Metrics (left 50%) | Anomaly Ledger (right 50%)

    auto root = Renderer([&](bool) -> Element {
        // Status bar
        std::ostringstream sb;
        sb << " llamaprobe "
           << "| packets: " << engine.packet_count.load()
           << "  anomalies: " << engine.anomaly_count.load()
           << (paused ? "  [PAUSED]" : "");

        std::string keyhint =
            " [Tab] Cycle Focus  [Space] Select Layer  "
            "[P] Pause  [Q] Quit  [?] Help";

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

        Element main = vbox({
            status,
            separator(),
            row1 | size(HEIGHT, LESS_THAN, 18),
            row2 | size(HEIGHT, LESS_THAN, 14),
            row3 | flex,
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
            focused = (focused + 1) % 5;
            update_focus();
            return true;
        }
        // Shift+Tab — reverse cycle
        if (e == Event::TabReverse) {
            focused = (focused + 4) % 5;
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

        // Dispatch j/k/h/l/Space/H/L/+/- to the focused panel
        switch (focused) {
            case 0: return p1->OnEvent(e);
            case 1: return p2->OnEvent(e);
            case 2: return p3->OnEvent(e);
            case 3: return p4->OnEvent(e);
            case 4: return p5->OnEvent(e);
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

    running = false;
    refresh.join();
}
