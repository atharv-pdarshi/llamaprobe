#include "anomaly_panel.hpp"

#include "ftxui/component/component.hpp"
#include "ftxui/component/event.hpp"
#include "ftxui/dom/elements.hpp"
#include "ftxui/screen/color.hpp"

#include <iomanip>
#include <sstream>

using namespace ftxui;

static Color severity_color(AnomalySeverity s) {
    switch (s) {
        case AnomalySeverity::Critical: return Color::Red;
        case AnomalySeverity::Warning:  return Color::Yellow;
        case AnomalySeverity::Info:     return Color::Cyan;
        default:                        return Color::White;
    }
}

static std::string format_ts(uint64_t us) {
    uint64_t ms  = us / 1000;
    uint64_t sec = ms / 1000;
    ms %= 1000;
    std::ostringstream oss;
    oss << std::setw(2) << std::setfill('0') << sec
        << "." << std::setw(3) << std::setfill('0') << ms;
    return oss.str();
}

ftxui::Component make_anomaly_panel(HookEngine& engine, bool& focused) {
    auto scroll = std::make_shared<int>(0);

    return Renderer([&engine, &focused, scroll]() -> Element {
        auto events = engine.anomalies.snapshot();

        Elements rows;
        if (events.empty()) {
            rows.push_back(
                text("  No anomalies detected.") | color(Color::GrayDark)
            );
        } else {
            int total   = static_cast<int>(events.size());
            int visible = 8;
            *scroll     = std::clamp(*scroll, 0, std::max(0, total - visible));
            int start   = std::max(0, total - visible - *scroll);
            int end     = std::min(total, start + visible);

            for (int i = start; i < end; ++i) {
                const auto& ev = events[i];
                Color c = severity_color(ev.severity);
                rows.push_back(
                    hbox({
                        text(format_ts(ev.timestamp_us) + " ") | color(Color::GrayLight),
                        text(std::string(severity_icon(ev.severity)) + " ") | color(c) | bold,
                        text(ev.description) | color(c),
                    })
                );
            }
        }

        auto border_color = focused ? Color::Cyan : Color::GrayDark;
        return window(
            text("── 5. NUMERICAL ANOMALY LEDGER") | color(border_color),
            vbox(std::move(rows)) | flex
        );
    })
    | CatchEvent([scroll](Event e) -> bool {
        if (e == Event::Character('j')) { (*scroll)--; return true; }
        if (e == Event::Character('k')) { (*scroll)++; return true; }
        return false;
    });
}
