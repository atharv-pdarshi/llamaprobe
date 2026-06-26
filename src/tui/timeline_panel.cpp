#include "timeline_panel.hpp"

#include "ftxui/component/component.hpp"
#include "ftxui/component/event.hpp"
#include "ftxui/dom/elements.hpp"
#include "ftxui/screen/color.hpp"

#include <algorithm>
#include <iomanip>
#include <sstream>
#include <string>

using namespace ftxui;

static std::string pad_left(const std::string& s, int width) {
    if (static_cast<int>(s.size()) >= width) return s.substr(0, width);
    return std::string(width - static_cast<int>(s.size()), ' ') + s;
}

ftxui::Component make_timeline_panel(HookEngine& engine, bool& focused) {
    auto scroll = std::make_shared<int>(0);

    return Renderer([&engine, &focused, scroll]() -> Element {
        auto timings = engine.token_timings.snapshot();

        auto border_color = focused ? Color::Cyan : Color::GrayDark;

        if (timings.empty()) {
            return window(
                text("── 6. PER-TOKEN TIMELINE") | color(border_color),
                text("  Waiting for token generation...") | color(Color::GrayDark) | center
            );
        }

        // Find max duration for bar scaling
        float max_dur = 0.f;
        float total_dur = 0.f;
        for (const auto& tt : timings) {
            if (tt.duration_ms > max_dur) max_dur = tt.duration_ms;
            total_dur += tt.duration_ms;
        }
        float avg_dur = total_dur / static_cast<float>(timings.size());

        const int bar_width  = 30;
        const int label_width = 10;

        // Clamp scroll
        int total   = static_cast<int>(timings.size());
        int visible = std::max(3, 8);  // fixed visible rows in the compact panel
        *scroll = std::clamp(*scroll, 0, std::max(0, total - visible));
        int start = *scroll;
        int end   = std::min(total, start + visible);

        Elements rows;
        for (int i = start; i < end; ++i) {
            const auto& tt = timings[i];

            // Label: left-pad token string to label_width
            std::string label = pad_left(tt.token_str, label_width);

            // Bar: scale to bar_width
            int filled = (max_dur > 0.f)
                ? static_cast<int>((tt.duration_ms / max_dur) * bar_width)
                : 0;
            filled = std::clamp(filled, 0, bar_width);

            std::string bar;
            for (int b = 0; b < bar_width; ++b)
                bar += (b < filled) ? "\xe2\x96\x88" : " ";  // UTF-8 full block

            std::ostringstream ms_str;
            ms_str << std::fixed << std::setprecision(1) << tt.duration_ms << "ms";

            Color bar_color = tt.duration_ms > avg_dur * 1.5f ? Color::Yellow : Color::Cyan;

            rows.push_back(hbox({
                text(label)      | color(Color::White),
                text(" ")        ,
                text(bar)        | color(bar_color),
                text(" ")        ,
                text(ms_str.str()) | color(Color::GrayLight),
            }));
        }

        // Footer: avg and max
        std::ostringstream footer;
        footer << " avg: " << std::fixed << std::setprecision(1) << avg_dur
               << "ms  max: " << max_dur << "ms"
               << "  (" << total << " tokens)";

        rows.push_back(separator());
        rows.push_back(text(footer.str()) | color(Color::GrayDark));

        return window(
            text("── 6. PER-TOKEN TIMELINE") | color(border_color),
            vbox(std::move(rows))
        );
    })
    | CatchEvent([scroll](Event e) -> bool {
        if (e == Event::Character('j')) { (*scroll)++; return true; }
        if (e == Event::Character('k')) { (*scroll)--; if (*scroll < 0) *scroll = 0; return true; }
        return false;
    });
}
