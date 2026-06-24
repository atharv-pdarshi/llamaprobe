#include "stream_panel.hpp"

#include "ftxui/component/component.hpp"
#include "ftxui/component/event.hpp"
#include "ftxui/dom/elements.hpp"
#include "ftxui/screen/color.hpp"
#include "ftxui/screen/terminal.hpp"

#include <cstring>

#include <algorithm>
#include <iomanip>
#include <sstream>

using namespace ftxui;

static Color layer_color(LayerType t) {
    switch (t) {
        case LayerType::Attention:  return Color::Cyan;
        case LayerType::MLP:        return Color::Yellow;
        case LayerType::LayerNorm:  return Color::GrayLight;
        case LayerType::Embedding:  return Color::Green;
        case LayerType::Output:     return Color::Magenta;
        default:                    return Color::White;
    }
}

static std::string format_ts(uint64_t us) {
    uint64_t ms  = us / 1000;
    uint64_t sec = ms / 1000;
    ms %= 1000;
    std::ostringstream oss;
    oss << std::setw(2) << std::setfill('0') << sec << "."
        << std::setw(3) << std::setfill('0') << ms;
    return oss.str();
}

ftxui::Component make_stream_panel(HookEngine& engine, bool& focused) {
    // Scroll offset — j/k move it
    auto scroll = std::make_shared<int>(0);

    return Renderer([&engine, &focused, scroll]() -> Element {
        auto packets = engine.packets.snapshot();

        // Header row
        Elements rows;
        rows.push_back(
            hbox({
                text(" ID   ") | bold,
                separator(),
                text(" TIMESTAMP ") | bold,
                separator(),
                text(" LAYER TYPE    ") | bold,
                separator(),
                text(" DEVICE         ") | bold,
            }) | color(Color::White)
        );
        rows.push_back(separator());

        // Clamp scroll
        int total = static_cast<int>(packets.size());
        int visible = std::max(4, ftxui::Terminal::Size().dimy - 24);
        *scroll = std::clamp(*scroll, 0, std::max(0, total - visible));
        int start = std::max(0, total - visible - *scroll);
        int end   = std::min(total, start + visible);

        for (int i = start; i < end; ++i) {
            const auto& p = packets[i];
            bool anomaly  = p.is_anomaly;
            Color c       = anomaly ? Color::Red : layer_color(p.type);

            rows.push_back(
                hbox({
                    text(" " + std::to_string(p.id)) | color(c),
                    text(std::string(std::max(0, 5 - (int)std::to_string(p.id).size()), ' ')),
                    separator(),
                    text(" " + format_ts(p.timestamp_us) + " ") | color(Color::GrayLight),
                    separator(),
                    text(" " + std::string(layer_type_str(p.type))
                         + std::string(std::max(0, 13 - (int)strlen(layer_type_str(p.type))), ' ')
                         + " ") | color(c),
                    separator(),
                    text(" " + std::string(device_str(p.device)) + " ")
                        | color(Color::GrayLight),
                })
            );
        }

        if (packets.empty()) {
            rows.push_back(
                text("  Waiting for inference...") | color(Color::GrayDark) | center
            );
        }

        auto title = std::string("── 2. LIVE PACKET STREAM");
        auto border_color = focused ? Color::Cyan : Color::GrayDark;

        return window(
            text(title) | color(border_color),
            vbox(std::move(rows)) | flex
        );
    })
    | CatchEvent([scroll](Event e) -> bool {
        if (e == Event::Character('j')) { (*scroll)--; return true; }
        if (e == Event::Character('k')) { (*scroll)++; return true; }
        return false;
    });
}