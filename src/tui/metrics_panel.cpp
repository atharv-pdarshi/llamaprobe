#include "metrics_panel.hpp"

#include "ftxui/component/component.hpp"
#include "ftxui/dom/elements.hpp"
#include "ftxui/screen/color.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <memory>
#include <optional>
#include <sstream>

using namespace ftxui;

// Render a filled bar: "████░░░░░ 54%"
static Element sparsity_bar(float sparsity, int width = 16) {
    int filled = static_cast<int>(sparsity * width);
    std::string bar_str;
    for (int i = 0; i < width; ++i)
        bar_str += (i < filled) ? "\xe2\x96\x88" : "\xe2\x96\x91";
    bar_str += " " + std::to_string(static_cast<int>(sparsity * 100)) + "%";

    Color c = sparsity > 0.8f ? Color::Yellow :
              sparsity > 0.5f ? Color::White   : Color::Green;
    return text(bar_str) | color(c);
}

static std::string shape_str(const std::vector<int64_t>& shape) {
    std::string s = "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        s += std::to_string(shape[i]);
        if (i + 1 < shape.size()) s += ", ";
    }
    return s + "]";
}

// Format a signed percentage delta with sign
static std::string fmt_pct(float old_val, float new_val) {
    if (std::fabs(old_val) < 1e-9f) return "(n/a)";
    float pct = (new_val - old_val) / std::fabs(old_val) * 100.f;
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(0);
    if (pct >= 0) oss << "(+" << pct << "%)";
    else          oss << "(" << pct << "%)";
    return oss.str();
}

ftxui::Component make_metrics_panel(HookEngine& engine,
                                     bool& focused,
                                     const std::string& selected_layer,
                                     std::shared_ptr<std::optional<LayerPacket>> compare) {
    return Renderer([&engine, &focused, &selected_layer, compare]() -> Element {
        // Find the most recent packet for the selected layer
        LayerPacket pkt{};
        bool found = false;

        if (!selected_layer.empty()) {
            auto snap = engine.packets.snapshot();
            for (int i = static_cast<int>(snap.size()) - 1; i >= 0; --i) {
                if (snap[i].layer_name == selected_layer) {
                    pkt   = snap[i];
                    found = true;
                    break;
                }
            }
        }

        Elements rows;
        if (!found) {
            rows.push_back(text("  Select a layer in Panel 1 (Space)") | color(Color::GrayDark));
        } else {
            std::ostringstream max_str, mean_str, lat_str;
            max_str  << std::fixed << std::setprecision(3) << pkt.max_val;
            mean_str << std::fixed << std::setprecision(4) << pkt.mean;
            lat_str  << std::fixed << std::setprecision(3) << pkt.latency_ms;

            Color lat_color = pkt.latency_ms > 5.f ? Color::Yellow : Color::Green;
            Color max_color = pkt.max_val > 6.f    ? Color::Red    : Color::White;

            rows = {
                hbox({ text("Layer   : ") | bold, text(pkt.layer_name) | color(Color::Cyan) }),
                hbox({ text("Type    : ") | bold, text(layer_type_str(pkt.type)) | color(Color::Yellow) }),
                hbox({ text("Shape   : ") | bold, text(shape_str(pkt.shape)) }),
                hbox({ text("Dtype   : ") | bold, text(dtype_str(pkt.dtype)) }),
                separator(),
                hbox({ text("Sparsity: ") | bold, sparsity_bar(pkt.sparsity) }),
                hbox({ text("Mean    : ") | bold, text(mean_str.str()) }),
                hbox({ text("Max     : ") | bold, text(max_str.str()) | color(max_color) }),
                separator(),
                hbox({
                    text("Latency : ") | bold,
                    text(lat_str.str() + "ms") | color(lat_color),
                    text(pkt.latency_ms > 5.f ? "  ⚠" : "  ✓") | color(lat_color),
                }),
                hbox({ text("Device  : ") | bold, text(device_str(pkt.device)) }),
            };

            // Comparison section (C key)
            if (compare && compare->has_value()) {
                const LayerPacket& cmp = compare->value();

                rows.push_back(separator());
                rows.push_back(text("── Comparison (C to update) ──") | color(Color::GrayLight));

                // Latency comparison
                {
                    float old_v = cmp.latency_ms;
                    float new_v = pkt.latency_ms;
                    std::ostringstream oss;
                    oss << std::fixed << std::setprecision(2)
                        << old_v << "ms → " << new_v << "ms  "
                        << fmt_pct(old_v, new_v);
                    bool worse = new_v > old_v;
                    Color dc = worse ? Color::Red : Color::Green;
                    rows.push_back(hbox({
                        text("Latency : ") | bold,
                        text(oss.str()) | color(dc),
                        text(worse ? " ⚠" : "") | color(Color::Yellow),
                    }));
                }
                // Sparsity comparison
                {
                    float old_v = cmp.sparsity;
                    float new_v = pkt.sparsity;
                    std::ostringstream oss;
                    oss << std::fixed << std::setprecision(0)
                        << (old_v * 100.f) << "% → " << (new_v * 100.f) << "%  "
                        << fmt_pct(old_v, new_v);
                    bool worse = new_v > old_v;  // higher sparsity may be a concern
                    Color dc = worse ? Color::Red : Color::Green;
                    rows.push_back(hbox({
                        text("Sparsity: ") | bold,
                        text(oss.str()) | color(dc),
                        text(worse ? " ⚠" : "") | color(Color::Yellow),
                    }));
                }
                // Max activation comparison
                {
                    float old_v = cmp.max_val;
                    float new_v = pkt.max_val;
                    std::ostringstream oss;
                    oss << std::fixed << std::setprecision(2)
                        << old_v << " → " << new_v << "  "
                        << fmt_pct(old_v, new_v);
                    bool worse = new_v > old_v;
                    Color dc = worse ? Color::Red : Color::Green;
                    rows.push_back(hbox({
                        text("Max     : ") | bold,
                        text(oss.str()) | color(dc),
                        text(worse ? " ✖" : "") | color(Color::Red),
                    }));
                }
                // Mean comparison
                {
                    float old_v = cmp.mean;
                    float new_v = pkt.mean;
                    std::ostringstream oss;
                    oss << std::fixed << std::setprecision(4)
                        << old_v << " → " << new_v << "  "
                        << fmt_pct(old_v, new_v);
                    bool worse = std::fabs(new_v) > std::fabs(old_v);
                    Color dc = worse ? Color::Red : Color::Green;
                    rows.push_back(hbox({
                        text("Mean    : ") | bold,
                        text(oss.str()) | color(dc),
                    }));
                }
            }
        }

        auto bcolor = focused ? Color::Cyan : Color::GrayDark;
        return window(
            text("── 4. RUNTIME METRICS INSPECTOR") | color(bcolor),
            vbox(std::move(rows)) | flex
        );
    });
}
