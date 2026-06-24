#include "topology_panel.hpp"

#include "ftxui/component/component.hpp"
#include "ftxui/component/event.hpp"
#include "ftxui/dom/elements.hpp"
#include "ftxui/screen/color.hpp"
#include "ftxui/screen/terminal.hpp"

#include <algorithm>
#include <vector>

using namespace ftxui;

// Flatten the tree into a list of (depth, node*) for rendering
struct FlatNode {
    int              depth;
    const LayerNode* node;
};

static void flatten(const LayerNode& node, int depth,
                    std::vector<FlatNode>& out) {
    out.push_back({depth, &node});
    if (node.expanded)
        for (const auto& child : node.children)
            flatten(child, depth + 1, out);
}

static Color node_color(LayerType t) {
    switch (t) {
        case LayerType::Attention:  return Color::Cyan;
        case LayerType::MLP:        return Color::Yellow;
        case LayerType::LayerNorm:  return Color::GrayLight;
        case LayerType::Embedding:  return Color::Green;
        case LayerType::Output:     return Color::Magenta;
        default:                    return Color::White;
    }
}

ftxui::Component make_topology_panel(HookEngine& engine,
                                      bool& focused,
                                      std::string& selected_layer) {
    auto cursor        = std::make_shared<int>(0);
    auto cached_tree   = std::make_shared<LayerNode>();
    auto known_layers  = std::make_shared<std::vector<std::string>>();

    return Renderer([&engine, &focused, &selected_layer,
                     cursor, cached_tree, known_layers]() -> Element {
        // Rebuild tree when new layers appear
        auto packets = engine.packets.snapshot();
        std::vector<std::string> names;
        for (const auto& p : packets) names.push_back(p.layer_name);
        std::sort(names.begin(), names.end());
        names.erase(std::unique(names.begin(), names.end()), names.end());

        if (names != *known_layers) {
            *known_layers = names;
            *cached_tree  = build_topology(names);
        }

        std::vector<FlatNode> flat;
        flatten(*cached_tree, 0, flat);

        int total   = static_cast<int>(flat.size());
        int visible = std::max(4, ftxui::Terminal::Size().dimy - 20);
        *cursor     = std::clamp(*cursor, 0, std::max(0, total - 1));
        int start   = std::max(0, *cursor - visible / 2);
        int end     = std::min(total, start + visible);

        Elements rows;
        for (int i = start; i < end; ++i) {
            const auto& fn   = flat[i];
            const auto* node = fn.node;
            bool is_cursor   = (i == *cursor);
            bool is_selected = (node->name == selected_layer);

            std::string indent(fn.depth * 2, ' ');
            std::string prefix;
            if (!node->children.empty())
                prefix = node->expanded ? "▼ " : "► ";
            else
                prefix = "● ";

            std::string label = indent + prefix + node->name;
            if (is_selected) label += " [Active Target]";

            Color c = node_color(node->type);
            auto  e = text(label) | color(c);
            if (is_cursor) e = e | inverted;
            rows.push_back(e);
        }

        if (flat.empty())
            rows.push_back(text("  Waiting for inference...") | color(Color::GrayDark));

        auto footer   = text(" [j/k Navigate | Space Select] ") | color(Color::GrayDark);
        auto bcolor   = focused ? Color::Cyan : Color::GrayDark;

        return window(
            text("── 1. MODEL TOPOLOGY") | color(bcolor),
            vbox({vbox(std::move(rows)) | flex, separator(), footer})
        );
    })
    | CatchEvent([cursor, &selected_layer, cached_tree](Event e) -> bool {
        if (e == Event::Character('j')) {
            (*cursor)++;
            return true;
        }
        if (e == Event::Character('k')) {
            (*cursor)--;
            return true;
        }
        if (e == Event::Character(' ')) {
            // Flatten and select node under cursor
            std::vector<FlatNode> flat;
            flatten(*cached_tree, 0, flat);
            int idx = std::clamp(*cursor, 0, (int)flat.size() - 1);
            if (!flat.empty()) {
                auto* node = const_cast<LayerNode*>(flat[idx].node);
                // Toggle expand/collapse for parents
                if (!node->children.empty())
                    node->expanded = !node->expanded;
                else
                    selected_layer = node->name;
            }
            return true;
        }
        return false;
    });
}