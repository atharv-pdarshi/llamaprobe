#include "attention_panel.hpp"

#include "ftxui/component/component.hpp"
#include "ftxui/component/event.hpp"
#include "ftxui/dom/elements.hpp"
#include "ftxui/screen/color.hpp"
#include "ftxui/screen/terminal.hpp"

#include <algorithm>
#include <sstream>

using namespace ftxui;

// Map a weight value [0, 1] to a block character with contrast scaling
static std::string weight_char(float v, float contrast) {
    v = std::clamp(v * contrast, 0.f, 1.f);
    if (v < 0.15f) return " ";
    if (v < 0.35f) return "░";
    if (v < 0.55f) return "▒";
    if (v < 0.75f) return "▓";
    return "█";
}

static Color weight_color(float v, float contrast) {
    v = std::clamp(v * contrast, 0.f, 1.f);
    if (v < 0.3f) return Color::GrayDark;
    if (v < 0.6f) return Color::GrayLight;
    if (v < 0.85f) return Color::White;
    return Color::Cyan;
}

ftxui::Component make_attention_panel(HookEngine& engine, bool& focused) {
    struct State {
        int   head       = 0;
        int   pan_row    = 0;
        int   pan_col    = 0;
        float contrast   = 1.5f;
        bool  fullscreen = false;
    };
    auto state = std::make_shared<State>();

    return Renderer([&engine, &focused, state]() -> Element {
        auto caps = engine.attention.snapshot();

        if (caps.empty()) {
            auto bcolor = focused ? Color::Cyan : Color::GrayDark;
            return window(
                text("── 3. ATTENTION MATRIX VISUALIZER") | color(bcolor),
                text("  Waiting for attention data...") | color(Color::GrayDark)
            );
        }

        // Find latest capture for selected head
        const AttentionCapture* cap = nullptr;
        for (int i = static_cast<int>(caps.size()) - 1; i >= 0; --i) {
            if (caps[i].head_idx == static_cast<uint32_t>(state->head)) {
                cap = &caps[i];
                break;
            }
        }
        if (!cap) cap = &caps.back();

        const auto& mat    = cap->matrix;
        const auto& tokens = cap->tokens;
        int seq_len        = static_cast<int>(mat.size());
        int total_heads    = static_cast<int>(cap->total_heads);

        // Viewport
        auto [tw, th] = ftxui::Terminal::Size();
        int vw = state->fullscreen ? tw - 12 : std::min(12, tw - 20);
        int vh = state->fullscreen ? th - 8  : std::min(8,  th - 30);
        vw = std::max(4, vw);
        vh = std::max(2, vh);
        state->pan_row = std::clamp(state->pan_row, 0, std::max(0, seq_len - vh));
        state->pan_col = std::clamp(state->pan_col, 0, std::max(0, seq_len - vw));

        // Token header row
        Elements header_cells = { text("        ") };
        for (int c = state->pan_col; c < std::min(seq_len, state->pan_col + vw); ++c) {
            std::string tok = (c < (int)tokens.size()) ? tokens[c] : "?";
            if (tok.size() > 5) tok = tok.substr(0, 4) + ".";
            header_cells.push_back(text(" [" + tok + "]") | color(Color::GrayLight));
        }
        Elements rows = { hbox(std::move(header_cells)) };
        rows.push_back(separator());

        // Matrix rows
        for (int r = state->pan_row; r < std::min(seq_len, state->pan_row + vh); ++r) {
            Elements cells;
            // Row label
            std::string tok = (r < (int)tokens.size()) ? tokens[r] : "?";
            if (tok.size() > 5) tok = tok.substr(0, 4) + ".";
            cells.push_back(text("[" + tok + "] ") | color(Color::GrayLight));

            for (int c = state->pan_col; c < std::min(seq_len, state->pan_col + vw); ++c) {
                float v = (r < (int)mat.size() && c < (int)mat[r].size()) ? mat[r][c] : 0.f;
                cells.push_back(
                    text(weight_char(v, state->contrast))
                    | color(weight_color(v, state->contrast))
                );
            }
            rows.push_back(hbox(std::move(cells)));
        }

        // Footer with controls
        std::string hint =
            " HEAD " + std::to_string(state->head + 1) + "/" + std::to_string(total_heads) +
            "  Viewport [" + std::to_string(state->pan_row) + "-" +
            std::to_string(std::min(seq_len, state->pan_row + vh) - 1) + "]" +
            "  Contrast:" + std::to_string(state->contrast).substr(0, 3) +
            "  [H/L] Head  [h/j/k/l] Pan  [+/-] Contrast  [F] Fullscreen";
        rows.push_back(separator());
        rows.push_back(text(hint) | color(Color::GrayDark));

        auto bcolor = focused ? Color::Cyan : Color::GrayDark;
        auto title  = text("── 3. ATTENTION MATRIX VISUALIZER (HEAD " +
                           std::to_string(state->head) + ")") | color(bcolor);

        auto content = vbox(std::move(rows));
        return window(title, std::move(content));
    })
    | CatchEvent([state, &engine](Event e) -> bool {
        auto caps = engine.attention.snapshot();
        int max_heads = caps.empty() ? 1 : (int)caps.back().total_heads;

        if (e == Event::Character('h')) { state->pan_col--; return true; }
        if (e == Event::Character('l')) { state->pan_col++; return true; }
        if (e == Event::Character('j')) { state->pan_row++; return true; }
        if (e == Event::Character('k')) { state->pan_row--; return true; }
        if (e == Event::Character('H')) { state->head = std::max(0, state->head - 1); return true; }
        if (e == Event::Character('L')) { state->head = std::min(max_heads - 1, state->head + 1); return true; }
        if (e == Event::Character('+')) { state->contrast = std::min(5.f, state->contrast + 0.25f); return true; }
        if (e == Event::Character('-')) { state->contrast = std::max(0.25f, state->contrast - 0.25f); return true; }
        if (e == Event::Character('f') || e == Event::Character('F')) {
            state->fullscreen = !state->fullscreen; return true;
        }
        return false;
    });
}