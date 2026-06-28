#include "attention_panel.hpp"

#include "ftxui/component/component.hpp"
#include "ftxui/component/event.hpp"
#include "ftxui/dom/elements.hpp"
#include "ftxui/screen/color.hpp"
#include "ftxui/screen/terminal.hpp"

#include <algorithm>
#include <string>

using namespace ftxui;

// Strip SentencePiece ▁ (U+2581 = 0xe2 0x96 0x81) then truncate to max_chars ASCII chars.
static std::string clean_token(const std::string& tok, int max_chars) {
    std::string s = tok;
    while (s.size() >= 3 &&
           (uint8_t)s[0] == 0xe2 && (uint8_t)s[1] == 0x96 && (uint8_t)s[2] == 0x81)
        s = s.substr(3);
    if (s.empty()) s = "sp";
    if ((int)s.size() > max_chars) s = s.substr(0, max_chars - 1) + ".";
    return s;
}

// 2-wide block character pair for attention weight value in [0, 1].
static std::string weight_chars(float v, float contrast) {
    v = std::clamp(v * contrast, 0.f, 1.f);
    if (v < 0.10f) return "  ";
    if (v < 0.30f) return "░░";
    if (v < 0.55f) return "▒▒";
    if (v < 0.75f) return "▓▓";
    return "██";
}

static Color weight_color(float v, float contrast) {
    v = std::clamp(v * contrast, 0.f, 1.f);
    if (v < 0.10f) return Color::GrayDark;
    if (v < 0.30f) return Color::RGB(60,  80, 120);
    if (v < 0.55f) return Color::RGB(80, 140, 200);
    if (v < 0.75f) return Color::White;
    return Color::Cyan;
}

ftxui::Component make_attention_panel(HookEngine& engine, bool& focused) {
    struct State {
        int   head       = 0;
        int   pan_row    = 0;
        int   pan_col    = 0;
        float contrast   = 2.0f;
        bool  fullscreen = false;
    };
    auto state = std::make_shared<State>();

    return Renderer([&engine, &focused, state]() -> Element {
        auto caps   = engine.attention.snapshot();
        auto bcolor = focused ? Color::Cyan : Color::GrayDark;

        if (caps.empty()) {
            return window(
                text("── 3. ATTENTION MATRIX VISUALIZER") | color(bcolor),
                text("  Waiting for attention data...") | color(Color::GrayDark)
            );
        }

        // Find latest capture for current head
        const AttentionCapture* cap = nullptr;
        for (int i = (int)caps.size() - 1; i >= 0; --i) {
            if (caps[i].head_idx == (uint32_t)state->head) { cap = &caps[i]; break; }
        }
        if (!cap) cap = &caps.back();

        const auto& mat    = cap->matrix;
        const auto& tokens = cap->tokens;
        int seq            = (int)mat.size();
        int total_heads    = (int)cap->total_heads;

        // ── Layout constants ──────────────────────────────────────────────────
        // Panel 3 is full-width in app.cpp (row2 = p3->Render() with no hbox partner).
        // COL_W=5: "[abc]" header = 5 ASCII chars ✓; cells use size(WIDTH,EQUAL,5) for
        // correct visual width even for multi-byte block characters (██ = 2 visual cols).
        const int LABEL_W   = 9;   // "[Expla]" right-aligned (7 chars) padded to 9
        const int COL_W     = 5;   // "[abc]" column header; cell constrained to 5 visual cols
        const int SIDEBAR_W = 32;

        auto [tw, th] = ftxui::Terminal::Size();
        int matrix_area_w = tw - LABEL_W - SIDEBAR_W - 3;  // 3 for borders/separators
        int vw = std::max(2, std::min(seq, matrix_area_w / COL_W));
        int vh = state->fullscreen ? th - 12 : std::min(9, std::max(2, th - 28));

        state->pan_row = std::clamp(state->pan_row, 0, std::max(0, seq - vh));
        state->pan_col = std::clamp(state->pan_col, 0, std::max(0, seq - vw));

        int row_end = std::min(seq, state->pan_row + vh);
        int col_end = std::min(seq, state->pan_col + vw);

        // ── Column header row ─────────────────────────────────────────────────
        // "Q\K" corner + one "[tok]" per visible key token
        Elements header_cells;
        {
            // Corner label right-aligned in LABEL_W (all ASCII, byte==visual)
            std::string corner = "Q\\Key";
            while ((int)corner.size() < LABEL_W) corner = " " + corner;
            header_cells.push_back(text(corner) | color(Color::GrayDark));
        }
        for (int c = state->pan_col; c < col_end; ++c) {
            // clean_token gives ≤3 ASCII chars; "[abc]" = exactly COL_W=5
            std::string tok = c < (int)tokens.size() ? clean_token(tokens[c], 3) : "?";
            std::string lbl = "[" + tok + "]";
            while ((int)lbl.size() < COL_W) lbl += " ";  // all ASCII, safe
            header_cells.push_back(text(lbl) | color(Color::GrayLight));
        }

        Elements matrix_rows;
        matrix_rows.push_back(hbox(std::move(header_cells)));
        matrix_rows.push_back(separator());

        // ── Matrix rows ───────────────────────────────────────────────────────
        for (int r = state->pan_row; r < row_end; ++r) {
            Elements cells;

            // Row label: clean_token ≤5 ASCII chars → "[expla]" ≤7 chars, right-align to LABEL_W
            std::string tok = r < (int)tokens.size() ? clean_token(tokens[r], 5) : "?";
            std::string lbl = "[" + tok + "]";
            while ((int)lbl.size() < LABEL_W) lbl = " " + lbl;
            cells.push_back(text(lbl) | color(Color::GrayLight));

            for (int c = state->pan_col; c < col_end; ++c) {
                float v = (r < (int)mat.size() && c < (int)mat[r].size())
                          ? mat[r][c] : 0.f;
                // Use size(WIDTH, EQUAL, COL_W) so FTXUI measures visual terminal columns,
                // not bytes — block chars like ██ are 6 bytes but only 2 terminal columns.
                cells.push_back(
                    text(weight_chars(v, state->contrast))
                    | color(weight_color(v, state->contrast))
                    | size(WIDTH, EQUAL, COL_W)
                );
            }
            matrix_rows.push_back(hbox(std::move(cells)));
        }

        // ── Right sidebar: viewport info + controls ───────────────────────────
        std::string vp = "Viewport Window: [" +
                         std::to_string(state->pan_row) + "-" +
                         std::to_string(row_end - 1) + "] x [" +
                         std::to_string(state->pan_col) + "-" +
                         std::to_string(col_end - 1) + "]";

        auto sidebar = vbox({
            text(" " + vp)                           | color(Color::GrayLight),
            separator(),
            text(" [Focus + F]: Open Fullscreen")    | color(Color::GrayDark),
            text(" [Arrows/hjkl]: Pan Matrix")       | color(Color::GrayDark),
            text(" [+/-]: Change Contrast")          | color(Color::GrayDark),
            text(" [H/L]: Prev/Next Head")           | color(Color::GrayDark),
            separator(),
            text(" Contrast: " + [&]{
                auto s = std::to_string(state->contrast);
                return s.substr(0, s.find('.') + 3);
            }() + "x")                               | color(Color::GrayLight),
            text(" Head: " + std::to_string(state->head + 1) +
                 "/" + std::to_string(total_heads))  | color(Color::GrayLight),
        }) | size(WIDTH, EQUAL, SIDEBAR_W);

        auto grid = hbox({
            vbox(std::move(matrix_rows)) | flex,
            separator(),
            sidebar,
        });

        auto title = text("── 3. ATTENTION MATRIX VISUALIZER (HEAD " +
                          std::to_string(state->head + 1) + "/" +
                          std::to_string(total_heads) + ")") | color(bcolor);
        return window(title, grid);
    })
    | CatchEvent([state, &engine](Event e) -> bool {
        auto caps     = engine.attention.snapshot();
        int max_heads = caps.empty() ? 1 : (int)caps.back().total_heads;

        if (e == Event::Character('h') || e == Event::ArrowLeft)
            { state->pan_col = std::max(0, state->pan_col - 1); return true; }
        if (e == Event::Character('l') || e == Event::ArrowRight)
            { state->pan_col++; return true; }
        if (e == Event::Character('j') || e == Event::ArrowDown)
            { state->pan_row++; return true; }
        if (e == Event::Character('k') || e == Event::ArrowUp)
            { state->pan_row = std::max(0, state->pan_row - 1); return true; }
        if (e == Event::Character('H'))
            { state->head = std::max(0, state->head - 1); return true; }
        if (e == Event::Character('L'))
            { state->head = std::min(max_heads - 1, state->head + 1); return true; }
        if (e == Event::Character('+'))
            { state->contrast = std::min(5.f, state->contrast + 0.25f); return true; }
        if (e == Event::Character('-'))
            { state->contrast = std::max(0.25f, state->contrast - 0.25f); return true; }
        if (e == Event::Character('f') || e == Event::Character('F'))
            { state->fullscreen = !state->fullscreen; return true; }
        return false;
    });
}
