#pragma once

#include "../hook_engine.hpp"
#include "ftxui/component/component.hpp"

// Panel 3 — Attention Matrix Visualizer
// Block-character heatmap with pan, contrast, head cycling, fullscreen
ftxui::Component make_attention_panel(HookEngine& engine, bool& focused);
