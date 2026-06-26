#pragma once
#include "../hook_engine.hpp"
#include "ftxui/component/component.hpp"

// Panel 6 — Per-Token Timeline
// Horizontal bar chart of per-token decode durations; j/k to scroll
ftxui::Component make_timeline_panel(HookEngine& engine, bool& focused);
