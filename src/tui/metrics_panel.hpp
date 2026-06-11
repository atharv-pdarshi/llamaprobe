#pragma once

#include "../hook_engine.hpp"
#include "ftxui/component/component.hpp"
#include <string>

// Panel 4 — Runtime Metrics Inspector
// Shows live stats for the layer selected in Panel 1
ftxui::Component make_metrics_panel(HookEngine& engine,
                                     bool& focused,
                                     const std::string& selected_layer);
