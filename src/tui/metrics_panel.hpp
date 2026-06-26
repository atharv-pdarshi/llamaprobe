#pragma once

#include "../hook_engine.hpp"
#include "../types.hpp"
#include "ftxui/component/component.hpp"
#include <memory>
#include <optional>
#include <string>

// Panel 4 — Runtime Metrics Inspector
// Shows live stats for the layer selected in Panel 1.
// compare: optional snapshot taken with C key for side-by-side diff.
ftxui::Component make_metrics_panel(HookEngine& engine,
                                     bool& focused,
                                     const std::string& selected_layer,
                                     std::shared_ptr<std::optional<LayerPacket>> compare);
