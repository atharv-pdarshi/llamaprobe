#pragma once

#include "../hook_engine.hpp"
#include "../topology.hpp"
#include "ftxui/component/component.hpp"
#include <string>

// Panel 1 — Model Topology Tree
// j/k to navigate, Space to set as active capture target
ftxui::Component make_topology_panel(HookEngine& engine,
                                      bool& focused,
                                      std::string& selected_layer);
