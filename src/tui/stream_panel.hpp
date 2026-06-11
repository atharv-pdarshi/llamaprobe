#pragma once

#include "../hook_engine.hpp"
#include "ftxui/component/component.hpp"

// Panel 2 — Live Packet Stream
// Scrollable table: ID | TIMESTAMP | LAYER TYPE | DEVICE
ftxui::Component make_stream_panel(HookEngine& engine, bool& focused);
