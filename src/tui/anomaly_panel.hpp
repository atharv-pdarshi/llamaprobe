#pragma once

#include "../hook_engine.hpp"
#include "ftxui/component/component.hpp"

// Panel 5 — Numerical Anomaly Ledger
// Scrollable timestamped log with ⚠ ✖ ℹ severity icons
ftxui::Component make_anomaly_panel(HookEngine& engine, bool& focused);
