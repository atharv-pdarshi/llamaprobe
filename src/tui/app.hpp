#pragma once

#include "../hook_engine.hpp"
#include <string>

// Launch the full interactive TUI. Blocks until user presses Q.
// If record_path is non-empty, recording starts immediately.
void run_tui(HookEngine& engine, const std::string& record_path = "");