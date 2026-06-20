#pragma once

// Thresholds used by HookEngine::check_anomalies().
// Constructed with defaults; CLI flags and config files override fields before
// the engine starts.
struct AnomalyConfig {
    float max_activation  = 6.0f;   // max |x| before Warning
    float sparsity        = 0.80f;  // fraction of near-zeros before Info
    float latency_mult    = 3.0f;   // spike if latency > N * rolling average
};
