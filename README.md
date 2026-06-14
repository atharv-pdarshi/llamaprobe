# llamaprobe

A C++ diagnostic tool that non-invasively hooks into local LLM inference via llama.cpp, captures real-time intermediate states (activations, layer latencies, attention matrices, anomalies), and presents everything in an interactive 5-panel terminal UI.

```
 [Tab] Cycle Focus  [Space] Select Layer  [P] Pause  [Q] Quit  [?] Help
╔══ 1. MODEL TOPOLOGY ════════════╗┌── 2. LIVE PACKET STREAM ──────────────────────┐
║ ▼ tinyllama                     ║│  ID   │ TIMESTAMP │ LAYER TYPE    │ DEVICE    │
║   ● token_embd                  ║├───────┼───────────┼───────────────┼───────────┤
║   ▼ blk                         ║│  104  │ 00.110    │ Attn (Self)   │ CPU       │
║    ▼ 0  [Active Target]         ║│  105  │ 00.114    │ MLP (SwiGLU)  │ CPU       │
║      ● 0.attn_norm              ║│  106  │ 00.119    │ LayerNorm     │ CPU       │
║      ● 0.attn                   ║│  107  │ 00.122    │ MLP (SwiGLU)  │ CPU       │
║      ● 0.ffn_norm               ║└───────┴───────────┴───────────────┴───────────┘
╚════════════ [j/k Navigate] ═════╝
┌── 3. ATTENTION MATRIX VISUALIZER (HEAD 0) ────────────────────────────────────────┐
│        [the]  [cat]  [sat]  [on]  [mat]                                           │
│ [the]    █      ░      ░      ░      ░    HEAD 1/8  [H/L] Head  [h/j/k/l] Pan    │
│ [cat]    ▒      █      ░      ░      ░    [+/-] Contrast  [F] Fullscreen          │
│ [sat]    ░      ▒      █      ░      ░                                            │
└───────────────────────────────────────────────────────────────────────────────────┘
┌── 4. RUNTIME METRICS INSPECTOR ──────────┐┌── 5. NUMERICAL ANOMALY LEDGER ───────┐
│ Layer   : blk.0.attn                     ││ 00.114 ⚠  Max activation 7.2 > 6.0  │
│ Shape   : [1, 32, 2048]  Dtype: float16  ││ 00.128 ℹ  High Sparsity blk.3 > 80% │
│ Sparsity: ████████░░░░ 54.2%            ││ 00.201 ℹ  Slow layer blk.5.ffn 8ms  │
│ Mean    : 0.0023   Max: 4.871            ││                                      │
│ Latency : 1.14ms                         ││                                      │
└──────────────────────────────────────────┘└──────────────────────────────────────┘
```

## Features

- **Non-invasive** — hooks `ggml_backend_sched_eval_callback`; zero changes to llama.cpp or the model
- **Panel 1 — Model Topology**: collapsible layer tree built live from inference, `j/k` navigation, `Space` to pin a layer as the metrics target
- **Panel 2 — Live Packet Stream**: scrolling table of every captured tensor with timestamp, layer type, and compute device; anomalous rows highlighted red
- **Panel 3 — Attention Matrix Visualizer**: block-character heatmap (`█ ▓ ▒ ░ ·`), pannable viewport, adjustable contrast, head cycling
- **Panel 4 — Runtime Metrics Inspector**: shape, dtype, sparsity bar, mean, max, and latency delta for the selected layer
- **Panel 5 — Anomaly Ledger**: timestamped log of NaN/Inf, activation outliers, high sparsity, and latency spikes with ⚠ ✖ ℹ severity icons

## Keybindings

| Key | Action |
|-----|--------|
| `Tab` / `Shift+Tab` | Cycle panel focus forward / backward |
| `j` / `k` | Navigate up / down in focused panel |
| `h` / `l` | Pan attention matrix left / right |
| `H` / `L` | Cycle attention head |
| `Space` | Select layer as metrics target (Panel 1) |
| `F` | Toggle attention matrix fullscreen |
| `+` / `-` | Increase / decrease attention contrast |
| `P` | Pause / resume live capture |
| `Q` | Quit |
| `?` | Toggle keybinding help overlay |

## Build

**Requirements:** CMake 3.20+, a C++17 compiler, Git.

```bash
# Clone with submodules (llama.cpp + FTXUI)
git clone --recurse-submodules https://github.com/atharv-pdarshi/llamaprobe
cd llamaprobe

# Build
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
# Binary: build/llamaprobe
```

## Download a model

llamaprobe works with any GGUF model. TinyLlama 1.1B is recommended for development (fast, ~600 MB):

```bash
mkdir -p models
wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
     -O models/tinyllama.gguf
```

For a more interesting demo, use Llama 3.2 3B:

```bash
wget https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf \
     -O models/llama3.2-3b.gguf
```

## Run

```bash
# Basic run — interactive TUI streams live inference data
./build/llamaprobe --model models/tinyllama.gguf --prompt "Explain transformers in 3 sentences"

# With a longer prompt to stress the attention visualizer
./build/llamaprobe --model models/llama3.2-3b.gguf \
    --prompt "The history of neural networks begins in the 1940s when Warren McCulloch and Walter Pitts..."
```

## Run tests

```bash
cmake --build build --target test_ring_buffer test_metrics test_anomaly test_topology
ctest --test-dir build --output-on-failure
```

## Anomaly detection rules

| Condition | Severity |
|-----------|----------|
| `max_val > 6.0` in float16 layers | ⚠ Warning |
| `sparsity > 80%` | ℹ Info |
| `latency > 3× rolling average` | ⚠ Warning |
| NaN or Inf anywhere in tensor | ✖ Critical |

Thresholds are defined in `src/hook_engine.hpp` (`thresh_max_activation`, `thresh_sparsity`, `thresh_latency_spike`).

## Project structure

```
llamaprobe/
├── CMakeLists.txt
├── src/
│   ├── main.cpp              # CLI arg parsing, llama context setup, inference loop
│   ├── hook_engine.hpp/.cpp  # ggml callback, tensor capture, anomaly flagging
│   ├── metrics.hpp/.cpp      # sparsity / mean / max / latency computations
│   ├── anomaly_detector.hpp  # AnomalyEvent types and severity icons
│   ├── ring_buffer.hpp       # thread-safe fixed-size circular buffer
│   ├── topology.hpp/.cpp     # layer name → collapsible tree builder
│   ├── session.hpp/.cpp      # session recording / replay (Phase 4)
│   └── tui/
│       ├── app.hpp/.cpp           # layout, focus routing, global keys
│       ├── topology_panel.hpp/.cpp
│       ├── stream_panel.hpp/.cpp
│       ├── attention_panel.hpp/.cpp
│       ├── metrics_panel.hpp/.cpp
│       └── anomaly_panel.hpp/.cpp
├── tests/
│   ├── test_ring_buffer.cpp
│   ├── test_topology.cpp
│   ├── test_metrics.cpp
│   └── test_anomaly.cpp
├── models/    # put .gguf files here (gitignored)
├── sessions/  # exported JSON sessions
└── third_party/
    ├── llama.cpp/   (submodule)
    ├── ftxui/       (submodule)
    └── nlohmann/    (header-only JSON)
```
