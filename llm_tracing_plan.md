# LLM Instrumentation, Tracing, and Replay Platform
## Implementation Plan — 2 Member Team

---

## Project Summary

A C++ diagnostic tool that non-invasively hooks into local transformer model inference (via llama.cpp),
captures real-time intermediate states (activations, layer latencies, attention matrices, anomalies),
and presents everything in an interactive 5-panel TUI with vim-style navigation.

---

## Tech Stack

| Layer | Choice | Reason |
|---|---|---|
| Inference Backend | [llama.cpp](https://github.com/ggerganov/llama.cpp) | CPU-first, exposes `ggml_backend_eval_callback` for non-invasive hooks |
| TUI Library | [FTXUI](https://github.com/ArthurSonzogni/FTXUI) | Modern C++17, composable components, handles keyboard events cleanly |
| Build System | CMake 3.20+ | Industry standard, works with llama.cpp and FTXUI as submodules |
| Threading | std::thread + std::mutex + std::atomic | Producer-consumer between hook callbacks and TUI render thread |
| Model (dev/test) | TinyLlama 1.1B (GGUF, ~600MB) | Fits any CPU, fast iteration; upgrade to Llama 3.2 3B for demo |
| Serialization | nlohmann/json (header-only) | Session export and replay file format |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    llama.cpp Inference                   │
│   ggml_backend_sched_eval_callback fires per tensor op  │
└────────────────────────┬────────────────────────────────┘
                         │  (non-invasive, no model edits)
                         ▼
┌─────────────────────────────────────────────────────────┐
│                      Hook Engine  [A]                    │
│  - Identifies layer type (Attn, MLP, LayerNorm, Embed)  │
│  - Records: tensor shape, dtype, sparsity, latency      │
│  - Detects: compute device (CPU/CUDA), anomalies        │
│  - Emits: LayerPacket structs                           │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│               Thread-Safe Buffers  [B]                  │
│  RingBuffer<LayerPacket, 512>                           │
│  AttentionBuffer<AttentionCapture, 32>                  │
│  AnomalyLog<AnomalyEvent, 100>                          │
└────────────────────────┬────────────────────────────────┘
                         │  (read by TUI render thread)
                         ▼
┌─────────────────────────────────────────────────────────┐
│                      TUI Engine                          │
│  Panel 1: Topology [B]     Panel 2: Live Stream [A]     │
│  Panel 3: Attention [B]    Panel 4: Metrics [B]         │
│              Panel 5: Anomaly Ledger [A]                │
└─────────────────────────────────────────────────────────┘
```

---

## Core Data Structures

```cpp
// Every captured layer event is one of these
struct LayerPacket {
    uint32_t    id;
    uint64_t    timestamp_us;      // microseconds since inference start
    std::string layer_name;        // e.g. "layers.1.attn"
    LayerType   type;              // Attn, MLP, LayerNorm, Embed, Other
    ComputeDevice device;          // CPU, CUDA_0, CUDA_1 ...
    std::vector<int64_t> shape;    // tensor dimensions
    DType       dtype;             // float16, float32, int8 ...
    float       sparsity;          // fraction of near-zero values
    float       mean, max_val;     // activation statistics
    float       latency_ms;        // time delta from previous packet
    bool        is_anomaly;        // flagged by anomaly detector
};

// Separate struct for attention — larger data, separate buffer
struct AttentionCapture {
    uint32_t    layer_idx;
    uint32_t    head_idx;
    std::vector<std::string> tokens;
    std::vector<std::vector<float>> matrix;  // [seq_len x seq_len]
    uint64_t    timestamp_us;
};

// Fixed-size ring buffer — prevents RAM blowup
template<typename T, size_t N>
class RingBuffer {
    std::array<T, N> buf;
    std::atomic<size_t> head{0}, tail{0};
    mutable std::mutex mtx;
public:
    void push(T&& item);
    bool pop(T& item);
    std::vector<T> snapshot() const;  // for TUI read
};
```

---

## How the Hook Works (Non-Invasive)

llama.cpp exposes a callback in `llama_context_params`:

```cpp
// Set this before creating the llama context — zero model code changes
params.cb_eval = [](struct ggml_tensor* t, bool ask, void* user_data) -> bool {
    if (ask) return true;  // always allow evaluation
    auto* engine = static_cast<HookEngine*>(user_data);
    engine->on_tensor_computed(t);  // our capture point
    return true;
};
params.cb_eval_user_data = &hook_engine;
```

Inside `on_tensor_computed`, parse `t->name` to identify layer type:
- Name contains `"attn"` → `LayerType::Attention`
- Name contains `"ffn"` or `"mlp"` → `LayerType::MLP`
- Name contains `"norm"` → `LayerType::LayerNorm`
- Name contains `"token_embd"` → `LayerType::Embedding`

---

## TUI Layout — 5 Panels

```
 [Tab]: Cycle Focus  |  [Q]: Quit  |  [R]: Toggle Record  |  [E]: Export Session
╔══ 1. MODEL TOPOLOGY ════════════════╗┌── 2. LIVE PACKET STREAM ─────────────────────────────┐
║ ▼ tinyllama-1.1b                    ║│  ID  │ TIMESTAMP    │ LAYER TYPE   │ COMPUTE DEVICE   │
║   ► token_embd                      ║├──────┼──────────────┼──────────────┼──────────────────┤
║   ▼ layers                          ║│  104 │ 00:01.110    │ Attn (Self)  │ CPU              │
║    ▶ layers.0                       ║│  105 │ 00:01.114    │ MLP (SwiGLU) │ CPU              │
║    ▼ layers.1  [Active Target]      ║│  106 │ 00:01.119    │ Attn (Self)  │ CPU              │
║      ● layers.1.attn ───────────────╢│  107 │ 00:01.122    │ MLP (SwiGLU) │ CPU              │
║      ● layers.1.mlp                 ║│  108 │ 00:01.128    │ LayerNorm    │ CPU              │
║    ► layers.2                       ║└──────┴──────────────┴──────────────┴──────────────────┘
╚════════════════ [j/k Navigate] ═════╝
┌── 3. ATTENTION MATRIX VISUALIZER (HEAD 0) ────────────────────────────────────────────────────┐
│ Tokens: [the] [cat] [sat] [on] [mat]              Viewport: [0-5] x [0-5]                     │
│ [the]    ██    ░░    ░░    ░░    ░░    [F]: Fullscreen  [h/j/k/l]: Pan  [+/-]: Contrast        │
│ [cat]    ▒▒    ██    ░░    ░░    ░░    [H/L]: Cycle Head                                       │
│ [sat]    ░░    ▒▒    ██    ░░    ░░                                                            │
└───────────────────────────────────────────────────────────────────────────────────────────────┘
┌── 4. RUNTIME METRICS INSPECTOR ────────────┐┌── 5. NUMERICAL ANOMALY LEDGER ─────────────────┐
│ Layer   : layers.1.attn                    ││ 00:01.114 ⚠  Outlier: Layer 0 Max > 6.0        │
│ Shape   : [1, 32, 2048]    Dtype: float16  ││ 00:01.128 ⚠  High Sparsity: Layer 3 > 80%      │
│ Sparsity: ████████░░░░░ 54.2%             ││ 00:01.201 ℹ  Slow Layer: layers.5.mlp 8.3ms     │
│ Mean    : 0.0023   Max: 4.871             ││                                                  │
│ Latency : 1.142ms  (Normal)               ││                                                  │
└────────────────────────────────────────────┘└──────────────────────────────────────────────────┘
```

### Keybindings

| Key | Action |
|-----|--------|
| `Tab` | Cycle focus between panels |
| `j` / `k` | Navigate up/down in focused panel |
| `h` / `l` | Pan left/right in attention matrix |
| `Space` | Select layer as capture target in Panel 1 |
| `F` | Toggle fullscreen for attention matrix |
| `+` / `-` | Increase/decrease attention weight contrast |
| `H` / `L` | Cycle attention head |
| `R` | Toggle session recording |
| `E` | Export current session to JSON |
| `P` | Pause/resume live capture |
| `Q` | Quit |

---

## Work Division

Each member owns paired backend + TUI responsibilities so both contribute equally to systems code and visualization code.

---

### Member A

**Backend:**
- CMake project setup, llama.cpp + FTXUI as git submodules
- `HookEngine` class: `cb_eval` callback registration, tensor name parsing, layer type detection, device detection
- `MetricsComputer`: sparsity, mean, max, latency delta calculations from raw tensor data
- `AnomalyDetector`: threshold-based flagging (max > 6.0, sparsity > 80%, latency spike > 3x avg, NaN/Inf)
- Session recorder + replay engine: serialize/deserialize `LayerPacket`s to JSON, `--replay` flag

**TUI:**
- Panel 2: Live Packet Stream — scrollable table wired to real `RingBuffer` snapshot, color-coded by layer type
- Panel 5: Numerical Anomaly Ledger — scrollable timestamped log with ⚠ ✖ ℹ icons, auto-scroll on new events
- Status bar: inference speed (tokens/sec), recording indicator (● REC), pause state
- Help overlay: `?` key shows modal keybinding cheatsheet

**Tests:**
- `tests/test_metrics.cpp`
- `tests/test_anomaly.cpp`

**Deliverables:**
- `src/hook_engine.hpp / .cpp`
- `src/metrics.hpp / .cpp`
- `src/anomaly_detector.hpp / .cpp`
- `src/session.hpp / .cpp`
- `src/tui/stream_panel.hpp / .cpp`
- `src/tui/anomaly_panel.hpp / .cpp`

---

### Member B

**Backend:**
- `RingBuffer<T, N>` template: thread-safe circular buffer with mutex, `push()`, `snapshot()`
- `AttentionBuffer`: separate fixed-size buffer capturing attention weight matrices per head
- Model topology tree builder: parse llama layer names into a tree of `LayerNode` structs
- Thread-safe producer-consumer handoff between inference thread and TUI render thread

**TUI:**
- FTXUI app scaffold: overall layout, Tab focus cycling, keyboard event router
- Panel 1: Model Topology — recursive tree renderer with `▼ ► ●` symbols, j/k navigation, Space to set capture target
- Panel 3: Attention Matrix Visualizer — block-character heatmap (`█ ▓ ▒ ░ ·`), h/j/k/l pan, +/- contrast, H/L head cycling, F fullscreen
- Panel 4: Runtime Metrics Inspector — live stats for selected layer (shape, dtype, sparsity bar, mean, max, latency)

**Tests:**
- `tests/test_ring_buffer.cpp`
- `tests/test_topology.cpp`

**Deliverables:**
- `src/ring_buffer.hpp`
- `src/topology.hpp / .cpp`
- `src/tui/app.hpp / .cpp`
- `src/tui/topology_panel.hpp / .cpp`
- `src/tui/attention_panel.hpp / .cpp`
- `src/tui/metrics_panel.hpp / .cpp`

---

## Project Structure

```
llm-tracer/
├── CMakeLists.txt
├── README.md
├── third_party/
│   ├── llama.cpp/          (git submodule)
│   ├── ftxui/              (git submodule)
│   └── nlohmann_json/      (header-only, single file)
├── src/
│   ├── main.cpp
│   ├── hook_engine.hpp / .cpp
│   ├── ring_buffer.hpp
│   ├── metrics.hpp / .cpp
│   ├── anomaly_detector.hpp / .cpp
│   ├── topology.hpp / .cpp
│   ├── session.hpp / .cpp
│   └── tui/
│       ├── app.hpp / .cpp
│       ├── topology_panel.hpp / .cpp
│       ├── stream_panel.hpp / .cpp
│       ├── attention_panel.hpp / .cpp
│       ├── metrics_panel.hpp / .cpp
│       └── anomaly_panel.hpp / .cpp
├── models/                 (gitignored, put .gguf files here)
├── sessions/               (exported JSON sessions)
└── tests/
    ├── test_ring_buffer.cpp
    ├── test_metrics.cpp
    └── test_anomaly.cpp
```

---

## Phase-by-Phase Plan

### Phase 0 — Setup (Day 1-2) | Both members

- [ ] Create GitHub repo, add llama.cpp + FTXUI as git submodules
- [ ] Write `CMakeLists.txt` linking llama.cpp, FTXUI, and your sources
- [ ] Download TinyLlama 1.1B GGUF model, verify it runs via llama.cpp CLI
- [ ] Confirm `cb_eval` callback fires during inference (print tensor names to stdout)
- [ ] Agree on shared data structures (`LayerPacket`, `AttentionCapture`) and commit headers

**Exit criterion:** `./llm-tracer --model models/tinyllama.gguf --prompt "hello"` prints tensor names to terminal.

---

### Phase 1 — Data Pipeline (Day 3-7)

**Member A:**
- [ ] Implement `HookEngine::on_tensor_computed()` — parse tensor name, identify layer type, fill `LayerPacket`
- [ ] Implement `MetricsComputer` — sparsity (count |x| < epsilon), mean, max from raw `float*` tensor data
- [ ] Scaffold Panel 2 (Live Stream) as a scrolling table reading from a mock vector of `LayerPacket`s
- [ ] Color-code Panel 2 rows by layer type (cyan=Attn, yellow=MLP, white=Norm, dim=Embed)

**Member B:**
- [ ] Implement `RingBuffer<T, N>` with mutex and `snapshot()` — unit test overflow + concurrent push/pop
- [ ] Scaffold FTXUI app with 5 placeholder panels and Tab focus cycling
- [ ] Implement keyboard event router (Q quits, Tab cycles focus, j/k dispatch to focused panel)
- [ ] Highlight active panel border on focus

**Exit criterion:** Real tensor packets flow through ring buffer; TUI shows live mock data with Tab/j/k working.

---

### Phase 2 — Core Features (Day 8-14)

**Member A:**
- [ ] Implement `AnomalyDetector` — configurable thresholds, emit `AnomalyEvent` with timestamp + description
- [ ] Implement device detection: check `t->backend` in ggml tensor for CPU vs CUDA
- [ ] Wire Panel 2 to real `RingBuffer` snapshot (replace mock source)
- [ ] Implement Panel 5 (Anomaly Ledger) — scrollable timestamped log with ⚠ ✖ ℹ icons, auto-scroll

**Member B:**
- [ ] Build model topology tree: parse layer names from llama context into a tree of `LayerNode` structs
- [ ] Implement `AttentionBuffer` — capture attention weight tensors, reshape `float*` into 2D matrix per head
- [ ] Thread-safe producer-consumer handoff: inference thread pushes, TUI thread polls via snapshots
- [ ] Panel 1: Render topology tree with `▼ ► ●` symbols, j/k navigation, Space to set capture target

**Exit criterion:** Real inference data streams into Panel 1 and 2; topology tree shows actual model structure; anomalies log correctly.

---

### Phase 3 — Attention Visualizer + Metrics Panel (Day 15-21)

**Member A:**
- [ ] Implement session recorder: on `R` keypress serialize `LayerPacket`s to JSON via nlohmann/json
- [ ] Implement replay engine: `--replay sessions/foo.json` feeds TUI at original timestamps
- [ ] Add status bar: inference speed (tokens/sec), recording indicator (● REC), pause state

**Member B:**
- [ ] Panel 3: Render attention matrix as block heatmap:
  - Map float [0,1] → `{' ', '░', '▒', '▓', '█'}` with adjustable contrast via +/-
  - h/j/k/l to pan viewport across large matrices
  - H/L to cycle attention heads
  - F to toggle fullscreen (Panel 3 expands to full terminal)
- [ ] Panel 4: Live stats for selected layer — shape, dtype, sparsity bar (`████░░░░`), mean, max, latency delta
- [ ] Handle terminal resize gracefully across all panels

**Exit criterion:** Full 5-panel TUI with real data; attention matrix pannable and fullscreenable; replay works end-to-end.

---

### Phase 4 — Polish + Config (Day 22-28)

**Member A:**
- [ ] Config file support: CLI flags or `config.toml` for anomaly thresholds, ring buffer size, model path
- [ ] Performance check: measure tokens/sec with and without hooks, ensure overhead < 5%
- [ ] Integration tests: run inference on 3 prompts, verify no crashes, anomalies logged correctly

**Member B:**
- [ ] Help overlay: `?` key shows modal keybinding cheatsheet
- [ ] Smooth TUI refresh: only re-render panels with new data (FTXUI `ScreenInteractive::PostEvent`)
- [ ] `P` key to pause/resume live capture without stopping inference
- [ ] README: build instructions, model download steps, annotated screenshot of TUI

**Exit criterion:** Can record → exit → replay a session. Help overlay shows all keybindings. No visible render lag.

---

### Phase 5 — Testing + Demo Prep (Day 29-35)

- [ ] Unit tests for ring buffer, metrics, anomaly detector (catch2 or simple assert-based)
- [ ] Integration test: run inference on 3 prompts, verify no crashes, anomalies logged correctly
- [ ] Create demo script: run Llama 3.2 3B on a multi-sentence prompt, record session, show replay
- [ ] README with build instructions, model download steps, screenshot of TUI
- [ ] Performance check: hook overhead < 5% inference slowdown (measure tokens/sec with and without hooks)

---

## Bonus Features (Do in Phase 4-5 if time permits)

### 1. Per-Token Timeline View
Add a 6th panel (or tab within Panel 2) showing generation time per token as a horizontal bar chart.
Shows which tokens caused slow layers — very useful for spotting KV cache misses.

### 2. Layer Comparison Mode
Press `C` to "bookmark" the current stats for a layer. Run another prompt. Panel 4 shows a diff:
`Latency: 1.2ms → 3.4ms (+183%) ⚠`. Great for comparing prompts or quantization levels.

### 3. Quantization Comparison
If you have both `float16` and `q4_0` versions of the same model, add `--compare-model` flag.
Run both, show side-by-side sparsity and anomaly counts. Makes for a killer demo.

### 4. Export to Flamegraph
Write a script that converts the session JSON to a format readable by `speedscope` or `inferno`.
Gives a proper flamegraph of transformer layer execution. One extra file, big visual impact.

### 5. Configurable Anomaly Rules (YAML/TOML)
Let users define custom thresholds in a config file instead of hardcoding:
```toml
[anomaly]
max_activation_threshold = 6.0
sparsity_warning = 0.8
latency_spike_multiplier = 3.0
```

---

## Anomaly Detection Rules

| Anomaly Type | Condition | Severity | Icon |
|---|---|---|---|
| Activation Outlier | `max_val > 6.0` (float16) | Warning | ⚠ |
| High Sparsity | `sparsity > 0.80` | Info | ℹ |
| Latency Spike | `latency > 3 * rolling_avg` | Warning | ⚠ |
| NaN/Inf Detected | any `isnan` or `isinf` in tensor | Critical | ✖ |
| CPU Fallback | device changed from GPU to CPU mid-inference | Info | ℹ |
| Shape Mismatch | tensor shape unexpected for layer type | Warning | ⚠ |

---

## Build Instructions (for README)

```bash
# Clone with submodules
git clone --recurse-submodules https://github.com/yourteam/llm-tracer

# Build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Download a model (example)
wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
     -O ../models/tinyllama.gguf

# Run
./llm-tracer --model ../models/tinyllama.gguf --prompt "Explain transformers in 3 sentences"

# Run with session recording
./llm-tracer --model ../models/tinyllama.gguf --prompt "..." --record sessions/demo.json

# Replay a saved session (no inference needed)
./llm-tracer --replay sessions/demo.json
```

---

## Key References

- [llama.cpp `ggml_backend_sched_eval_callback`](https://github.com/ggerganov/llama.cpp/blob/master/ggml/include/ggml-backend.h)
- [FTXUI documentation + examples](https://github.com/ArthurSonzogni/FTXUI)
- [llama.cpp examples/simple](https://github.com/ggerganov/llama.cpp/tree/master/examples/simple) — starting point for inference loop
- [btop source](https://github.com/aristocratos/btop) — TUI inspiration
- [lazygit source](https://github.com/jesseduffield/lazygit) — keyboard navigation inspiration

---

## Timeline Summary

| Phase | Days | Focus |
|---|---|---|
| 0 | 1-2 | Setup, submodules, verify callback fires |
| 1 | 3-7 | Ring buffer, hook engine, TUI skeleton |
| 2 | 8-14 | Topology tree, real data pipeline, Panel 1+2+4 |
| 3 | 15-21 | Attention visualizer (Panel 3), anomaly ledger (Panel 5) |
| 4 | 22-28 | Replay, config, polish, help overlay |
| 5 | 29-35 | Tests, demo prep, README, performance check |

**Total: ~5 weeks.** Phases 0-3 cover all mandatory PS requirements.
Phases 4-5 cover replay + bonus features.
