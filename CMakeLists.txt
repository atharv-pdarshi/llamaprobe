cmake_minimum_required(VERSION 3.20)
project(llamaprobe LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ── llama.cpp ────────────────────────────────────────────────────────────────
set(LLAMA_BUILD_TESTS     OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES  OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER    OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/llama.cpp)

# ── FTXUI ────────────────────────────────────────────────────────────────────
set(FTXUI_BUILD_EXAMPLES  OFF CACHE BOOL "" FORCE)
set(FTXUI_BUILD_DOCS      OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/ftxui)

# ── Sources ──────────────────────────────────────────────────────────────────
set(SOURCES
    src/main.cpp
    src/hook_engine.cpp
    src/metrics.cpp
    src/anomaly_detector.cpp
    src/topology.cpp
    src/session.cpp
    src/tui/app.cpp
    src/tui/topology_panel.cpp
    src/tui/stream_panel.cpp
    src/tui/attention_panel.cpp
    src/tui/metrics_panel.cpp
    src/tui/anomaly_panel.cpp
)

add_executable(llamaprobe ${SOURCES})

target_include_directories(llamaprobe PRIVATE
    src/
    third_party/nlohmann
    third_party/llama.cpp/include
    third_party/llama.cpp/ggml/include
)

target_link_libraries(llamaprobe PRIVATE
    llama
    ggml
    ftxui::screen
    ftxui::dom
    ftxui::component
)

# ── Tests ────────────────────────────────────────────────────────────────────
enable_testing()

foreach(test_name ring_buffer metrics anomaly topology)
    add_executable(test_${test_name} tests/test_${test_name}.cpp)
    target_include_directories(test_${test_name} PRIVATE src/ third_party/nlohmann)
    target_link_libraries(test_${test_name} PRIVATE llama ggml)
    add_test(NAME ${test_name} COMMAND test_${test_name})
endforeach()
