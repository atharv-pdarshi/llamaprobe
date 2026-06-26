cmake_minimum_required(VERSION 3.20)
project(llamaprobe LANGUAGES CXX)

set(BUILD_SHARED_LIBS OFF)

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
    src/config.cpp
    src/tui/app.cpp
    src/tui/topology_panel.cpp
    src/tui/stream_panel.cpp
    src/tui/attention_panel.cpp
    src/tui/metrics_panel.cpp
    src/tui/anomaly_panel.cpp
    src/tui/timeline_panel.cpp
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

foreach(test_name ring_buffer anomaly)
    add_executable(test_${test_name} tests/test_${test_name}.cpp)
    target_include_directories(test_${test_name} PRIVATE src/ third_party/nlohmann)
    target_link_libraries(test_${test_name} PRIVATE llama ggml)
    add_test(NAME ${test_name} COMMAND test_${test_name})
endforeach()

add_executable(test_metrics tests/test_metrics.cpp src/metrics.cpp)
target_include_directories(test_metrics PRIVATE
    src/ third_party/nlohmann
    third_party/llama.cpp/include
    third_party/llama.cpp/ggml/include)
target_link_libraries(test_metrics PRIVATE llama ggml)
add_test(NAME metrics COMMAND test_metrics)

add_executable(test_topology tests/test_topology.cpp src/topology.cpp)
target_include_directories(test_topology PRIVATE src/ third_party/nlohmann)
target_link_libraries(test_topology PRIVATE llama ggml)
add_test(NAME topology COMMAND test_topology)

add_executable(test_integration tests/test_integration.cpp src/session.cpp)
target_include_directories(test_integration PRIVATE src/ third_party/nlohmann)
target_link_libraries(test_integration PRIVATE llama ggml)
add_test(NAME integration COMMAND test_integration)
