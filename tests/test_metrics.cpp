#include "../src/metrics.hpp"
#include "ggml.h"
#include <cassert>
#include <cmath>

static void test_dtype_detection() {
    // dtype_of on null returns Other
    assert(MetricsComputer::dtype_of(nullptr) == DType::Other);
}

static void test_latency_delta() {
    MetricsComputer mc;
    mc.reset();

    float d1 = mc.latency_delta_ms("layer.0");
    assert(d1 == 0.f);  // first call — no previous timestamp

    // small sleep then second call should give positive delta
    volatile int x = 0;
    for (int i = 0; i < 1000000; ++i) x++;  // busy wait ~few ms

    float d2 = mc.latency_delta_ms("layer.0");
    assert(d2 >= 0.f);

    // different layer — independent clock
    float d3 = mc.latency_delta_ms("layer.1");
    assert(d3 == 0.f);
}

static void test_elapsed_us() {
    MetricsComputer mc;
    mc.reset();
    volatile int x = 0;
    for (int i = 0; i < 500000; ++i) x++;
    uint64_t elapsed = mc.elapsed_us();
    assert(elapsed > 0);
}

int main() {
    test_dtype_detection();
    test_latency_delta();
    test_elapsed_us();
    return 0;
}
