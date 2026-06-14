#include "../src/ring_buffer.hpp"
#include <cassert>
#include <thread>
#include <vector>

static void test_basic_push_pop() {
    RingBuffer<int, 4> rb;
    assert(rb.empty());

    rb.push(1); rb.push(2); rb.push(3);
    assert(rb.size() == 3);

    auto snap = rb.snapshot();
    assert(snap.size() == 3);
    assert(snap[0] == 1 && snap[1] == 2 && snap[2] == 3);
}

static void test_overflow_overwrites_oldest() {
    RingBuffer<int, 3> rb;
    rb.push(1); rb.push(2); rb.push(3);
    rb.push(4);  // overwrites 1

    auto snap = rb.snapshot();
    assert(snap.size() == 3);
    assert(snap[0] == 2 && snap[1] == 3 && snap[2] == 4);
}

static void test_latest() {
    RingBuffer<int, 4> rb;
    rb.push(10); rb.push(20);
    int v = 0;
    assert(rb.latest(v) && v == 20);
}

static void test_concurrent_push_snapshot() {
    RingBuffer<int, 128> rb;
    std::thread producer([&] {
        for (int i = 0; i < 200; ++i) rb.push(i);
    });
    // Snapshot while producer is running — must not crash
    for (int i = 0; i < 20; ++i) {
        auto snap = rb.snapshot();
        (void)snap;
    }
    producer.join();
}

static void test_clear() {
    RingBuffer<int, 4> rb;
    rb.push(1); rb.push(2);
    rb.clear();
    assert(rb.empty());
    assert(rb.size() == 0);
}

int main() {
    test_basic_push_pop();
    test_overflow_overwrites_oldest();
    test_latest();
    test_concurrent_push_snapshot();
    test_clear();
    return 0;
}
