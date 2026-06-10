#pragma once

#include <array>
#include <mutex>
#include <vector>
#include <cstddef>

// Fixed-size thread-safe circular buffer.
// Push overwrites the oldest entry when full (no blocking, no allocation).
template<typename T, size_t N>
class RingBuffer {
    static_assert(N > 0, "RingBuffer size must be > 0");

public:
    void push(T item) {
        std::lock_guard<std::mutex> lock(mtx_);
        buf_[head_] = std::move(item);
        head_ = (head_ + 1) % N;
        if (count_ < N) ++count_;
        else tail_ = (tail_ + 1) % N;  // overwrite: advance tail
    }

    // Returns up to N items in insertion order (oldest → newest).
    std::vector<T> snapshot() const {
        std::lock_guard<std::mutex> lock(mtx_);
        std::vector<T> out;
        out.reserve(count_);
        for (size_t i = 0; i < count_; ++i)
            out.push_back(buf_[(tail_ + i) % N]);
        return out;
    }

    // Returns the most recent item, or false if empty.
    bool latest(T& out) const {
        std::lock_guard<std::mutex> lock(mtx_);
        if (count_ == 0) return false;
        out = buf_[(head_ + N - 1) % N];
        return true;
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mtx_);
        head_ = tail_ = count_ = 0;
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mtx_);
        return count_;
    }

    bool empty() const { return size() == 0; }

private:
    mutable std::mutex  mtx_;
    std::array<T, N>    buf_;
    size_t              head_  = 0;   // next write position
    size_t              tail_  = 0;   // oldest item position
    size_t              count_ = 0;   // number of valid items
};
