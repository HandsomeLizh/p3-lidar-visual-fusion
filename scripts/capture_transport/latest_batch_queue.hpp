#pragma once
#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <optional>
#include <utility>

namespace sensor_capture_ros2 {
// One pending complete batch, in addition to the consumer's active batch.
// push never waits for the consumer and never splits a stereo pair.
template<class T> class LatestBatchQueue {
public:
    enum class Result { accepted, replaced, oversized, closed };
    Result push(T value, size_t bytes, size_t limit) {
        if (bytes > limit) return Result::oversized;
        std::lock_guard<std::mutex> lock(mutex_);
        if (closed_) return Result::closed;
        const bool replaced = pending_.has_value();
        pending_ = std::move(value);
        cv_.notify_one();
        return replaced ? Result::replaced : Result::accepted;
    }
    std::optional<T> pop() {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [&] { return closed_ || pending_.has_value(); });
        if (!pending_) return std::nullopt;
        auto value = std::move(pending_);
        pending_.reset();
        return value;
    }
    void close(bool discard = false) {
        std::lock_guard<std::mutex> lock(mutex_);
        closed_ = true;
        if (discard) pending_.reset();
        cv_.notify_all();
    }
    // Only after both old threads have been joined.
    void reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        pending_.reset();
        closed_ = false;
    }
private:
    std::mutex mutex_;
    std::condition_variable cv_;
    std::optional<T> pending_;
    bool closed_ = false;
};
}  // namespace sensor_capture_ros2
