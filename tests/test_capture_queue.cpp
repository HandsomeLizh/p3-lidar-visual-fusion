#include "latest_batch_queue.hpp"
#include <cassert>
#include <chrono>
#include <future>
#include <iostream>
#include <memory>
#include <thread>
using sensor_capture_ros2::LatestBatchQueue;
using namespace std::chrono_literals;
struct Batch { int id; std::unique_ptr<int[]> pixels; };
int main() {
    LatestBatchQueue<Batch> q;
    using Result = LatestBatchQueue<Batch>::Result;
    assert(q.push({1, std::make_unique<int[]>(100)}, 400, 1024) == Result::accepted);
    auto active = q.pop();
    assert(active->id == 1);
    // A blocked consumer retains a single active batch. Producer never waits;
    // all intermediate pending batches are replaced as owned complete objects.
    auto producer = std::async(std::launch::async, [&] {
        for (int id = 2; id <= 10000; ++id) {
            auto result = q.push({id, std::make_unique<int[]>(100)}, 400, 1024);
            assert(result == (id == 2 ? Result::accepted : Result::replaced));
        }
    });
    assert(producer.wait_for(2s) == std::future_status::ready);
    producer.get();
    assert(q.push({10001, nullptr}, 1025, 1024) == Result::oversized);
    q.close();  // drain latest pending batch, then terminate
    assert(q.pop()->id == 10000);
    assert(!q.pop());
    assert(q.push({1, nullptr}, 0, 1024) == Result::closed);
    q.reset();
    auto sleeping = std::async(std::launch::async, [&] { return q.pop(); });
    assert(sleeping.wait_for(30ms) == std::future_status::timeout);
    q.close(true);
    assert(sleeping.wait_for(1s) == std::future_status::ready && !sleeping.get());
    for (int i = 0; i < 100; ++i) {
        q.reset();
        q.push({i, nullptr}, 0, 1024);
        q.close(true);
        assert(!q.pop());
    }
    std::cout << "PASS: blocked consumer, latest whole batch, byte cap, drain, discard, restart, wakeup\n";
}
