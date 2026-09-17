#include "sensor_capture_ros2/sensor_capture_node.hpp"
#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>

namespace sensor_capture_ros2 {
namespace {
using Steady = std::chrono::steady_clock;
double seconds(Steady::duration elapsed) {
    return std::chrono::duration<double>(elapsed).count();
}
std::string sensorName(uint8_t type) {
    switch (type) {
        case SENSOR_RGB: return "rgb";
        case SENSOR_TOF: return "tof";
        case SENSOR_CPU_LIDAR: return "cpulidar";
        case SENSOR_EVENT: return "event";
        default: return "sensor_" + std::to_string(type);
    }
}
std::string countsJson(const std::map<uint8_t, int>& counts) {
    std::string out = "{";
    for (const auto& entry : counts) {
        if (out.size() > 1) out += ',';
        out += '"' + sensorName(entry.first) + "\":" + std::to_string(entry.second);
    }
    return out + '}';
}
}

void SensorCaptureNode::configurePublication() {
    parallel_publication_ = declare_parameter<bool>("parallel_publication", true);
    publication_max_age_sec_ = declare_parameter<double>("publication_max_age_sec", 3.0);
    const int64_t mib = declare_parameter<int64_t>("publication_batch_mib", 64);
    if (mib < 8 || mib > 256 || !std::isfinite(publication_max_age_sec_) ||
        publication_max_age_sec_ <= 0.0 || publication_max_age_sec_ > 30.0) {
        throw std::invalid_argument("publication_batch_mib must be 8..256; max age must be (0,30] seconds");
    }
    publication_batch_bytes_ = static_cast<size_t>(mib) * 1024 * 1024;
    RCLCPP_INFO(get_logger(), "Publication parallel=%d pending_batches=1 batch_limit_mib=%ld max_age_sec=%.2f",
        parallel_publication_, static_cast<long>(mib), publication_max_age_sec_);
}

void SensorCaptureNode::enqueuePublication(std::string metadata, int processed, double decode_seconds) {
    CompletedBatch batch;
    batch.items = std::move(batch_items_);
    batch.stamp = batch_publish_stamp_;  // immutable snapshot; never read the next batch's stamp
    batch.metadata = std::move(metadata);
    batch.processed = processed;
    batch.counts = batch_recv_counts_;
    batch.totals = total_recv_counts_;
    batch.requested = batch_request_steady_;
    batch.ready = Steady::now();
    batch.prepare_sec = seconds(batch.ready - batch.requested);
    batch.decode_sec = decode_seconds;
    batch.ready_age_sec = timestamp_mode_ == "meta_relative" ?
        (now().nanoseconds() - batch.stamp.nanoseconds()) / 1e9 : 0.0;
    size_t bytes = batch.metadata.capacity();
    for (const auto& item : batch.items) {
        bytes += item.file_data.capacity() + item.prepared_image.total() * item.prepared_image.elemSize();
    }
    batch.bytes = bytes;
    if (bytes > publication_batch_bytes_) {
        ++publication_oversized_;
        RCLCPP_ERROR(get_logger(), "Dropping oversized complete batch: %zu bytes", bytes);
        return;
    }
    if (!parallel_publication_) {
        publishCompleteBatch(batch);
        return;
    }
    const auto result = publication_queue_.push(std::move(batch), bytes, publication_batch_bytes_);
    if (result == LatestBatchQueue<CompletedBatch>::Result::replaced) {
        ++publication_replaced_;
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
            "Publisher busy: replaced oldest pending COMPLETE batch (total=%lu)",
            static_cast<unsigned long>(publication_replaced_.load()));
    }
}

void SensorCaptureNode::publicationLoop() {
    while (auto batch = publication_queue_.pop()) {
        if (!rclcpp::ok()) break;
        try {
            publishCompleteBatch(*batch);
        } catch (const std::exception& error) {
            ++publication_errors_;
            RCLCPP_ERROR(get_logger(), "Batch publication failed: %s", error.what());
        }
    }
}

void SensorCaptureNode::publishCompleteBatch(CompletedBatch& batch) {
    const auto started = Steady::now();
    const auto age = [&] {
        // Both clocks are checked: wall-clock adjustment cannot make queued old data fresh.
        const double elapsed = seconds(Steady::now() - batch.ready);
        return std::max(batch.ready_age_sec + elapsed,
            timestamp_mode_ == "meta_relative" ?
                (now().nanoseconds() - batch.stamp.nanoseconds()) / 1e9 : elapsed);
    };
    if (age() > publication_max_age_sec_ || age() < -0.1) {
        ++publication_stale_;
        RCLCPP_WARN(get_logger(), "Dropping stale complete batch %u age=%.3fs",
            batch.items.front().header.batch_id, age());
        return;
    }
    // Publish the stereo pair first. Topic writes themselves are not transactional;
    // the receiver still requires a complete timestamp-matched pair.
    std::stable_sort(batch.items.begin(), batch.items.end(), [](const BatchItem& a, const BatchItem& b) {
        return (a.header.sensor_type == SENSOR_RGB) > (b.header.sensor_type == SENSOR_RGB);
    });
    size_t sent = 0;
    double slowest_sec = 0.0;
    std::string slowest_key;
    for (const auto& item : batch.items) {
        if (!rclcpp::ok()) return;
        if (age() > publication_max_age_sec_) {
            ++publication_stale_;
            break;  // DDS may have blocked mid-batch; never send its remaining stale sensors.
        }
        const auto begin = Steady::now();
        publishSensorData(item.header, item.file_data, item.header.file_name, batch.stamp, item.prepared_image);
        const double elapsed = seconds(Steady::now() - begin);
        if (elapsed > slowest_sec) {
            slowest_sec = elapsed;
            slowest_key = makePublisherKey(item.header.sensor_type, item.header.camera_id);
        }
        ++sent;
    }
    if (!rclcpp::ok()) return;
    const double sensor_publish_sec = seconds(Steady::now() - started);
    // The original metadata is exposed unchanged, including any optional IMU fields.
    // It is metadata, not a validated sensor_msgs/Imu measurement.
    if (!batch.metadata.empty()) {
        std_msgs::msg::String meta;
        meta.data = batch.metadata;
        meta_pub_->publish(meta);
    }
    std::ostringstream status;
    status.precision(12);
    status << "{\"batch\":" << batch.processed << ",\"items\":" << batch.items.size()
        << ",\"batch_id\":" << batch.items.front().header.batch_id
        << ",\"stamp_ns\":" << batch.stamp.nanoseconds()
        << ",\"batch_counts\":" << countsJson(batch.counts)
        << ",\"total_counts\":" << countsJson(batch.totals)
        << ",\"transport\":{\"parallel\":" << (parallel_publication_ ? "true" : "false")
        << ",\"batch_bytes\":" << batch.bytes << ",\"published_items\":" << sent
        << ",\"complete\":" << (sent == batch.items.size() ? "true" : "false")
        << ",\"prepare_sec\":" << batch.prepare_sec << ",\"decode_sec\":" << batch.decode_sec
        << ",\"queue_sec\":" << seconds(started - batch.ready)
        << ",\"sensor_publish_sec\":" << sensor_publish_sec
        << ",\"slowest_sensor\":\"" << slowest_key << "\",\"slowest_sensor_sec\":" << slowest_sec
        << ",\"age_sec\":" << age() << ",\"replaced_batches\":" << publication_replaced_.load()
        << ",\"stale_batches\":" << publication_stale_.load()
        << ",\"oversized_batches\":" << publication_oversized_.load()
        << ",\"publication_errors\":" << publication_errors_.load() << "}}";
    std_msgs::msg::String message;
    message.data = status.str();
    batch_status_pub_->publish(message);
    RCLCPP_INFO(get_logger(), "Transport batch=%u ready=%.3fs decode=%.3fs queue=%.3fs publish=%.3fs age=%.3fs sent=%zu/%zu replaced=%lu",
        batch.items.front().header.batch_id, batch.prepare_sec, batch.decode_sec,
        seconds(started - batch.ready), seconds(Steady::now() - started), age(), sent, batch.items.size(),
        static_cast<unsigned long>(publication_replaced_.load()));
}
}  // namespace sensor_capture_ros2
