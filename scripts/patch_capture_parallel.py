"""Reproducible additions to the private capture copy; original P3 remains untouched."""
from pathlib import Path
import shutil


def patch_parallel(target):
    assets = Path(__file__).with_name('capture_transport')

    def replace(file, old, new):
        path = target / file
        text = path.read_text()
        if text.count(old) != 1:
            raise RuntimeError(f'Parallel capture patch no longer matches {file}: {old[:80]!r}')
        path.write_text(text.replace(old, new))

    h = 'include/sensor_capture_ros2/sensor_capture_node.hpp'
    c = 'src/sensor_capture_node.cpp'
    shutil.copy2(assets / 'latest_batch_queue.hpp', target / 'include/sensor_capture_ros2/latest_batch_queue.hpp')
    shutil.copy2(assets / 'capture_publication.cpp', target / 'src/capture_publication.cpp')
    replace('CMakeLists.txt', '  src/sensor_capture_node.cpp\n',
            '  src/sensor_capture_node.cpp\n  src/capture_publication.cpp\n')
    replace(h, '#include "sensor_capture_ros2/batch_timing.hpp"',
            '#include "sensor_capture_ros2/batch_timing.hpp"\n#include "sensor_capture_ros2/latest_batch_queue.hpp"')
    replace(h, '    int sock_fd_;', '''    struct CompletedBatch {
        std::vector<BatchItem> items;
        rclcpp::Time stamp{0, 0, RCL_ROS_TIME};
        std::string metadata;
        std::map<uint8_t, int> counts, totals;
        std::chrono::steady_clock::time_point requested, ready;
        double prepare_sec = 0.0, decode_sec = 0.0, ready_age_sec = 0.0;
        size_t bytes = 0;
        int processed = 0;
    };
    LatestBatchQueue<CompletedBatch> publication_queue_;
    std::thread publication_thread_;
    bool parallel_publication_ = true;
    double publication_max_age_sec_ = 3.0;
    size_t publication_batch_bytes_ = 64 * 1024 * 1024;
    std::chrono::steady_clock::time_point batch_request_steady_;
    std::atomic<uint64_t> publication_replaced_{0}, publication_stale_{0},
        publication_oversized_{0}, publication_errors_{0};
    void configurePublication();
    void enqueuePublication(std::string metadata, int processed, double decode_seconds);
    void publicationLoop();
    void publishCompleteBatch(CompletedBatch& batch);
    int sock_fd_;''')
    signature = 'const std::string & file_name, const cv::Mat & prepared = cv::Mat());'
    replace(h, signature,
            'const std::string & file_name, const rclcpp::Time & batch_stamp, const cv::Mat & prepared = cv::Mat());')
    replace(c, '    // ---- 根据 output_topics 动态创建发布者 ----',
            '    configurePublication();\n\n    // ---- 根据 output_topics 动态创建发布者 ----')
    replace(c, 'auto qos10 = rclcpp::QoS(10);', 'auto qos10 = rclcpp::QoS(1).transient_local();')
    replace(c, '"/sensor/batch_status", rclcpp::QoS(10));',
            '"/sensor/batch_status", rclcpp::QoS(1));')
    replace(c, '    running_ = true;\n    capture_thread_ =', '''    if (capture_thread_.joinable()) capture_thread_.join();
    if (publication_thread_.joinable()) publication_thread_.join();
    publication_queue_.reset();
    running_ = true;
    if (parallel_publication_) publication_thread_ = std::thread(&SensorCaptureNode::publicationLoop, this);
    capture_thread_ =''')
    replace(c, '    running_ = false;\n    queue_cv_.notify_all();',
            '    running_ = false;\n    publication_queue_.close(true);\n    queue_cv_.notify_all();')
    replace(c, '    disconnectFromServer();\n}',
            '    if (publication_thread_.joinable()) publication_thread_.join();\n    disconnectFromServer();\n}')
    replace(c, '            batch_request_stamp_ = this->now();',
            '            batch_request_steady_ = request_started;\n            batch_request_stamp_ = this->now();')
    replace(c, '    disconnectFromServer();\n    running_ = false;', '''    disconnectFromServer();
    publication_queue_.close(!rclcpp::ok());
    if (publication_thread_.joinable()) publication_thread_.join();
    running_ = false;''')
    replace(c, '    if (!recvExact(payload, payload_size)) return false;', '''    // Bound allocation before trusting a TCP size field (also applies to legacy mode).
    if (payload_size > publication_batch_bytes_ || payload_size < 4) return false;
    if (!recvExact(payload, payload_size)) return false;''')
    replace(c, '    uint32_t expected_count = 0;',
            '    uint32_t expected_count = 0;\n    size_t received_bytes = 0;')
    replace(c, '        if (cmd != CMD_DATA) continue;', '''        received_bytes += payload.size();
        if (received_bytes > 2 * publication_batch_bytes_) {
            RCLCPP_ERROR(get_logger(), "Wire batch exceeded memory budget");
            batch_error_ = true;
            batch_done_ = true;
            queue_cv_.notify_all();
            return;
        }
        if (cmd != CMD_DATA) continue;''')
    replace(c, '        if (!parseFileHeader(payload, header)) continue;', '''        if (!parseFileHeader(payload, header)) continue;
        if (payload.size() < 156 || header.file_size > payload.size() - 156 ||
            (header.file_name == "meta.json" && header.file_size > 1024 * 1024)) {
            batch_error_ = true;
            batch_done_ = true;
            queue_cv_.notify_all();
            return;
        }''')
    replace(c, 'if (is_png && !file_data.empty() &&',
            'if (is_png && file_data.size() >= PNG_MAGIC.size() &&')
    replace(c, '    int processed = 0;\n    std::string batch_metadata;',
            '    int processed = 0;\n    double decode_seconds = 0.0;\n    size_t retained_bytes = 0;\n    std::string batch_metadata;')
    replace(c, '                // meta.json 仅保存磁盘，不再发布 topic',
            '                // Original meta is published by the complete-batch publication worker.')
    replace(c, '                        item.prepared_image = cv::imdecode(cv::Mat(item.file_data, false), cv::IMREAD_UNCHANGED);', '''                        // Check PNG dimensions before allowing the decoder to allocate memory.
                        if (item.file_data.size() < 24) throw std::runtime_error("Truncated RGB PNG");
                        auto png_u32 = [&](size_t at) {
                            uint32_t value = 0;
                            for (size_t i = at; i < at + 4; ++i) value = (value << 8) | item.file_data[i];
                            return value;
                        };
                        const uint64_t pixels = uint64_t(png_u32(16)) * png_u32(20);
                        if (!pixels || pixels > publication_batch_bytes_ / 8)
                            throw std::runtime_error("Decoded RGB exceeds memory budget");
                        const auto decoding = std::chrono::steady_clock::now();
                        item.prepared_image = cv::imdecode(cv::Mat(item.file_data, false), cv::IMREAD_UNCHANGED);
                        decode_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - decoding).count();
                        if (item.prepared_image.empty()) throw std::runtime_error("RGB PNG decode failed");''')
    replace(c, '                    batch_items_.push_back(std::move(item));', '''                    retained_bytes += item.file_data.capacity() + item.prepared_image.total() * item.prepared_image.elemSize();
                    if (retained_bytes > publication_batch_bytes_) throw std::runtime_error("Decoded batch exceeds memory budget");
                    batch_items_.push_back(std::move(item));''')
    replace(c, '            RCLCPP_ERROR(this->get_logger(), "处理数据异常: %s", e.what());', '''            RCLCPP_ERROR(this->get_logger(), "处理数据异常: %s", e.what());
            // Never allow a failed left image to masquerade as a complete batch.
            batch_error_ = true;
            batch_done_ = true;
            queue_cv_.notify_all();
            if (sock_fd_ >= 0) shutdown(sock_fd_, SHUT_RDWR);''')
    path = target / c
    text = path.read_text()
    start = text.index('        for (auto & item : batch_items_) {')
    end = text.index('\n    batch_items_.clear();\n}', start)
    text = text[:start] + '        enqueuePublication(std::move(batch_metadata), processed, decode_seconds);\n    }\n' + text[end:]
    path.write_text(text)
    replace(c, 'const std::string & /*file_name*/, const cv::Mat & prepared)',
            'const std::string & /*file_name*/, const rclcpp::Time & batch_stamp, const cv::Mat & prepared)')
    replace(c, 'timestamp_mode_ == "meta_relative" ? batch_publish_stamp_ :',
            'timestamp_mode_ == "meta_relative" ? batch_stamp :')
    replace(c, '        RCLCPP_ERROR(this->get_logger(), "发布 %s 失败: %s", base_key.c_str(), e.what());',
            '        RCLCPP_ERROR(this->get_logger(), "发布 %s 失败: %s", base_key.c_str(), e.what());\n        throw; // Complete-batch worker records failure; do not report success.')
    # Keep thread exceptions inside the process, and wake the sibling on failure.
    replace(c, '''    receive_thread_ = std::thread(&SensorCaptureNode::receiveThread, this,
                                  batch_index, batch_folder);
    process_thread_ = std::thread(&SensorCaptureNode::processThread, this,
                                  batch_folder);''', '''    auto fail = [this](const std::exception& error) {
        RCLCPP_ERROR(get_logger(), "Capture worker exception: %s", error.what());
        batch_error_ = true;
        batch_done_ = true;
        if (sock_fd_ >= 0) shutdown(sock_fd_, SHUT_RDWR);
        queue_cv_.notify_all();
    };
    receive_thread_ = std::thread([this, batch_index, batch_folder, fail] {
        try { receiveThread(batch_index, batch_folder); } catch (const std::exception& error) { fail(error); }
    });
    process_thread_ = std::thread([this, batch_folder, fail] {
        try { processThread(batch_folder); } catch (const std::exception& error) { fail(error); }
    });''')
    # Distinguish point-cloud construction from blocking DDS writes.
    replace(c, '    if (!buildLidarPointCloud(data, msg)) {',
            '    const auto cloud_build_started = std::chrono::steady_clock::now();\n    if (!buildLidarPointCloud(data, msg)) {')
    replace(c, '''        pub->publish(msg);
        RCLCPP_INFO(this->get_logger(),
            "LiDAR 点云已发布: key=%s, %u points", key.c_str(), msg.width);''', '''        const auto cloud_write_started = std::chrono::steady_clock::now();
        pub->publish(msg);
        RCLCPP_INFO(this->get_logger(),
            "LiDAR 点云已发布: key=%s, %u points build=%.6fs dds_write=%.6fs", key.c_str(), msg.width,
            std::chrono::duration<double>(cloud_write_started - cloud_build_started).count(),
            std::chrono::duration<double>(std::chrono::steady_clock::now() - cloud_write_started).count());''')
