// Galaxy2 dual-camera ROS2 acquisition node.
//
// Opens two Daheng GigE cameras via the official Galaxy SDK and publishes:
//   <ns>/Cam_Left/image_raw                 sensor_msgs/Image  BayerRG8 (bayer_rggr8)
//   <ns>/Cam_Left/image_raw/color           sensor_msgs/Image  RGB8 (rgb8)
//   <ns>/Cam_Left/image_raw/color/compressed sensor_msgs/CompressedImage  JPEG
//   <ns>/Cam_Left/image_raw/color/light      sensor_msgs/Image  RGB8 (缩放 1/8, 远程查看用)
//   <ns>/Cam_Left/camera_info               sensor_msgs/CameraInfo
//   (likewise for Cam_Right)
//
// The namespace defaults to "galaxy" but can be remapped to "daheng" for a
// drop-in replacement of the legacy Aravis-based node.
//
// ros2 run galaxy2 galaxy2_node --ros-args -r __ns:=/Car/galaxy
//
// Parameters: cam1_ip, cam2_ip, cam1_frame_id, cam2_frame_id, frame_rate,
//             exposure_time, publish_raw, publish_color, publish_color_compressed,
//             jpeg_quality, trigger_mode, trigger_source, trigger_activation,
//             enable_save, save_dir.
//
// Topics (subscriptions):
//   ~/set_camera_params  std_msgs/Float64MultiArray  [exposure_time_us, gain_db]
//     Dynamically update camera exposure and gain at runtime.

#include <memory>
#include <string>
#include <vector>
#include <array>
#include <algorithm>
#include <stdexcept>
#include <filesystem>
#include <cstdio>
#include <ctime>
#include <chrono>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/compressed_image.hpp"
#include "std_msgs/msg/header.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"

#include <mutex>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "galaxy2/galaxy_camera.h"

namespace galaxy2 {

class GalaxyNode : public rclcpp::Node {
public:
  GalaxyNode() : rclcpp::Node("galaxy_node") {
    // ---- parameters ----
    cam1_ip_ = declare_parameter<std::string>("cam1_ip", "192.168.19.10");
    cam2_ip_ = declare_parameter<std::string>("cam2_ip", "192.168.19.11");
    topic_prefix_ = declare_parameter<std::string>("topic_prefix", "Car/T5/");
    cam1_name_ = declare_parameter<std::string>("cam1_name", "Cam_Left");
    cam2_name_ = declare_parameter<std::string>("cam2_name", "Cam_Right");
    topic_image_raw_ = declare_parameter<std::string>("topic_image_raw", "image_raw");
    topic_image_color_ = declare_parameter<std::string>("topic_image_color", "image_raw/color");
    topic_image_compressed_ = declare_parameter<std::string>("topic_image_compressed", "image_raw/color/compressed");
    topic_image_light_ = declare_parameter<std::string>("topic_image_light", "image_raw/color/light");
    topic_camera_info_ = declare_parameter<std::string>("topic_camera_info", "camera_info");
    cam1_fid_ = declare_parameter<std::string>(
        "cam1_frame_id", "galaxy_Cam_Left_optical_frame");
    cam2_fid_ = declare_parameter<std::string>(
        "cam2_frame_id", "galaxy_Cam_Right_optical_frame");
    frame_rate_ = declare_parameter<double>("frame_rate", 10.0);
    exposure_time_ = declare_parameter<double>("exposure_time", 100000.0);
    publish_raw_ = declare_parameter<bool>("publish_raw", true);
    publish_color_ = declare_parameter<bool>("publish_color", true);
    publish_color_compressed_ = declare_parameter<bool>("publish_color_compressed", true);
    publish_light_ = declare_parameter<bool>("publish_light", true);
    publish_mapping_ = declare_parameter<bool>("publish_mapping", false);
    mapping_width_ = declare_parameter<int>("mapping_width", 640);
    mapping_height_ = declare_parameter<int>("mapping_height", 536);
    if(mapping_width_<=0 || mapping_height_<=0 || mapping_width_>2448 || mapping_height_>2048)
      throw std::runtime_error("Invalid mapping image dimensions");
    light_scale_ = declare_parameter<double>("light_scale", 0.125);  // 长宽各缩放为 1/8
    jpeg_quality_ = declare_parameter<int>("jpeg_quality", 80);
    auto_wb_ = declare_parameter<bool>("auto_white_balance", true);
    auto_gain_ = declare_parameter<bool>("auto_gain", true);
    gain_ = declare_parameter<double>("gain", 0.0);
    gamma_enable_ = declare_parameter<bool>("gamma_enable", true);
    wb_red_ = declare_parameter<double>("wb_red", 1.0);
    wb_green_ = declare_parameter<double>("wb_green", 1.0);
    wb_blue_ = declare_parameter<double>("wb_blue", 1.0);
    binning_horizontal_ = declare_parameter<int>("binning_horizontal", 1);
    binning_vertical_ = declare_parameter<int>("binning_vertical", 1);
    binning_mode_ = declare_parameter<std::string>("binning_mode", "Average");
    decimation_horizontal_ = declare_parameter<int>("decimation_horizontal", 1);
    decimation_vertical_ = declare_parameter<int>("decimation_vertical", 1);
    // ---- Trigger ----
    trigger_mode_ = declare_parameter<std::string>(
        "trigger_mode", "continuous");
    trigger_source_ = declare_parameter<std::string>(
        "trigger_source", "Line2");
    trigger_activation_ = declare_parameter<std::string>(
        "trigger_activation", "RisingEdge");
    trigger_topic_ = declare_parameter<std::string>(
        "trigger_topic", "/sensors_trigger");
    // ---- Image save ----
    enable_save_ = declare_parameter<bool>("enable_save", false);
    save_dir_ = declare_parameter<std::string>("save_dir", "./Image");

    // 启动时生成时间戳子目录: {save_dir}/Galaxy2_{YYYYMMDD_HHMMSS}/
    if (enable_save_) {
      auto t = std::time(nullptr);
      char ts[32];
      std::strftime(ts, sizeof(ts), "%Y%m%d_%H%M%S", std::localtime(&t));
      actual_save_dir_ = save_dir_ + "/Galaxy2_" + ts;
      const char* cam_names[2] = {cam1_name_.c_str(), cam2_name_.c_str()};
      const std::string cam_ips[2] = {cam1_ip_, cam2_ip_};
      for (int i = 0; i < 2; ++i) {
        if (cam_ips[i].empty()) continue;  // 跳过未连接的相机
        std::string dir = actual_save_dir_ + "/" + cam_names[i];
        std::error_code ec;
        std::filesystem::create_directories(dir, ec);
      }
      RCLCPP_INFO(get_logger(), "图像保存目录: %s", actual_save_dir_.c_str());
    }

    RCLCPP_INFO(get_logger(),
                "Galaxy2 node starting: Cam_Left=%s Cam_Right=%s topic_prefix=%s fps=%.1f exp=%.0fus "
                "gain(auto=%d val=%.2f) gamma(en=%d) wb(auto=%d R=%.2f G=%.2f B=%.2f) "
                "bin(H=%d V=%d mode=%s) decim(H=%d V=%d) ROI=auto "
                "trigger(mode=%s src=%s act=%s) save(en=%d dir=%s) "
                "light(en=%d scale=%.2f)",
                cam1_ip_.empty() ? "(disabled)" : cam1_ip_.c_str(),
                cam2_ip_.empty() ? "(disabled)" : cam2_ip_.c_str(),
                topic_prefix_.c_str(),
                frame_rate_, exposure_time_,
                auto_gain_, gain_, gamma_enable_,
                auto_wb_, wb_red_, wb_green_, wb_blue_,
                binning_horizontal_, binning_vertical_, binning_mode_.c_str(),
                decimation_horizontal_, decimation_vertical_,
                trigger_mode_.c_str(), trigger_source_.c_str(),
                trigger_activation_.c_str(),
                enable_save_, save_dir_.c_str(),
                publish_light_, light_scale_);

    if (!sdk_.init()) {
      RCLCPP_FATAL(get_logger(), "Galaxy SDK init failed; aborting");
      throw std::runtime_error("Galaxy SDK init failed");
    }

    // ---- publishers ----
    // Topic = topic_prefix + camN_name + "/" + topic_suffix
    // e.g. "Car/T5/" + "Cam_Left" + "/" + "image_raw" → /Car/T5/Cam_Left/image_raw
    std::string prefix = topic_prefix_;
    if (!prefix.empty() && prefix[0] != '/') prefix = "/" + prefix;
    if (!prefix.empty() && prefix.back() != '/') prefix += "/";

    const char* cam_names[2] = {cam1_name_.c_str(), cam2_name_.c_str()};
    const std::string cam_ips[2] = {cam1_ip_, cam2_ip_};
    for (int i = 0; i < 2; ++i) {
      if (cam_ips[i].empty()) continue;  // 跳过未连接的相机, 不创建发布器
      std::string base = prefix + cam_names[i] + "/";
      auto qos = rclcpp::QoS(rclcpp::KeepLast(5)).reliable();  // RELIABLE, 保证远程订阅端收到完整帧
      if (publish_raw_) {
        cams_[i].raw_pub = create_publisher<sensor_msgs::msg::Image>(
            base + topic_image_raw_, qos);
      }
      if (publish_color_) {
        cams_[i].color_pub = create_publisher<sensor_msgs::msg::Image>(
            base + topic_image_color_, qos);
      }
      if (publish_mapping_) {
        cams_[i].mapping_pub = create_publisher<sensor_msgs::msg::Image>(
            base + "image_mono/mapping", rclcpp::QoS(2).reliable());
      }
      if (publish_color_compressed_) {
        cams_[i].compressed_pub = create_publisher<sensor_msgs::msg::CompressedImage>(
            base + topic_image_compressed_, qos);
      }
      if (publish_light_) {
        cams_[i].light_pub = create_publisher<sensor_msgs::msg::Image>(
            base + topic_image_light_, qos);
      }
      cams_[i].info_pub = create_publisher<sensor_msgs::msg::CameraInfo>(
          base + topic_camera_info_, qos);
    }
    cams_[0].frame_id = cam1_fid_;
    cams_[1].frame_id = cam2_fid_;

    // ---- open + start cameras ----
    openCamera(0, cam1_ip_);
    openCamera(1, cam2_ip_);

    if (cams_[0].cam == nullptr && cams_[1].cam == nullptr) {
      RCLCPP_FATAL(get_logger(),
                   "No camera could be opened; aborting");
      throw std::runtime_error("no camera opened");
    }

    // ---- trigger topic subscriber (Header stamp = frame timestamp) ----
    // 所有模式均可订阅, 收到 Header 后: software 模式发送软触发, 其他模式仅存 stamp
    trigger_sub_ = create_subscription<std_msgs::msg::Header>(
        trigger_topic_, 10,
        [this](const std_msgs::msg::Header::SharedPtr msg) {
          rclcpp::Time stamp(msg->stamp);
          if (trigger_mode_ == "software") {
            // Serialize trigger assignment against BOTH acquisition threads.
            // Each camera consumes its own stamp. Do not let one image consume
            // the other camera's trigger or overwrite an outstanding exposure.
            std::lock_guard<std::mutex> lock(trigger_mutex_);
            if(has_pending_trigger_stamp_[0] || has_pending_trigger_stamp_[1]) {
              RCLCPP_WARN_THROTTLE(get_logger(),*get_clock(),2000,"Previous stereo trigger incomplete; skipping trigger");
              return;
            }
            for (int i = 0; i < 2; ++i) {
              if(!cams_[i].cam)continue;
              pending_trigger_stamp_[i]=stamp;
              has_pending_trigger_stamp_[i]=true;
              if (!cams_[i].cam->sendSoftwareTrigger()) {
                has_pending_trigger_stamp_[i]=false;
                RCLCPP_WARN(get_logger(), "cam%d software trigger failed: %s",
                            i + 1, cams_[i].cam->lastError().c_str());
              }
            }
          } else {
            // Free-running exposures have no relationship to software triggers.
            // Hardware-trigger timestamps require a separate verified driver.
            RCLCPP_WARN_THROTTLE(get_logger(),*get_clock(),5000,"Software stamps ignored outside software trigger mode");
          }
        });
    RCLCPP_INFO(get_logger(), "Trigger topic: %s", trigger_topic_.c_str());

    // ---- dynamic camera parameter topic ----
    // Receives Float64MultiArray [exposure_time_us, gain_db].
    //   data[0] > 0  → set exposure time (us)
    //   data[1] >= 0 → set gain (dB); turns off auto-gain
    // Either field can be omitted (shorter array) or set to a sentinel
    // (<=0 for exposure, <0 for gain) to skip that parameter.
    cam_cmd_sub_ = create_subscription<std_msgs::msg::Float64MultiArray>(
        "~/set_camera_params",
        rclcpp::QoS(rclcpp::KeepLast(1)).reliable(),
        [this](const std_msgs::msg::Float64MultiArray::SharedPtr msg) {
          if (msg->data.empty()) return;
          // exposure time (us)
          if (msg->data.size() >= 1 && msg->data[0] > 0.0) {
            for (int i = 0; i < 2; ++i) {
              if (!cams_[i].cam) continue;
              // 设备当前曝光时间与指令相同时跳过
              if (cams_[i].cam->config().exposure_time == msg->data[0]) {
                RCLCPP_DEBUG(get_logger(), "cam%d exposure already %.0f us, skip",
                             i + 1, msg->data[0]);
                continue;
              }
              if (cams_[i].cam->setExposureTime(msg->data[0])) {
                RCLCPP_INFO(get_logger(), "cam%d exposure set to %.0f us",
                            i + 1, msg->data[0]);
              } else {
                RCLCPP_WARN(get_logger(), "cam%d set exposure failed: %s",
                            i + 1, cams_[i].cam->lastError().c_str());
              }
            }
          }
          // gain (dB)
          if (msg->data.size() >= 2 && msg->data[1] >= 0.0) {
            for (int i = 0; i < 2; ++i) {
              if (cams_[i].cam && cams_[i].cam->setGain(msg->data[1])) {
                RCLCPP_INFO(get_logger(), "cam%d gain set to %.2f dB",
                            i + 1, msg->data[1]);
              } else if (cams_[i].cam) {
                RCLCPP_WARN(get_logger(), "cam%d set gain failed: %s",
                            i + 1, cams_[i].cam->lastError().c_str());
              }
            }
          }
        });
    RCLCPP_INFO(get_logger(),
                "Camera params topic: ~/set_camera_params [exposure_us, gain_db]");
  }

  ~GalaxyNode() override {
    for (int i = 0; i < 2; ++i) {
      if (cams_[i].cam) cams_[i].cam->stop();
    }
    for (int i = 0; i < 2; ++i) {
      if (cams_[i].cam) cams_[i].cam->close();
    }
  }

private:
  struct CamCtx {
    std::unique_ptr<GalaxyCamera> cam;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr raw_pub;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr color_pub;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr mapping_pub;
    rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr compressed_pub;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr light_pub;
    rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr info_pub;
    std::string frame_id;
    uint64_t count = 0;
    bool info_sent = false;
    uint64_t save_count = 0;  // 保存图像计数
  };

  void openCamera(int idx, const std::string& ip) {
    if (ip.empty()) {
      RCLCPP_WARN(get_logger(), "cam%d ip is empty, skipping", idx + 1);
      return;
    }
    CameraConfig cfg;
    cfg.ip = ip;
    cfg.frame_id = cams_[idx].frame_id;
    cfg.frame_rate = frame_rate_;
    cfg.exposure_time = exposure_time_;
    cfg.publish_raw = publish_raw_;
    cfg.publish_color = publish_color_;
    cfg.auto_white_balance = auto_wb_;
    cfg.auto_gain = auto_gain_;
    cfg.gain = gain_;
    cfg.gamma_enable = gamma_enable_;
    cfg.wb_red = wb_red_;
    cfg.wb_green = wb_green_;
    cfg.wb_blue = wb_blue_;
    cfg.binning_horizontal = binning_horizontal_;
    cfg.binning_vertical = binning_vertical_;
    cfg.binning_mode = binning_mode_;
    cfg.decimation_horizontal = decimation_horizontal_;
    cfg.decimation_vertical = decimation_vertical_;
    cfg.trigger_mode = trigger_mode_;
    cfg.trigger_source = trigger_source_;
    cfg.trigger_activation = trigger_activation_;

    auto cam = std::make_unique<GalaxyCamera>();
    if (!cam->open(cfg)) {
      RCLCPP_ERROR(get_logger(), "cam%d (%s) open failed: %s", idx + 1,
                   ip.c_str(), cam->lastError().c_str());
      return;
    }
    const int raw_idx = idx;  // captured
    GalaxyCamera::FrameCallback cb =
        [this, raw_idx](const GalaxyFrame& f) { onFrame(raw_idx, f); };
    if (!cam->start(cb)) {
      RCLCPP_ERROR(get_logger(), "cam%d (%s) start failed: %s", idx + 1,
                   ip.c_str(), cam->lastError().c_str());
      cam->close();
      return;
    }
    cams_[idx].cam = std::move(cam);
    RCLCPP_INFO(get_logger(), "cam%d (%s) streaming (%s, payload=%u)",
                idx + 1, ip.c_str(), cams_[idx].cam->isColor() ? "color" : "mono",
                cams_[idx].cam->payloadSize());
  }

  void onFrame(int idx, const GalaxyFrame& f) {
    auto& ctx = cams_[idx];
    // Use trigger stamp if available, otherwise use current time
    rclcpp::Time stamp;
    {
      std::lock_guard<std::mutex> lock(trigger_mutex_);
      if (has_pending_trigger_stamp_[idx]) {
        stamp = pending_trigger_stamp_[idx];
        has_pending_trigger_stamp_[idx] = false;
        if((now()-stamp).seconds()>.5) {
          RCLCPP_WARN(get_logger(),"Stale triggered camera frame rejected");
          return;
        }
      } else {
        if(trigger_mode_=="software")return;
        stamp = now();
      }
    }
    ctx.count++;

    // Preserve the calibrated full field of view. Publish only the compact
    // mono image to the mapping pipeline, before ROS serialization/Python.
    if (publish_mapping_ && f.rgb_data != nullptr && ctx.mapping_pub) {
      cv::Mat rgb(static_cast<int>(f.height),static_cast<int>(f.width),
                  CV_8UC3,const_cast<uint8_t*>(f.rgb_data));
      cv::Mat mono,compact;
      cv::cvtColor(rgb,mono,cv::COLOR_RGB2GRAY);
      cv::resize(mono,compact,cv::Size(mapping_width_,mapping_height_),0,0,cv::INTER_AREA);
      sensor_msgs::msg::Image image;
      image.header.stamp=stamp;image.header.frame_id=ctx.frame_id;
      image.height=mapping_height_;image.width=mapping_width_;
      image.encoding="mono8";image.step=mapping_width_;
      image.data.assign(compact.data,compact.data+compact.total());
      ctx.mapping_pub->publish(std::move(image));
    }

    // Raw BayerRG8 image.
    if (publish_raw_ && ctx.raw_pub) {
      sensor_msgs::msg::Image img;
      img.header.stamp = stamp;
      img.header.frame_id = ctx.frame_id;
      img.height = f.height;
      img.width = f.width;
      img.encoding = "bayer_rggr8";  // BayerRG8 == RGGB pattern
      img.is_bigendian = 0;
      img.step = f.width;
      img.data.assign(f.raw_data, f.raw_data + static_cast<size_t>(f.width) * f.height);
      ctx.raw_pub->publish(std::move(img));
    }

    // Color RGB8 image.
    if (publish_color_ && f.rgb_data != nullptr && ctx.color_pub) {
      sensor_msgs::msg::Image img;
      img.header.stamp = stamp;
      img.header.frame_id = ctx.frame_id;
      img.height = f.height;
      img.width = f.width;
      img.encoding = "rgb8";
      img.is_bigendian = 0;
      img.step = f.width * 3u;
      img.data.assign(f.rgb_data,
                      f.rgb_data + static_cast<size_t>(f.width) * f.height * 3);
      ctx.color_pub->publish(std::move(img));
    }

    // Compressed JPEG color image.
    if (publish_color_compressed_ && f.rgb_data != nullptr && ctx.compressed_pub) {
      cv::Mat rgb_mat(static_cast<int>(f.height), static_cast<int>(f.width),
                      CV_8UC3, const_cast<uint8_t*>(f.rgb_data));
      std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, jpeg_quality_};
      std::vector<uint8_t> jpeg_buf;
      if (cv::imencode(".jpg", rgb_mat, jpeg_buf, params)) {
        sensor_msgs::msg::CompressedImage cimg;
        cimg.header.stamp = stamp;
        cimg.header.frame_id = ctx.frame_id;
        cimg.format = "jpeg";
        cimg.data.assign(jpeg_buf.begin(), jpeg_buf.end());
        ctx.compressed_pub->publish(std::move(cimg));
      }
    }

    // Lightweight downscaled RGB8 image (for remote viewing over WiFi).
    // cv::resize INTER_AREA 最适合下采样, 输出尺寸 = 原始 * light_scale_
    if (publish_light_ && f.rgb_data != nullptr && ctx.light_pub) {
      cv::Mat rgb_mat(static_cast<int>(f.height), static_cast<int>(f.width),
                      CV_8UC3, const_cast<uint8_t*>(f.rgb_data));
      int lw = std::max(1, static_cast<int>(f.width * light_scale_));
      int lh = std::max(1, static_cast<int>(f.height * light_scale_));
      cv::Mat light_mat;
      cv::resize(rgb_mat, light_mat, cv::Size(lw, lh), 0, 0, cv::INTER_AREA);
      sensor_msgs::msg::Image img;
      img.header.stamp = stamp;
      img.header.frame_id = ctx.frame_id;
      img.height = static_cast<uint32_t>(lh);
      img.width = static_cast<uint32_t>(lw);
      img.encoding = "rgb8";
      img.is_bigendian = 0;
      img.step = static_cast<uint32_t>(lw * 3);
      img.data.assign(light_mat.data, light_mat.data + static_cast<size_t>(lw) * lh * 3);
      ctx.light_pub->publish(std::move(img));
    }

    // ---- Save image to file ----
    if (enable_save_ && f.rgb_data != nullptr) {
      // RGB -> BGR for OpenCV imwrite
      cv::Mat rgb_mat(static_cast<int>(f.height), static_cast<int>(f.width),
                      CV_8UC3, const_cast<uint8_t*>(f.rgb_data));
      cv::Mat bgr_mat;
      cv::cvtColor(rgb_mat, bgr_mat, cv::COLOR_RGB2BGR);

      const char* cam_names[2] = {cam1_name_.c_str(), cam2_name_.c_str()};
      auto now = std::chrono::system_clock::now();
      auto now_time_t = std::chrono::system_clock::to_time_t(now);
      auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
          now.time_since_epoch()) % 1000;
      char ts[32];
      std::strftime(ts, sizeof(ts), "%Y%m%d_%H%M%S", std::localtime(&now_time_t));
      // 曝光时间 (us -> s) 嵌入文件名
      double exp_us = cams_[idx].cam ? cams_[idx].cam->config().exposure_time : exposure_time_;
      double exp_s = exp_us / 1000000.0;
      char exp_str[32];
      snprintf(exp_str, sizeof(exp_str), "%g", exp_s);

      char filename[512];
      snprintf(filename, sizeof(filename), "%s/%s/%s_%s_%03llu_%ss.png",
               actual_save_dir_.c_str(), cam_names[idx],
               cam_names[idx], ts,
               static_cast<unsigned long long>(ms.count()),
               exp_str);
      if (cv::imwrite(filename, bgr_mat)) {
        ctx.save_count++;
        if (ctx.save_count % 100 == 0) {
          RCLCPP_INFO(get_logger(), "cam%d saved %llu images to %s/%s",
                      idx + 1,
                      static_cast<unsigned long long>(ctx.save_count),
                      actual_save_dir_.c_str(), cam_names[idx]);
        }
      } else {
        RCLCPP_WARN(get_logger(), "cam%d save image failed: %s",
                    idx + 1, filename);
      }
    }

    // CameraInfo: published periodically (every 30 frames) so that subscribers
    // started after the first frame still receive it. Fields are constant
    // for a given resolution.
    if (ctx.info_pub && (ctx.count % 30 == 1)) {
      sensor_msgs::msg::CameraInfo ci;
      ci.header.stamp = stamp;
      ci.header.frame_id = ctx.frame_id;
      ci.height = f.height;
      ci.width = f.width;
      ci.distortion_model = "plumb_bob";
      ci.d.resize(5, 0.0);
      // Placeholder intrinsics: fx = fy = max(w,h), principal point at center.
      double f0 = std::max(f.width, f.height);
      ci.k = {f0, 0, f.width / 2.0, 0, f0, f.height / 2.0, 0, 0, 1};
      ci.r = {1, 0, 0, 0, 1, 0, 0, 0, 1};
      ci.p = {f0, 0, f.width / 2.0, 0, 0, f0, f.height / 2.0, 0, 0, 0, 1, 0};
      ctx.info_pub->publish(ci);
      ctx.info_sent = true;
    }
  }

  GalaxySDKLib sdk_;
  std::array<CamCtx, 2> cams_;

  std::string cam1_ip_, cam2_ip_, cam1_fid_, cam2_fid_;
  std::string topic_prefix_ = "Car/T5/";
  std::string cam1_name_ = "cam1";
  std::string cam2_name_ = "cam2";
  std::string topic_image_raw_ = "image_raw";
  std::string topic_image_color_ = "image_raw/color";
  std::string topic_image_compressed_ = "image_raw/color/compressed";
  std::string topic_image_light_ = "image_raw/color/light";
  std::string topic_camera_info_ = "camera_info";
  double frame_rate_ = 10.0;
  double exposure_time_ = 40000.0;
  bool publish_raw_ = true;
  bool publish_color_ = true;
  bool publish_color_compressed_ = true;
  bool publish_light_ = true;
  bool publish_mapping_ = false;
  int mapping_width_ = 640, mapping_height_ = 536;
  double light_scale_ = 0.25;  // 长宽各缩放为 1/4
  int jpeg_quality_ = 80;
  bool auto_wb_ = true;
  bool auto_gain_ = true;
  double gain_ = 0.0;
  bool gamma_enable_ = true;
  double wb_red_ = 1.0;
  double wb_green_ = 1.0;
  double wb_blue_ = 1.0;
  int binning_horizontal_ = 1;
  int binning_vertical_ = 1;
  std::string binning_mode_ = "Average";
  int decimation_horizontal_ = 1;
  int decimation_vertical_ = 1;
  std::string trigger_mode_ = "continuous";
  std::string trigger_source_ = "Line2";
  std::string trigger_activation_ = "RisingEdge";
  std::string trigger_topic_ = "/sensors_trigger";

  // ---- Image save ----
  bool enable_save_ = false;
  std::string save_dir_ = "./Image";
  std::string actual_save_dir_;  // 启动时生成: save_dir_/Galaxy2_YYYYMMDD_HHMMSS

  // ---- Trigger topic ----
  rclcpp::Subscription<std_msgs::msg::Header>::SharedPtr trigger_sub_;
  std::mutex trigger_mutex_;
  std::array<rclcpp::Time,2> pending_trigger_stamp_;
  std::array<bool,2> has_pending_trigger_stamp_{false,false};

  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr cam_cmd_sub_;
};

}  // namespace galaxy2

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<galaxy2::GalaxyNode>());
  rclcpp::shutdown();
  return 0;
}
