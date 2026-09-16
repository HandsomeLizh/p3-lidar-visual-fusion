// Copyright 2026 yanfa
// Galaxy SDK (大恒图像) dual GigE camera wrapper.
//
// Wraps the official Galaxy SDK (libgxiapi) for a single color GigE camera:
//   - opens the device by IP (GXOpenDevice + GX_OPEN_IP)
//   - configures acquisition (continuous / external-trigger / software-trigger)
//   - runs an acquisition thread (GXDQBuf / GXQBuf)
//   - converts BayerRG8 -> RGB24 via DxRaw8toRGB24
//   - hands each frame to a caller-provided callback
//
// The library (GXInitLib / GXCloseLib) is managed by GalaxySDKLib (RAII).

#pragma once

#include <cstdint>
#include <atomic>
#include <functional>
#include <string>
#include <thread>

#include "GxIAPI.h"
#include "DxImageProc.h"

namespace galaxy2 {

/// Per-camera configuration.  All fields are shared across both cameras so the
/// captured image pairs stay consistent in exposure, gain, white balance, etc.
struct CameraConfig {
  std::string ip;                 ///< Camera IP to open, e.g. "192.168.19.10".
  std::string frame_id;           ///< TF/optical frame id for the image header.
  double frame_rate = 10.0;       ///< AcquisitionFrameRate (fps).
  double exposure_time = 100000.0; ///< ExposureTime (us).
  int64_t packet_size = 0;        ///< GevSCPSPacketSize; 0 = use optimal value.
  bool publish_raw = true;        ///< Publish BayerRG8 raw image.
  bool publish_color = true;      ///< Publish debayered RGB8 color image.
  bool auto_white_balance = true; ///< BalanceWhiteAuto = Continuous.

  // ---- Manual gain ----
  bool auto_gain = true;          ///< GainAuto = Continuous when true.
  double gain = 0.0;              ///< Gain (dB) when auto_gain is false.

  // ---- Gamma ----
  bool gamma_enable = true;       ///< GammaEnable on/off.

  // ---- Manual white balance (used when auto_white_balance == false) ----
  double wb_red = 1.0;            ///< BalanceRatio for Red channel.
  double wb_green = 1.0;          ///< BalanceRatio for Green channel.
  double wb_blue = 1.0;           ///< BalanceRatio for Blue channel.

  // ---- Binning (downsampling by averaging/summing adjacent pixels) ----
  int binning_horizontal = 1;     ///< 1 = off, 2 = 2x horizontal binning.
  int binning_vertical = 1;       ///< 1 = off, 2 = 2x vertical binning.
  std::string binning_mode = "Average";  ///< "Sum" or "Average".

  // ---- Decimation (skipping pixels) ----
  int decimation_horizontal = 1;  ///< 1 = off, 2 = skip every other pixel.
  int decimation_vertical = 1;    ///< 1 = off, 2 = skip every other line.

  // ---- Trigger (acquisition mode) ----
  /// "continuous" = free-run (default), "external" = hardware trigger,
  /// "software" = software trigger via sendSoftwareTrigger().
  std::string trigger_mode = "continuous";
  std::string trigger_source = "Line2";      ///< TriggerSource: "Line0"/"Line2"/"Line3".
  std::string trigger_activation = "RisingEdge";  ///< "RisingEdge"/"FallingEdge".

  // Width/Height is auto-calculated from WidthMax/HeightMax after
  // binning/decimation — no manual ROI specification needed.
};

/// Immutable snapshot of one acquired frame, handed to the callback.
/// raw_data points into the SDK buffer (valid only during the callback).
/// rgb_data points into an internal reusable buffer (valid during callback).
struct GalaxyFrame {
  uint32_t width = 0;
  uint32_t height = 0;
  uint64_t frame_id = 0;
  int64_t pixel_format = 0;       ///< GX_PIXEL_FORMAT_* of the raw frame.
  int64_t color_filter = 0;       ///< GX_COLOR_FILTER_* of the device.
  const uint8_t* raw_data = nullptr;  ///< BayerRG8 buffer, width*height bytes.
  const uint8_t* rgb_data = nullptr;  ///< RGB24 buffer, width*height*3 bytes.
};

/// RAII guard for the Galaxy SDK library (GXInitLib / GXCloseLib).
/// A single instance should be kept alive while any GalaxyCamera is open.
class GalaxySDKLib {
public:
  /// Calls GXInitLib and enumerates devices. Returns false on failure.
  bool init();
  ~GalaxySDKLib();

  bool inited() const { return inited_; }
  uint32_t deviceCount() const { return device_count_; }

private:
  bool inited_ = false;
  uint32_t device_count_ = 0;
};

/// Owns one camera: open / stream / stop / close.
class GalaxyCamera {
public:
  using FrameCallback = std::function<void(const GalaxyFrame&)>;

  GalaxyCamera() = default;
  ~GalaxyCamera();

  GalaxyCamera(const GalaxyCamera&) = delete;
  GalaxyCamera& operator=(const GalaxyCamera&) = delete;

  /// Open the camera at cfg.ip (must have called GalaxySDKLib::init first).
  /// Returns false and fills lastError() on failure.
  bool open(const CameraConfig& cfg);

  /// Start streaming; cb is invoked from an internal acquisition thread per
  /// successfully grabbed frame. Returns false if streaming cannot start.
  bool start(FrameCallback cb);

  /// Signal the acquisition thread to stop and join it. Safe to call repeatedly.
  void stop();

  /// Close the device. Called again by the destructor.
  void close();

  /// Send a software trigger command (only valid in "software" trigger mode).
  /// Returns false if the camera is not open or the command fails.
  bool sendSoftwareTrigger();

  /// Dynamically update exposure time (us). Camera must be open.
  /// Returns false on failure.
  bool setExposureTime(double us);

  /// Dynamically update gain (dB). Turns off auto-gain if enabled.
  /// Returns false on failure.
  bool setGain(double db);

  const CameraConfig& config() const { return cfg_; }
  bool isOpen() const { return handle_ != nullptr; }
  bool isColor() const { return is_color_; }
  bool isTriggerMode() const { return trigger_mode_ != "continuous"; }
  uint32_t payloadSize() const { return payload_size_; }
  std::string lastError() const { return last_error_; }

private:
  bool configureDevice();
  void acquireLoop();
  void setError(const std::string& context, GX_STATUS status);

  CameraConfig cfg_;
  GX_DEV_HANDLE handle_ = nullptr;     ///< Device handle.
  GX_DS_HANDLE ds_handle_ = nullptr;    ///< Data stream handle.
  uint32_t payload_size_ = 0;           ///< Bytes per frame (raw).
  int64_t color_filter_ = 0;            ///< GX_COLOR_FILTER_*.
  bool is_color_ = false;
  std::string trigger_mode_ = "continuous";  ///< Cached trigger mode.

  uint8_t* rgb_buf_ = nullptr;         ///< Reusable RGB24 conversion buffer.
  uint8_t* raw8_buf_ = nullptr;         ///< Reusable RAW16->RAW8 buffer.

  std::atomic<bool> running_{false};
  std::thread thread_;
  FrameCallback callback_;

  std::string last_error_;
};

}  // namespace galaxy2
