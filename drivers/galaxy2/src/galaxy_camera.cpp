// Galaxy SDK camera wrapper implementation.
#include "galaxy2/galaxy_camera.h"

#include <cstdio>
#include <cstring>
#include <cstdlib>

namespace galaxy2 {

namespace {

constexpr uint64_t kAcqBufferNum = 10;       ///< Acquisition queue depth.
constexpr int64_t kStreamTransferSize = 64 * 1024;
constexpr int64_t kStreamTransferUrb = 64;

/// Translate a GX_STATUS to a human readable string.
std::string statusText(GX_STATUS status) {
  char buf[1024] = {0};
  size_t size = sizeof(buf);
  GX_STATUS err = status;
  if (GXGetLastError(&err, buf, &size) == GX_STATUS_SUCCESS && size > 0) {
    return std::string(buf);
  }
  char num[32];
  snprintf(num, sizeof(num), "0x%08X", static_cast<unsigned>(status));
  return std::string("status ") + num;
}

/// True if the named feature is writable (RW or WO).
bool isWritable(GX_PORT_HANDLE port, const char* name) {
  GX_NODE_ACCESS_MODE mode = GX_NODE_ACCESS_MODE_NA;
  if (GXGetNodeAccessMode(port, name, &mode) != GX_STATUS_SUCCESS) return false;
  return mode == GX_NODE_ACCESS_MODE_RW || mode == GX_NODE_ACCESS_MODE_WO;
}

/// True if the named feature exists and is readable.
bool isReadable(GX_PORT_HANDLE port, const char* name) {
  GX_NODE_ACCESS_MODE mode = GX_NODE_ACCESS_MODE_NA;
  if (GXGetNodeAccessMode(port, name, &mode) != GX_STATUS_SUCCESS) return false;
  return mode == GX_NODE_ACCESS_MODE_RW || mode == GX_NODE_ACCESS_MODE_RO;
}

/// Set an enum by string, ignoring failures (feature may be absent).
GX_STATUS trySetEnum(GX_PORT_HANDLE port, const char* name, const char* value) {
  if (!isWritable(port, name)) return GX_STATUS_SUCCESS;
  return GXSetEnumValueByString(port, name, value);
}

/// Set a float, ignoring failures.
GX_STATUS trySetFloat(GX_PORT_HANDLE port, const char* name, double value) {
  if (!isWritable(port, name)) return GX_STATUS_SUCCESS;
  return GXSetFloatValue(port, name, value);
}

/// Set an int, ignoring failures.
GX_STATUS trySetInt(GX_PORT_HANDLE port, const char* name, int64_t value) {
  if (!isWritable(port, name)) return GX_STATUS_SUCCESS;
  return GXSetIntValue(port, name, value);
}

/// Set a bool, ignoring failures.
GX_STATUS trySetBool(GX_PORT_HANDLE port, const char* name, bool value) {
  if (!isWritable(port, name)) return GX_STATUS_SUCCESS;
  return GXSetBoolValue(port, name, value);
}

}  // namespace

// ---------------------------------------------------------------------------
// GalaxySDKLib
// ---------------------------------------------------------------------------

bool GalaxySDKLib::init() {
  if (inited_) return true;
  GX_STATUS status = GXInitLib();
  if (status != GX_STATUS_SUCCESS) {
    fprintf(stderr, "[GalaxySDKLib] GXInitLib failed: %s\n",
            statusText(status).c_str());
    return false;
  }
  inited_ = true;
  device_count_ = 0;
  status = GXUpdateAllDeviceList(&device_count_, 2000);
  if (status != GX_STATUS_SUCCESS) {
    fprintf(stderr, "[GalaxySDKLib] GXUpdateAllDeviceList failed: %s\n",
            statusText(status).c_str());
  }
  fprintf(stdout, "[GalaxySDKLib] library v%s, %u device(s) enumerated\n",
          GXGetLibVersion(), device_count_);
  return true;
}

GalaxySDKLib::~GalaxySDKLib() {
  if (inited_) {
    GXCloseLib();
    inited_ = false;
  }
}

// ---------------------------------------------------------------------------
// GalaxyCamera
// ---------------------------------------------------------------------------

GalaxyCamera::~GalaxyCamera() {
  stop();
  close();
}

void GalaxyCamera::setError(const std::string& context, GX_STATUS status) {
  last_error_ = context + ": " + statusText(status);
  fprintf(stderr, "[GalaxyCamera %s] %s\n", cfg_.ip.c_str(), last_error_.c_str());
}

bool GalaxyCamera::open(const CameraConfig& cfg) {
  cfg_ = cfg;
  if (cfg_.ip.empty()) {
    last_error_ = "empty camera IP";
    return false;
  }

  // Open by IP directly (the SDK discovers devices via its own transport layer).
  GX_OPEN_PARAM open_param;
  std::string ip = cfg_.ip;
  open_param.pszContent = const_cast<char*>(ip.c_str());
  open_param.openMode = GX_OPEN_IP;
  open_param.accessMode = GX_ACCESS_EXCLUSIVE;

  GX_STATUS status = GXOpenDevice(&open_param, &handle_);
  if (status != GX_STATUS_SUCCESS || handle_ == nullptr) {
    setError("GXOpenDevice", status);
    handle_ = nullptr;
    return false;
  }

  // Log device identity.
  if (isReadable(handle_, "DeviceModelName")) {
    GX_STRING_VALUE model;
    if (GXGetStringValue(handle_, "DeviceModelName", &model) == GX_STATUS_SUCCESS)
      fprintf(stdout, "[GalaxyCamera %s] %s opened\n", cfg_.ip.c_str(),
              model.strCurValue);
  }

  // Detect color filter (Bayer pattern).
  is_color_ = isReadable(handle_, "PixelColorFilter");
  if (is_color_) {
    GX_ENUM_VALUE ev;
    if (GXGetEnumValue(handle_, "PixelColorFilter", &ev) == GX_STATUS_SUCCESS)
      color_filter_ = ev.stCurValue.nCurValue;
  }

  // Data stream handle (payload size is read AFTER configureDevice because
  // Binning/Decimation changes affect the effective frame size).
  uint32_t ds_num = 0;
  status = GXGetDataStreamNumFromDev(handle_, &ds_num);
  if (status != GX_STATUS_SUCCESS || ds_num < 1) {
    setError("GXGetDataStreamNumFromDev", status);
    close();
    return false;
  }
  status = GXGetDataStreamHandleFromDev(handle_, 1, &ds_handle_);
  if (status != GX_STATUS_SUCCESS || ds_handle_ == nullptr) {
    setError("GXGetDataStreamHandleFromDev", status);
    close();
    return false;
  }

  if (!configureDevice()) {
    close();
    return false;
  }

  // Read payload size AFTER configureDevice (Binning/Decimation/ROI affect it).
  status = GXGetPayLoadSize(ds_handle_, &payload_size_);
  if (status != GX_STATUS_SUCCESS || payload_size_ == 0) {
    setError("GXGetPayLoadSize", status);
    close();
    return false;
  }
  return true;
}

bool GalaxyCamera::configureDevice() {
  // Optimal GigE packet size (matches the link MTU). Override allowed.
  if (cfg_.packet_size > 0) {
    trySetInt(handle_, "GevSCPSPacketSize", cfg_.packet_size);
  } else if (isWritable(handle_, "GevSCPSPacketSize")) {
    uint32_t optimal = 0;
    if (GXGetOptimalPacketSize(handle_, &optimal) == GX_STATUS_SUCCESS)
      trySetInt(handle_, "GevSCPSPacketSize", static_cast<int64_t>(optimal));
  }

  // Packet delay: use the minimum to maximise throughput.
  if (isReadable(handle_, "GevSCPD")) {
    GX_INT_VALUE iv;
    if (GXGetIntValue(handle_, "GevSCPD", &iv) == GX_STATUS_SUCCESS)
      trySetInt(handle_, "GevSCPD", iv.nMin);
  }

  // Force the BayerRG8 pixel format on color cameras (deterministic).
  if (is_color_) trySetEnum(handle_, "PixelFormat", "BayerRG8");

  // ---- Binning (must be set before Width/Height) ----
  trySetEnum(handle_, "BinningHorizontalMode", cfg_.binning_mode.c_str());
  trySetEnum(handle_, "BinningVerticalMode", cfg_.binning_mode.c_str());
  trySetInt(handle_, "BinningHorizontal", cfg_.binning_horizontal);
  trySetInt(handle_, "BinningVertical", cfg_.binning_vertical);

  // ---- Decimation (must be set before Width/Height) ----
  trySetInt(handle_, "DecimationHorizontal", cfg_.decimation_horizontal);
  trySetInt(handle_, "DecimationVertical", cfg_.decimation_vertical);

  // ---- ROI: auto-calculate from WidthMax/HeightMax ----
  // After binning/decimation the sensor's effective resolution changes.
  // Read WidthMax/HeightMax from the camera and set Width/Height to the
  // maximum allowed value, so the full effective frame is captured.
  // e.g. 2448x2048 with binning=2 → WidthMax=1224, HeightMax=1024
  trySetInt(handle_, "OffsetX", 0);  // must zero offset before resizing
  trySetInt(handle_, "OffsetY", 0);

  if (isReadable(handle_, "WidthMax")) {
    GX_INT_VALUE wv, hv;
    if (GXGetIntValue(handle_, "WidthMax", &wv) == GX_STATUS_SUCCESS)
      trySetInt(handle_, "Width", wv.nCurValue);
    if (isReadable(handle_, "HeightMax") &&
        GXGetIntValue(handle_, "HeightMax", &hv) == GX_STATUS_SUCCESS)
      trySetInt(handle_, "Height", hv.nCurValue);
  } else {
    // Fallback: compute from sensor resolution.
    int64_t eff_w = 2448 / (cfg_.binning_horizontal * cfg_.decimation_horizontal);
    int64_t eff_h = 2048 / (cfg_.binning_vertical * cfg_.decimation_vertical);
    trySetInt(handle_, "Width", eff_w);
    trySetInt(handle_, "Height", eff_h);
  }

  // Log the effective resolution after binning/decimation.
  if (isReadable(handle_, "Width") && isReadable(handle_, "Height")) {
    GX_INT_VALUE cw, ch;
    if (GXGetIntValue(handle_, "Width", &cw) == GX_STATUS_SUCCESS &&
        GXGetIntValue(handle_, "Height", &ch) == GX_STATUS_SUCCESS)
      fprintf(stdout, "[GalaxyCamera %s] ROI = %lldx%lld (bin H=%d V=%d, decim H=%d V=%d)\n",
              cfg_.ip.c_str(),
              static_cast<long long>(cw.nCurValue),
              static_cast<long long>(ch.nCurValue),
              cfg_.binning_horizontal, cfg_.binning_vertical,
              cfg_.decimation_horizontal, cfg_.decimation_vertical);
  }

  // ---- Acquisition / trigger mode ----
  // AcquisitionMode is always "Continuous" (one frame per trigger).
  // The actual acquisition behaviour is controlled by TriggerMode.
  trySetEnum(handle_, "AcquisitionMode", "Continuous");

  if (cfg_.trigger_mode == "external") {
    // Hardware trigger: camera captures one frame per external signal edge.
    trySetEnum(handle_, "TriggerMode", "On");
    trySetEnum(handle_, "TriggerSource", cfg_.trigger_source.c_str());
    trySetEnum(handle_, "TriggerActivation", cfg_.trigger_activation.c_str());
    fprintf(stdout, "[GalaxyCamera %s] trigger=external src=%s act=%s\n",
            cfg_.ip.c_str(), cfg_.trigger_source.c_str(),
            cfg_.trigger_activation.c_str());
  } else if (cfg_.trigger_mode == "software") {
    // Software trigger: caller must invoke sendSoftwareTrigger() per frame.
    trySetEnum(handle_, "TriggerMode", "On");
    trySetEnum(handle_, "TriggerSource", "Software");
    fprintf(stdout, "[GalaxyCamera %s] trigger=software\n", cfg_.ip.c_str());
  } else {
    // Continuous (free-run): default mode, no trigger needed.
    trySetEnum(handle_, "TriggerMode", "Off");
  }
  trigger_mode_ = cfg_.trigger_mode;

  // Exposure and frame rate (best-effort; feature set varies).
  trySetFloat(handle_, "ExposureTime", cfg_.exposure_time);
  // Some cameras gate AcquisitionFrameRate behind an enable boolean.
  trySetInt(handle_, "AcquisitionFrameRateEnable", 1);
  trySetFloat(handle_, "AcquisitionFrameRate", cfg_.frame_rate);

  // ---- Gain (auto or manual) ----
  if (cfg_.auto_gain) {
    trySetEnum(handle_, "GainAuto", "Continuous");
  } else {
    trySetEnum(handle_, "GainAuto", "Off");
    trySetFloat(handle_, "Gain", cfg_.gain);
  }

  // ---- Gamma ----
  if (cfg_.gamma_enable) {
    trySetBool(handle_, "GammaEnable", true);
  } else {
    trySetBool(handle_, "GammaEnable", false);
  }

  // ---- White balance (auto or manual RGB ratios) ----
  if (cfg_.auto_white_balance) {
    trySetEnum(handle_, "BalanceWhiteAuto", "Continuous");
  } else {
    trySetEnum(handle_, "BalanceWhiteAuto", "Off");
    // Galaxy SDK: select each channel via BalanceRatioSelector then set value.
    trySetEnum(handle_, "BalanceRatioSelector", "Red");
    trySetFloat(handle_, "BalanceRatio", cfg_.wb_red);
    trySetEnum(handle_, "BalanceRatioSelector", "Green");
    trySetFloat(handle_, "BalanceRatio", cfg_.wb_green);
    trySetEnum(handle_, "BalanceRatioSelector", "Blue");
    trySetFloat(handle_, "BalanceRatio", cfg_.wb_blue);
  }

  // Acquisition queue depth.
  GXSetAcqusitionBufferNumber(handle_, kAcqBufferNum);

  // Data-stream transfer tuning (best-effort).
  trySetInt(ds_handle_, "StreamTransferSize", kStreamTransferSize);
  trySetInt(ds_handle_, "StreamTransferNumberUrb", kStreamTransferUrb);

  return true;
}

bool GalaxyCamera::start(FrameCallback cb) {
  if (!isOpen()) {
    last_error_ = "start called on a closed camera";
    return false;
  }
  if (running_.load()) return true;

  // Allocate reusable conversion buffers.
  if (is_color_ && !rgb_buf_) {
    rgb_buf_ = static_cast<uint8_t*>(std::malloc(payload_size_ * 3));
  }
  callback_ = std::move(cb);

  GX_STATUS status = GXStreamOn(handle_);
  if (status != GX_STATUS_SUCCESS) {
    setError("GXStreamOn", status);
    return false;
  }

  running_.store(true);
  thread_ = std::thread(&GalaxyCamera::acquireLoop, this);
  return true;
}

bool GalaxyCamera::sendSoftwareTrigger() {
  if (!isOpen()) {
    last_error_ = "sendSoftwareTrigger: camera not open";
    return false;
  }
  if (trigger_mode_ != "software") {
    last_error_ = "sendSoftwareTrigger: not in software trigger mode";
    return false;
  }
  GX_STATUS status = GXSendCommand(handle_, GX_COMMAND_TRIGGER_SOFTWARE);
  if (status != GX_STATUS_SUCCESS) {
    setError("GXSendCommand(TriggerSoftware)", status);
    return false;
  }
  return true;
}

bool GalaxyCamera::setExposureTime(double us) {
  if (!isOpen()) {
    last_error_ = "setExposureTime: camera not open";
    return false;
  }
  GX_STATUS status = trySetFloat(handle_, "ExposureTime", us);
  if (status != GX_STATUS_SUCCESS) {
    setError("setExposureTime", status);
    return false;
  }
  cfg_.exposure_time = us;
  return true;
}

bool GalaxyCamera::setGain(double db) {
  if (!isOpen()) {
    last_error_ = "setGain: camera not open";
    return false;
  }
  // Turn off auto-gain before setting manual gain.
  trySetEnum(handle_, "GainAuto", "Off");
  GX_STATUS status = trySetFloat(handle_, "Gain", db);
  if (status != GX_STATUS_SUCCESS) {
    setError("setGain", status);
    return false;
  }
  cfg_.auto_gain = false;
  cfg_.gain = db;
  return true;
}

void GalaxyCamera::stop() {
  if (!running_.exchange(false)) return;
  if (thread_.joinable()) thread_.join();
  if (isOpen()) {
    GX_STATUS status = GXStreamOff(handle_);
    if (status != GX_STATUS_SUCCESS)
      setError("GXStreamOff", status);
  }
}

void GalaxyCamera::close() {
  if (!isOpen()) return;
  GXCloseDevice(handle_);
  handle_ = nullptr;
  ds_handle_ = nullptr;
  if (rgb_buf_) { std::free(rgb_buf_); rgb_buf_ = nullptr; }
  if (raw8_buf_) { std::free(raw8_buf_); raw8_buf_ = nullptr; }
}

void GalaxyCamera::acquireLoop() {
  GX_STATUS status;
  PGX_FRAME_BUFFER frame = nullptr;
  uint64_t count = 0;
  time_t t_begin = 0, t_end = 0;
  // Bound shutdown latency even when software triggers have stopped. Two
  // consecutive 5 s waits used to exceed ros2 launch's shutdown deadline.
  const int dq_timeout = 100;

  while (running_.load()) {
    status = GXDQBuf(handle_, &frame, dq_timeout);
    if (status != GX_STATUS_SUCCESS) {
      if (status == GX_STATUS_TIMEOUT) continue;
      setError("GXDQBuf", status);
      break;
    }
    if (frame->nStatus != GX_FRAME_STATUS_SUCCESS) {
      // Abnormal frame: requeue and keep going.
      GXQBuf(handle_, frame);
      continue;
    }

    // Bayer -> RGB24 conversion into the reusable buffer.
    const uint8_t* rgb_ptr = nullptr;
    if (is_color_ && rgb_buf_) {
      switch (frame->nPixelFormat) {
        case GX_PIXEL_FORMAT_BAYER_GR8:
        case GX_PIXEL_FORMAT_BAYER_RG8:
        case GX_PIXEL_FORMAT_BAYER_GB8:
        case GX_PIXEL_FORMAT_BAYER_BG8: {
          VxInt32 dx = DxRaw8toRGB24(
              frame->pImgBuf, rgb_buf_, frame->nWidth, frame->nHeight,
              RAW2RGB_NEIGHBOUR, DX_PIXEL_COLOR_FILTER(color_filter_), false);
          if (dx == DX_OK) rgb_ptr = rgb_buf_;
          break;
        }
        case GX_PIXEL_FORMAT_BAYER_GR10:
        case GX_PIXEL_FORMAT_BAYER_RG10:
        case GX_PIXEL_FORMAT_BAYER_GB10:
        case GX_PIXEL_FORMAT_BAYER_BG10:
        case GX_PIXEL_FORMAT_BAYER_GR12:
        case GX_PIXEL_FORMAT_BAYER_RG12:
        case GX_PIXEL_FORMAT_BAYER_GB12:
        case GX_PIXEL_FORMAT_BAYER_BG12: {
          if (!raw8_buf_)
            raw8_buf_ = static_cast<uint8_t*>(std::malloc(payload_size_));
          VxInt32 dx = DxRaw16toRaw8(frame->pImgBuf, raw8_buf_, frame->nWidth,
                                     frame->nHeight, DX_BIT_2_9);
          if (dx == DX_OK)
            dx = DxRaw8toRGB24(raw8_buf_, rgb_buf_, frame->nWidth, frame->nHeight,
                               RAW2RGB_NEIGHBOUR,
                               DX_PIXEL_COLOR_FILTER(color_filter_), false);
          if (dx == DX_OK) rgb_ptr = rgb_buf_;
          break;
        }
        default:
          break;
      }
    }

    if (callback_) {
      GalaxyFrame gf;
      gf.width = frame->nWidth;
      gf.height = frame->nHeight;
      gf.frame_id = frame->nFrameID;
      gf.pixel_format = frame->nPixelFormat;
      gf.color_filter = color_filter_;
      gf.raw_data = static_cast<const uint8_t*>(frame->pImgBuf);
      gf.rgb_data = rgb_ptr;
      callback_(gf);
    }

    // Periodic log (~1 Hz).
    if (count == 0) time(&t_begin);
    ++count;
    time(&t_end);
    if (t_end - t_begin >= 1) {
      fprintf(stdout, "[GalaxyCamera %s] %llu fps, %ux%u frameID=%llu\n",
              cfg_.ip.c_str(), static_cast<unsigned long long>(count),
              frame->nWidth, frame->nHeight,
              static_cast<unsigned long long>(frame->nFrameID));
      count = 0;
    }

    // Requeue the SDK buffer (must happen after callback uses raw_data).
    status = GXQBuf(handle_, frame);
    if (status != GX_STATUS_SUCCESS) {
      setError("GXQBuf", status);
      break;
    }
  }
  fprintf(stdout, "[GalaxyCamera %s] acquisition thread exiting\n",
          cfg_.ip.c_str());
}

}  // namespace galaxy2
