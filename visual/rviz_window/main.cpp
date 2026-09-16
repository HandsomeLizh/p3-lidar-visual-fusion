// Read-only visualization. No mapper settings or vehicle commands are changed.
#include <algorithm>
#include <cmath>
#include <csignal>
#include <condition_variable>
#include <cstdio>
#include <iostream>
#include <map>
#include <mutex>
#include <thread>
#include <deque>
#include <QApplication>
#include <QCheckBox>
#include <QCloseEvent>
#include <QComboBox>
#include <QElapsedTimer>
#include <QFile>
#include <QHBoxLayout>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLabel>
#include <QMainWindow>
#include <QMouseEvent>
#include <QPushButton>
#include <QScreen>
#include <QSplitter>
#include <QTimer>
#include <QVBoxLayout>
#include <QWindow>
#include <OgreCamera.h>
#include <OgreViewport.h>
#include <OgreRenderTarget.h>
#include <OgreRenderWindow.h>
#include <rclcpp/rclcpp.hpp>
#include <rviz_common/config.hpp>
#include <rviz_common/display_group.hpp>
#include <rviz_common/logging.hpp>
#include <rviz_common/properties/vector_property.hpp>
#include <rviz_common/render_panel.hpp>
#include <rviz_common/ros_integration/ros_node_abstraction.hpp>
#include <rviz_common/tool_manager.hpp>
#include <rviz_common/view_controller.hpp>
#include <rviz_common/view_manager.hpp>
#include <rviz_common/visualization_manager.hpp>
#include <rviz_common/yaml_config_reader.hpp>
#include <rviz_rendering/render_window.hpp>

QJsonObject read_json(const QString & path) {
  QFile file(path);
  return file.open(QIODevice::ReadOnly) ? QJsonDocument::fromJson(file.readAll()).object() : QJsonObject();
}

// These tiny IPC snapshots are disposable UI state, not durable map results.
// QSaveFile::commit() calls fdatasync: a captured Jetson backtrace showed the
// GUI blocked there for seconds. Keep ALL writes off the GUI thread and do not
// force storage durability for heartbeats. Pending work is bounded per filename.
class VisualStateWriter {
 public:
  explicit VisualStateWriter(QString runtime) : runtime_(std::move(runtime)), worker_([this]() { run(); }) {}
  ~VisualStateWriter() {
    { std::lock_guard<std::mutex> lock(mutex_); stopping_ = true; pending_.clear(); }
    ready_.notify_one();
    worker_.join();
  }
  void submit(const QString & name, const QJsonObject & value) {
    { std::lock_guard<std::mutex> lock(mutex_); pending_[name] = QJsonDocument(value).toJson(); }
    ready_.notify_one();
  }
 private:
  void run() {
    while (true) {
      std::map<QString, QByteArray> batch;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        ready_.wait(lock, [this]() { return stopping_ || !pending_.empty(); });
        if (stopping_) return;
        batch.swap(pending_);
      }
      for (const auto & item : batch) {
        const auto destination = runtime_ + "/" + item.first;
        const auto temporary = destination + ".tmp";
        QFile file(temporary);
        if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate)) continue;
        const bool written = file.write(item.second) == item.second.size();
        file.close();
        if (written) std::rename(temporary.toLocal8Bit().constData(), destination.toLocal8Bit().constData());
      }
    }
  }
  QString runtime_;
  std::mutex mutex_;
  std::condition_variable ready_;
  std::map<QString, QByteArray> pending_;
  bool stopping_ = false;
  std::thread worker_;
};

class DemoWindow : public QMainWindow {
 public:
  DemoWindow(const QString & config_path, const QString & runtime,
             std::shared_ptr<rviz_common::ros_integration::RosNodeAbstraction> node)
      : runtime_(runtime), state_writer_(runtime), node_(std::move(node)) {
    steady_.start();
    setWindowTitle(QStringLiteral("定位与建图"));
    auto * body = new QWidget(this);
    auto * layout = new QVBoxLayout(body);
    layout->setContentsMargins(6, 6, 6, 6);
    auto * bar = new QHBoxLayout();
    auto * global = new QPushButton(QStringLiteral("全局视角"));
    auto * top = new QPushButton(QStringLiteral("俯视全局"));
    auto * near = new QPushButton(QStringLiteral("车辆近景"));
    automatic_ = new QCheckBox(QStringLiteral("自动容纳完整地图"));
    automatic_->setChecked(true);
    for (auto * button : {global, top, near}) bar->addWidget(button);
    bar->addWidget(automatic_);
    bar->addWidget(new QLabel(QStringLiteral("参考网格")));
    auto * spacing = new QComboBox();
    for (double value : {1., 5., 10., 20.}) spacing->addItem(QString::number(value) + " m", value);
    spacing->setCurrentIndex(2);
    bar->addWidget(spacing);
    bar->addStretch();
    bar->addWidget(new QLabel(QStringLiteral("左拖旋转 · 滚轮缩放 · Shift＋左拖平移")));
    layout->addLayout(bar);
    splitter_ = new QSplitter(Qt::Horizontal, body);
    auto * left = new QWidget(splitter_);
    auto * left_layout = new QVBoxLayout(left);
    left_layout->setContentsMargins(0, 0, 0, 0);
    caption_ = new QLabel(QStringLiteral("三维点云 · 橙线：轨迹 · 青色箭头：车辆"));
    left_layout->addWidget(caption_);
    render_ = new rviz_common::RenderPanel(left);
    left_layout->addWidget(render_, 1);
    placeholder_ = new QLabel(QStringLiteral("正在加载轨迹、位姿与相机画面…"), splitter_);
    placeholder_->setMinimumWidth(700);
    splitter_->addWidget(left);
    splitter_->addWidget(placeholder_);
    splitter_->setChildrenCollapsible(false);
    splitter_->setSizes({1050, 780});
    layout->addWidget(splitter_, 1);
    setCentralWidget(body);
    resize(1840, 1000);
    // Materialize the native parent hierarchy before Ogre binds its X11 drawable.
    winId();
    layout->activate();
    render_->getRenderWindow()->resize(1024, 768);
    render_->getRenderWindow()->initialize();
    manager_ = new rviz_common::VisualizationManager(render_, node_, nullptr,
                                                     node_->get_raw_node()->get_clock());
    render_->initialize(manager_);
    manager_->initialize();
    rviz_common::YamlConfigReader reader;
    rviz_common::Config config;
    reader.readFile(config, config_path);
    if (reader.error()) throw std::runtime_error(reader.errorMessage().toStdString());
    manager_->load(config.mapGetChild("Visualization Manager"));
    if (manager_->getToolManager()->numTools() > 1)
      manager_->getToolManager()->setCurrentTool(manager_->getToolManager()->getTool(1));
    manager_->startUpdate();
    connect(manager_, &rviz_common::VisualizationManager::preUpdate, this, [this]() {
      const auto now = steady_.elapsed();
      if (last_update_ms_ >= 0) {
        update_gaps_.push_back(now-last_update_ms_);
        if (update_gaps_.size() > 120) update_gaps_.pop_front();
      }
      last_update_ms_ = now;
    });
    QTimer::singleShot(10000, this, [this]() {
      auto * viewport = rviz_rendering::RenderWindowOgreAdapter::getOgreViewport(render_->getRenderWindow());
      auto * target = viewport->getTarget();
      auto * camera = viewport->getCamera();
      std::cout << "Render audit frames=" << manager_->getFrameCount()
                << " root=" << manager_->getRootDisplayGroup()->isEnabled()
                << " exposed=" << render_->getRenderWindow()->isExposed()
                << " target=" << target->getWidth() << "x" << target->getHeight()
                << " active=" << target->isActive() << " auto=" << target->isAutoUpdated()
                << " viewport=" << viewport->getActualWidth() << "x" << viewport->getActualHeight()
                << " background=" << viewport->getBackgroundColour()
                << " camera=" << camera->getName() << " far=" << camera->getFarClipDistance()
                << " position=" << camera->getDerivedPosition() << std::endl;
      if (qEnvironmentVariableIsSet("T3_RENDER_AUDIT"))
        render_->getRenderWindow()->captureScreenShot((runtime_ + "/render_audit.png").toStdString());
    });
    connect(spacing, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this, spacing](int) {
      grid_spacing_ = spacing->currentData().toDouble();
      update_grid();
    });
    update_grid();
    connect(global, &QPushButton::clicked, this, [this]() {
      pitch_ = 1.05; automatic_->setChecked(true); fit_global();
    });
    connect(top, &QPushButton::clicked, this, [this]() {
      pitch_ = 1.565; automatic_->setChecked(true); fit_global();
    });
    connect(near, &QPushButton::clicked, this, [this]() {
      automatic_->setChecked(false);
      auto * view = manager_->getViewManager()->getCurrent();
      view->subProp("Target Frame")->setValue("base_link");
      set_focal(view, Ogre::Vector3::ZERO);
      view->subProp("Distance")->setValue(45.0);
      view->subProp("Pitch")->setValue(0.85);
      caption_->setText(QStringLiteral("三维点云 · 车辆近景（点击“全局视角”恢复）"));
    });
    connect(automatic_, &QCheckBox::toggled, this, [this](bool enabled) {
      if (enabled) fit_global();
    });
    auto * timer = new QTimer(this);
    connect(timer, &QTimer::timeout, this, [this]() {
      if (!rclcpp::ok()) { close(); return; }
      set_fps(steady_.elapsed()-last_interaction_ms_ < 2000 ? 60 : 30);
      attach_monitor();
      // On this Jetson/Qt combination the first resize arrives before expose,
      // so RViz's resize handler can leave the Ogre target at 0 x 0. Synchronize
      // the actual drawable size explicitly after the native window is visible.
      auto * drawable = render_->getRenderWindow();
      auto * viewport = rviz_rendering::RenderWindowOgreAdapter::getOgreViewport(drawable);
      auto * target = dynamic_cast<Ogre::RenderWindow *>(viewport->getTarget());
      const auto ratio = drawable->devicePixelRatio();
      const unsigned int width = std::max(1, int(drawable->width()*ratio));
      const unsigned int height = std::max(1, int(drawable->height()*ratio));
      if (drawable->isExposed() && (target->getWidth() != width || target->getHeight() != height)) {
        target->resize(width, height);
        target->windowMovedOrResized();
        viewport->getCamera()->setAspectRatio(float(width)/height);
      }
      render_->getRenderWindow()->renderNow();
      state_writer_.submit("render_status.json", QJsonObject{{"qt_width", drawable->width()}, {"qt_height", drawable->height()},
          {"ogre_width", int(target->getWidth())}, {"ogre_height", int(target->getHeight())},
          {"exposed", drawable->isExposed()}, {"frames", int(manager_->getFrameCount())},
          {"target_fps", fps_}, {"monotonic_ms", double(steady_.elapsed())},
          {"maximum_recent_update_gap_ms", double(update_gaps_.empty() ? 0 : *std::max_element(update_gaps_.begin(), update_gaps_.end()))}});
      auto data = read_json(runtime_ + "/view_bounds.json");
      if (!data.isEmpty() && (data != bounds_ || render_->size() != fitted_size_)) {
        bounds_ = data;
        update_grid();
        if (automatic_->isChecked()) fit_global();
      }
    });
    timer->start(500);
    // Direct mouse navigation disables auto-fit so new data never fights the user.
    render_->getRenderWindow()->installEventFilter(this);
    render_->installEventFilter(this);
    resize(1840, 1000);
  }

  ~DemoWindow() override {
    // The foreign monitor is our own service. Do not close any unrelated window.
    if (foreign_) { foreign_->setParent(nullptr); foreign_->hide(); }
    delete manager_;
  }

 protected:
  bool eventFilter(QObject * watched, QEvent * event) override {
    const bool press = event->type() == QEvent::MouseButtonPress || event->type() == QEvent::Wheel;
    const bool drag = event->type() == QEvent::MouseMove && static_cast<QMouseEvent *>(event)->buttons() != Qt::NoButton;
    if (press)
      automatic_->setChecked(false);
    if (press || drag) { last_interaction_ms_ = steady_.elapsed(); set_fps(60); }
    return QMainWindow::eventFilter(watched, event);
  }

 private:
  void set_fps(int value) {
    if (fps_ == value) return;
    fps_ = value;
    manager_->getRootDisplayGroup()->subProp("Global Options")->subProp("Frame Rate")->setValue(value);
  }
  void update_grid() {
    auto * grid = manager_->getRootDisplayGroup()->subProp(QStringLiteral("参考网格"));
    grid->subProp("Cell Size")->setValue(grid_spacing_);
    auto low = bounds_["min"].toArray(), high = bounds_["max"].toArray();
    if (low.size() == 3 && high.size() == 3) {
      const double span = std::max(high[0].toDouble()-low[0].toDouble(), high[1].toDouble()-low[1].toDouble());
      const int cells = 2 * int(std::ceil((span/grid_spacing_ + 6)/2));
      grid->subProp("Plane Cell Count")->setValue(std::max(20, cells));
      auto * offset = dynamic_cast<rviz_common::properties::VectorProperty *>(grid->subProp("Offset"));
      if (offset) offset->setVector(Ogre::Vector3(
        std::round((low[0].toDouble()+high[0].toDouble())/2/grid_spacing_)*grid_spacing_,
        std::round((low[1].toDouble()+high[1].toDouble())/2/grid_spacing_)*grid_spacing_, 0));
    }
    state_writer_.submit("view_settings.json", QJsonObject{{"reference_grid_m", grid_spacing_}});
  }

  static void set_focal(rviz_common::ViewController * view, const Ogre::Vector3 & point) {
    auto * focal = dynamic_cast<rviz_common::properties::VectorProperty *>(view->subProp("Focal Point"));
    if (focal) focal->setVector(point);
  }

  void fit_global() {
    auto low = bounds_["min"].toArray(), high = bounds_["max"].toArray();
    if (low.size() != 3 || high.size() != 3) return;
    Ogre::Vector3 a, b;
    for (int i = 0; i < 3; ++i) {
      a[i] = low[i].toDouble(); b[i] = high[i].toDouble();
      if (!std::isfinite(a[i]) || !std::isfinite(b[i]) || b[i] < a[i]) return;
    }
    auto * view = manager_->getViewManager()->getCurrent();
    auto * camera = view->getCamera();
    const double half_vertical = camera->getFOVy().valueRadians() / 2.0;
    const double aspect = double(std::max(1, render_->width())) / std::max(1, render_->height());
    const double half_angle = std::min(half_vertical, std::atan(std::tan(half_vertical) * aspect));
    const double radius = std::max(10.0, double((b-a).length()) / 2.0);
    // Fit a sphere enclosing all preview points and trajectory, with a 10% margin.
    const double distance = 1.1 * radius / std::sin(std::max(0.05, half_angle));
    view->subProp("Target Frame")->setValue("map");
    set_focal(view, (a+b)/2.0f);
    view->subProp("Distance")->setValue(distance);
    view->subProp("Pitch")->setValue(pitch_);
    view->subProp("Yaw")->setValue(0.8);
    fitted_size_ = render_->size();
    caption_->setText(QStringLiteral("三维点云 · 全局范围 %1 × %2 m · 橙线：轨迹 · 青色：车辆")
                      .arg(b.x-a.x, 0, 'f', 0).arg(b.y-a.y, 0, 'f', 0));
  }

  void attach_monitor() {
    if (foreign_) return;
    const auto info = read_json(runtime_ + "/monitor_window.json");
    const auto pid = info["pid"].toInt();
    const auto wid = info["window"].toVariant().toULongLong();
    if (!pid || !wid) return;
    QFile command(QString("/proc/%1/cmdline").arg(pid));
    if (!command.open(QIODevice::ReadOnly) || !command.readAll().contains("visual_monitor.py")) return;
    foreign_ = QWindow::fromWinId(WId(wid));
    auto * container = QWidget::createWindowContainer(foreign_, nullptr);
    container->setMinimumWidth(700);
    splitter_->replaceWidget(1, container);
    container->show();
    delete placeholder_;
    placeholder_ = nullptr;
    splitter_->setSizes({1050, 780});
    std::cout << "Unified RViz/monitor window ready" << std::endl;
  }

  QString runtime_;
  VisualStateWriter state_writer_;
  QElapsedTimer steady_;
  qint64 last_interaction_ms_ = -10000;
  qint64 last_update_ms_ = -1;
  std::deque<qint64> update_gaps_;
  int fps_ = 30;
  std::shared_ptr<rviz_common::ros_integration::RosNodeAbstraction> node_;
  rviz_common::VisualizationManager * manager_ = nullptr;
  rviz_common::RenderPanel * render_ = nullptr;
  QSplitter * splitter_ = nullptr;
  QCheckBox * automatic_ = nullptr;
  QLabel * caption_ = nullptr;
  QLabel * placeholder_ = nullptr;
  QWindow * foreign_ = nullptr;
  QJsonObject bounds_;
  QSize fitted_size_;
  double pitch_ = 1.05;
  double grid_spacing_ = 10.0;
};

int main(int argc, char ** argv) {
  if (argc < 3) { std::cerr << "usage: t3_visual_window CONFIG RUNTIME [--ros-args ...]\n"; return 2; }
  const QString config = argv[1], runtime = argv[2];
  rclcpp::init(argc, argv);
  int qt_argc = 1;
  QApplication app(qt_argc, argv);
  rviz_common::install_rviz_rendering_log_handlers();
  app.setApplicationName("T3 Localization and Mapping");
  auto node = std::make_shared<rviz_common::ros_integration::RosNodeAbstraction>("t3_rviz");
  int result = 0;
  {
    DemoWindow window(config, runtime, node);
    window.showMaximized();
    result = app.exec();
  }
  rclcpp::shutdown();
  return result;
}
