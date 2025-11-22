//  Copyright 2024 Massachusetts Institute of Technology and
//  Institute of Automatic Control RWTH Aachen University
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//
//  Author: Kimera VIO Integration Team
//
//  GraphTimeCentricBackendAdapter implementation

#include "kimera-vio/integration/GraphTimeCentricBackendAdapter.h"
#include <iostream>
#include <sstream>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <iomanip>

#include "online_fgo_core/integration/KimeraIntegrationInterface.h"
#include "online_fgo_core/interface/ApplicationInterface.h"
#include "online_fgo_core/interface/LoggerInterface.h"
#include "online_fgo_core/interface/ParameterInterface.h"
#include "online_fgo_core/data/DataTypesFGO.h"

#include <map>

namespace {
  // Standalone application implementation for testing
  class StandaloneLogger : public fgo::core::LoggerInterface {
  public:
    void debug(const std::string& msg) override { std::cout << "[DEBUG] " << msg << std::endl; }
    void info(const std::string& msg) override { std::cout << "[INFO] " << msg << std::endl; }
    void warn(const std::string& msg) override { std::cout << "[WARN] " << msg << std::endl; }
    void error(const std::string& msg) override { std::cerr << "[ERROR] " << msg << std::endl; }
  };
  
  class StandaloneParameters : public fgo::core::ParameterInterface {
  public:
    bool hasParameter(const std::string& name) const override { return string_params_.count(name) > 0; }
    bool getBool(const std::string& name, bool default_value) override { return default_value; }
    int getInt(const std::string& name, int default_value) override { return default_value; }
    double getDouble(const std::string& name, double default_value) override { return default_value; }
    std::string getString(const std::string& name, const std::string& default_value) override {
      if (string_params_.count(name)) {
        return string_params_.at(name);
      }
      return default_value;
    }
    std::vector<double> getDoubleArray(const std::string& name, const std::vector<double>& default_value) override { return default_value; }
    std::vector<int> getIntArray(const std::string& name, const std::vector<int>& default_value) override { return default_value; }
    std::vector<std::string> getStringArray(const std::string& name, const std::vector<std::string>& default_value) override { return default_value; }
    void setBool(const std::string& name, bool value) override {}
    void setInt(const std::string& name, int value) override {}
    void setDouble(const std::string& name, double value) override {}
    void setString(const std::string& name, const std::string& value) override {
      string_params_[name] = value;
    }
    void setDoubleArray(const std::string& name, const std::vector<double>& value) override {}
    void setIntArray(const std::string& name, const std::vector<int>& value) override {}
    void setStringArray(const std::string& name, const std::vector<std::string>& value) override {}
    void loadFromYAML(const std::string& filename) override {}
  private:
    std::map<std::string, std::string> string_params_;
  };
  
  class StandaloneApp : public fgo::core::ApplicationInterface {
  public:
    StandaloneApp() : logger_(), params_() {}
    
    fgo::core::LoggerInterface& getLogger() override { return logger_; }
    fgo::core::ParameterInterface& getParameters() override { return params_; }
    fgo::core::TimeStamp now() const override { 
      return fgo::core::TimeStamp(std::chrono::system_clock::now().time_since_epoch().count() / 1e9);
    }
    std::string getName() const override { return "StandaloneApp"; }
    
  protected:
    template<typename MsgType>
    std::shared_ptr<fgo::core::PublisherInterface<MsgType>> createPublisherImpl(const std::string& topic) {
      return nullptr;
    }
    
  private:
    StandaloneLogger logger_;
    StandaloneParameters params_;
  };
}

namespace VIO {

GraphTimeCentricBackendAdapter::GraphTimeCentricBackendAdapter(
    const BackendParams& backend_params,
    const ImuParams& imu_params)
    : backend_params_(backend_params)
    , imu_params_(imu_params)
    , initialized_(false)
    , num_states_(0)
    , last_optimization_time_(0.0)
    , last_imu_timestamp_sec_(0.0) {
  LOG(INFO) << "GraphTimeCentricBackendAdapter: created";
  std::cout << "[Kimera-VIO] GraphTimeCentricBackendAdapter: ENABLED IMPLEMENTATION ACTIVE" << std::endl;
}

GraphTimeCentricBackendAdapter::~GraphTimeCentricBackendAdapter() {
  LOG(INFO) << "GraphTimeCentricBackendAdapter: destroyed";
}

bool GraphTimeCentricBackendAdapter::initialize() {
  if (initialized_) {
    LOG(WARNING) << "GraphTimeCentricBackendAdapter: already initialized";
    return true;
  }
  
  LOG(INFO) << "GraphTimeCentricBackendAdapter: initializing...";
  std::cout << "[Kimera-VIO] GraphTimeCentricBackendAdapter: INITIALIZING" << std::endl;
  
  try {
    // Create standalone application for testing
    standalone_app_ = std::make_unique<StandaloneApp>();
    
    std::string smootherTypeString;
    switch (backend_params_.smootherType_) {
      case 0:
        smootherTypeString = "ISAM2";
        break;
      case 1:
        smootherTypeString = "Batch";
        break;
      case 2:
        smootherTypeString = "IncrementalFixedLag";
        break;
      case 3:
        smootherTypeString = "BatchFixedLag";
        break;
      default:
        LOG(FATAL) << "Unknown smoother type: " << backend_params_.smootherType_;
    }
    standalone_app_->getParameters().setString("GNSSFGO.Optimizer.smootherType", smootherTypeString);

    // Create integration interface
    integration_interface_ = std::make_unique<fgo::integration::KimeraIntegrationInterface>(*standalone_app_);
    
    // Create integration parameters from Kimera params
    auto integration_params = createIntegrationParams();
    
    // Initialize the interface
    if (!integration_interface_->initialize(integration_params)) {
      LOG(ERROR) << "GraphTimeCentricBackendAdapter: failed to initialize interface";
      return false;
    }
    
    initialized_ = true;
    LOG(INFO) << "GraphTimeCentricBackendAdapter: initialized successfully";
    return true;
    
  } catch (const std::exception& e) {
    LOG(ERROR) << "GraphTimeCentricBackendAdapter: initialization failed: " << e.what();
    return false;
  }
}

bool GraphTimeCentricBackendAdapter::isInitialized() const {
  return initialized_;
}

bool GraphTimeCentricBackendAdapter::bufferNonKeyframeState(
    const Timestamp& timestamp,
    const gtsam::Pose3& pose,
    const gtsam::Vector3& velocity,
    const gtsam::imuBias::ConstantBias& bias) {
  if (!initialized_) {
    LOG(ERROR) << "GraphTimeCentricBackendAdapter: not initialized, cannot buffer state";
    return false;
  }
  
  std::lock_guard<std::mutex> lock(state_buffer_mutex_);
  
  BufferedState buffered_state;
  buffered_state.timestamp = timestamp;
  buffered_state.pose = pose;
  buffered_state.velocity = velocity;
  buffered_state.bias = bias;
  
  non_keyframe_buffer_.push_back(buffered_state);
  
  const double timestamp_sec = timestampToSeconds(timestamp);
  LOG(INFO) << "GraphTimeCentricBackendAdapter: buffered non-keyframe state at t=" 
            << std::fixed << std::setprecision(6) << timestamp_sec
            << " (buffer size: " << non_keyframe_buffer_.size() << ")";
  
  return true;
}

void GraphTimeCentricBackendAdapter::addKeyframeState(
    const Timestamp& timestamp,
    const gtsam::Pose3& pose,
    const gtsam::Vector3& velocity,
    const gtsam::imuBias::ConstantBias& bias) {
  if (!initialized_) {
    LOG(ERROR) << "GraphTimeCentricBackendAdapter: not initialized, cannot add keyframe";
    return;
  }
  
  const double keyframe_timestamp_sec = timestampToSeconds(timestamp);
  
  LOG(INFO) << "GraphTimeCentricBackendAdapter: adding keyframe at t="
            << std::fixed << std::setprecision(6) << keyframe_timestamp_sec;
  
  try {
    // Add the keyframe state directly
    auto keyframe_state_handle = integration_interface_->createStateAtTimestamp(keyframe_timestamp_sec);
    
    if (keyframe_state_handle.valid) {
      state_timestamps_.push_back(keyframe_timestamp_sec);
      num_states_++;
      
      LOG(INFO) << "GraphTimeCentricBackendAdapter: created keyframe state " 
                << keyframe_state_handle.index 
                << " at timestamp " << keyframe_state_handle.timestamp 
                << " (total states: " << num_states_ << ")";
    } else {
      LOG(WARNING) << "GraphTimeCentricBackendAdapter: failed to create keyframe state at t=" 
                   << keyframe_timestamp_sec;
    }
    
  } catch (const std::exception& e) {
    LOG(ERROR) << "GraphTimeCentricBackendAdapter: failed to add keyframe: " << e.what();
  }
}

bool GraphTimeCentricBackendAdapter::optimizeGraph() {
  if (!initialized_) {
    LOG(ERROR) << "GraphTimeCentricBackendAdapter: not initialized, cannot optimize";
    return false;
  }
  
  if (state_timestamps_.empty()) {
    LOG(WARNING) << "GraphTimeCentricBackendAdapter: no states added, skipping optimization";
    return false;
  }
  
  LOG(INFO) << "GraphTimeCentricBackendAdapter: optimizing graph with " 
            << state_timestamps_.size() << " states";
  
  try {
    // Trigger optimization through the interface
    auto result = integration_interface_->optimize();
    
    if (result.success) {
      if (result.num_states > 0) {
          last_optimization_time_ = state_timestamps_.back();
          LOG(INFO) << "GraphTimeCentricBackendAdapter: optimization succeeded"
                    << " - optimized " << result.num_states << " states"
                    << ", time: " << result.optimization_time_ms << " ms";
      } else {
          LOG(WARNING) << "GraphTimeCentricBackendAdapter: optimization skipped, no states were optimized.";
      }
      return true;
    } else {
      LOG(ERROR) << "GraphTimeCentricBackendAdapter: optimization failed: " << result.error_message;
      return false;
    }
    
  } catch (const std::exception& e) {
    LOG(ERROR) << "GraphTimeCentricBackendAdapter: optimization failed with exception: " << e.what();
    return false;
  }
}

bool GraphTimeCentricBackendAdapter::optimize(double /*timestep*/) {
  return optimizeGraph();
}

double GraphTimeCentricBackendAdapter::getLastOptimizationTime() const {
  return last_optimization_time_;
}

std::optional<gtsam::Pose3> GraphTimeCentricBackendAdapter::getOptimizedPoseAtTime(double timestamp) const {
  if (!initialized_ || !integration_interface_) {
    LOG(ERROR) << "GraphTimeCentricBackendAdapter: not initialized";
    return std::nullopt;
  }
  
  auto state_handle = findStateHandleNearTimestamp(timestamp);
  if (!state_handle.has_value()) {
    return std::nullopt;
  }
  
  auto nav_state = integration_interface_->getOptimizedState(state_handle.value());
  if (nav_state.has_value()) {
    return nav_state->pose();
  }
  
  return std::nullopt;
}

std::optional<gtsam::Vector3> GraphTimeCentricBackendAdapter::getOptimizedVelocityAtTime(double timestamp) const {
  if (!initialized_ || !integration_interface_) {
    return std::nullopt;
  }
  
  auto state_handle = findStateHandleNearTimestamp(timestamp);
  if (!state_handle.has_value()) {
    return std::nullopt;
  }
  
  auto nav_state = integration_interface_->getOptimizedState(state_handle.value());
  if (nav_state.has_value()) {
    return nav_state->velocity();
  }
  
  return std::nullopt;
}

std::optional<gtsam::imuBias::ConstantBias> GraphTimeCentricBackendAdapter::getOptimizedBiasAtTime(double timestamp) const {
  if (!initialized_ || !integration_interface_) {
    return std::nullopt;
  }
  
  auto state_handle = findStateHandleNearTimestamp(timestamp);
  if (!state_handle.has_value()) {
    return std::nullopt;
  }
  
  return integration_interface_->getOptimizedBias(state_handle.value());
}

std::optional<gtsam::Matrix> GraphTimeCentricBackendAdapter::getStateCovarianceAtTime(double timestamp) const {
  if (!initialized_ || !integration_interface_) {
    return std::nullopt;
  }
  
  auto state_handle = findStateHandleNearTimestamp(timestamp);
  if (!state_handle.has_value()) {
    return std::nullopt;
  }
  
  return integration_interface_->getStateCovariance(state_handle.value());
}

gtsam::Values GraphTimeCentricBackendAdapter::getLastResult() {
  gtsam::Values values;
  
  if (!initialized_ || state_timestamps_.empty()) {
    return values;
  }
  
  try {
    for (const auto& timestamp : state_timestamps_) {
      auto state_handle = findStateHandleNearTimestamp(timestamp);
      if (!state_handle.has_value()) {
        continue;
      }
      
      auto nav_state = integration_interface_->getOptimizedState(state_handle.value());
      auto bias = integration_interface_->getOptimizedBias(state_handle.value());
      
      if (nav_state.has_value()) {
        gtsam::Key pose_key = gtsam::Symbol('x', state_handle->index).key();
        gtsam::Key vel_key = gtsam::Symbol('v', state_handle->index).key();
        gtsam::Key bias_key = gtsam::Symbol('b', state_handle->index).key();
        
        values.insert(pose_key, nav_state->pose());
        values.insert(vel_key, nav_state->velocity());
        if (bias.has_value()) {
          values.insert(bias_key, bias.value());
        }
      }
    }
  } catch (const std::exception& e) {
    LOG(ERROR) << "GraphTimeCentricBackendAdapter: failed to get last result: " << e.what();
  }
  
  return values;
}

std::optional<gtsam::NavState> GraphTimeCentricBackendAdapter::getStateAtTime(Timestamp timestamp) {
  const double timestamp_sec = timestampToSeconds(timestamp);
  auto state_handle = findStateHandleNearTimestamp(timestamp_sec);
  if (!state_handle.has_value()) {
    return std::nullopt;
  }
  return integration_interface_->getOptimizedState(state_handle.value());
}

std::optional<gtsam::NavState> GraphTimeCentricBackendAdapter::getLatestState() {
  if (!initialized_ || !integration_interface_) {
    return std::nullopt;
  }
  return integration_interface_->getLatestOptimizedState();
}

std::optional<gtsam::imuBias::ConstantBias> GraphTimeCentricBackendAdapter::getLatestIMUBias() {
  if (!initialized_ || !integration_interface_) {
    return std::nullopt;
  }
  return integration_interface_->getLatestOptimizedBias();
}

std::optional<gtsam::Matrix> GraphTimeCentricBackendAdapter::getStateCovariance(Timestamp timestamp) {
  const double timestamp_sec = timestampToSeconds(timestamp);
  auto state_handle = findStateHandleNearTimestamp(timestamp_sec);
  if (!state_handle.has_value()) {
    return std::nullopt;
  }
  return integration_interface_->getStateCovariance(state_handle.value());
}

std::optional<gtsam::Matrix> GraphTimeCentricBackendAdapter::getLatestStateCovariance() {
  if (!initialized_ || state_timestamps_.empty()) {
    return std::nullopt;
  }
  
  const double latest_timestamp = state_timestamps_.back();
  auto state_handle = findStateHandleNearTimestamp(latest_timestamp);
  if (!state_handle.has_value()) {
    return std::nullopt;
  }
  return integration_interface_->getStateCovariance(state_handle.value());
}

bool GraphTimeCentricBackendAdapter::addIMUMeasurement(const ImuAccGyr& imu_measurement) {
  // ImuAccGyr is a 6x1 vector: [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
  // Note: This method cannot be used without timestamp - needs redesign
  LOG(WARNING) << "addIMUMeasurement called without timestamp - not implemented";
  return false;
}

size_t GraphTimeCentricBackendAdapter::addIMUMeasurements(const std::vector<ImuAccGyr>& imu_measurements) {
  if (!initialized_) {
    return 0;
  }
  
  // ImuAccGyr is a 6x1 vector without timestamps
  // This method cannot be used without timestamps - needs redesign
  LOG(WARNING) << "addIMUMeasurements called without timestamps - not implemented";
  return 0;
}

bool GraphTimeCentricBackendAdapter::addIMUMeasurements(const ImuStampS& timestamps,
                                                        const ImuAccGyrS& measurements) {
  if (!initialized_) {
    return false;
  }

  if (timestamps.cols() != measurements.cols()) {
    LOG(ERROR) << "Inconsistent IMU data size";
    return false;
  }

  std::vector<double> timestamps_sec;
  std::vector<Eigen::Vector3d> accels;
  std::vector<Eigen::Vector3d> gyros;
  std::vector<double> dts;

  timestamps_sec.reserve(timestamps.cols());
  accels.reserve(timestamps.cols());
  gyros.reserve(timestamps.cols());
  dts.reserve(timestamps.cols());

  for (int i = 0; i < timestamps.cols(); ++i) {
    timestamps_sec.push_back(static_cast<double>(timestamps(0, i)) / 1e9);
    accels.emplace_back(measurements.col(i).head<3>());
    gyros.emplace_back(measurements.col(i).tail<3>());
    if (i > 0) {
      dts.push_back(static_cast<double>(timestamps(0, i) - timestamps(0, i - 1)) / 1e9);
    } else {
      dts.push_back(0.0);
    }
  }
  if (!dts.empty() && dts.size() > 1) {
    dts[0] = dts[1];
  }
  
  integration_interface_->addIMUDataBatch(timestamps_sec, accels, gyros, dts);
  
  return true;
}

bool GraphTimeCentricBackendAdapter::addIMUTimestamps(const std::vector<double>& /*imu_timestamps*/) {
  LOG(WARNING) << "GraphTimeCentricBackendAdapter::addIMUTimestamps not implemented - use addIMUMeasurements instead";
  return true;
}

bool GraphTimeCentricBackendAdapter::preintegrateIMUBetweenStates(Timestamp /*t_i*/, Timestamp /*t_j*/) {
  LOG(INFO) << "GraphTimeCentricBackendAdapter::preintegrateIMUBetweenStates - "
            << "preintegration is handled automatically during optimization";
  return true;
}

bool GraphTimeCentricBackendAdapter::addKeyframeState(Timestamp timestamp, const gtsam::Pose3& pose_estimate) {
  gtsam::Vector3 velocity = gtsam::Vector3::Zero();
  gtsam::imuBias::ConstantBias bias;
  addKeyframeState(timestamp, pose_estimate, velocity, bias);
  return true;
}

bool GraphTimeCentricBackendAdapter::addKeyframeState(Timestamp timestamp, const gtsam::NavState& nav_state) {
  gtsam::imuBias::ConstantBias bias;
  addKeyframeState(timestamp, nav_state.pose(), nav_state.velocity(), bias);
  return true;
}

bool GraphTimeCentricBackendAdapter::addStateValues(unsigned long /*frame_id*/, double timestamp, const gtsam::NavState& navstate) {
  Timestamp kimera_timestamp = static_cast<Timestamp>(timestamp * 1e9);
  gtsam::imuBias::ConstantBias bias;
  addKeyframeState(kimera_timestamp, navstate.pose(), navstate.velocity(), bias);
  return true;
}

size_t GraphTimeCentricBackendAdapter::getNumStates() const {
  return num_states_;
}

size_t GraphTimeCentricBackendAdapter::getNumBufferedIMU() const {
  return 0; // IMU buffer not implemented in current design
}

size_t GraphTimeCentricBackendAdapter::getNumBufferedStates() const {
  std::lock_guard<std::mutex> lock(state_buffer_mutex_);
  return non_keyframe_buffer_.size();
}

std::string GraphTimeCentricBackendAdapter::getStatistics() const {
  std::ostringstream oss;
  oss << "GraphTimeCentricBackendAdapter Statistics:\n";
  oss << "  Initialized: " << (initialized_ ? "yes" : "no") << "\n";
  oss << "  Num states: " << num_states_ << "\n";
  oss << "  Num buffered non-keyframes: " << getNumBufferedStates() << "\n";
  oss << "  Last optimization time: " << last_optimization_time_ << " s\n";
  return oss.str();
}

double GraphTimeCentricBackendAdapter::timestampToSeconds(const Timestamp& timestamp) const {
  return static_cast<double>(timestamp) / 1e9;
}

Timestamp GraphTimeCentricBackendAdapter::secondsToTimestamp(double seconds) const {
  return static_cast<Timestamp>(seconds * 1e9);
}

fgo::integration::KimeraIntegrationParams GraphTimeCentricBackendAdapter::createIntegrationParams() const {
  fgo::integration::KimeraIntegrationParams params;
  
  params.use_isam2 = true;
  params.use_gp_priors = backend_params_.addBetweenStereoFactors_;
  params.optimize_on_keyframe = true;
  params.smoother_lag = 5.0;
  
  params.imu_rate = 200.0;
  params.accel_noise_sigma = imu_params_.acc_noise_density_;
  params.gyro_noise_sigma = imu_params_.gyro_noise_density_;
  params.accel_bias_rw_sigma = imu_params_.acc_random_walk_;
  params.gyro_bias_rw_sigma = imu_params_.gyro_random_walk_;
  params.gravity = imu_params_.n_gravity_.norm() > 0 ? imu_params_.n_gravity_ : Eigen::Vector3d(0.0, 0.0, -9.81);
  
  if (params.use_gp_priors) {
    params.gp_type = "WNOJ";
  }
  
  params.optimization_period = 0.1;
  
  LOG(INFO) << "Created integration params: use_isam2=" << params.use_isam2
            << ", use_gp=" << params.use_gp_priors
            << ", smoother_lag=" << params.smoother_lag;
  
  return params;
}

std::optional<fgo::integration::StateHandle> GraphTimeCentricBackendAdapter::findStateHandleNearTimestamp(double timestamp) const {
  if (state_timestamps_.empty()) {
    return std::nullopt;
  }
  
  auto it = std::min_element(state_timestamps_.begin(), state_timestamps_.end(),
                             [timestamp](double a, double b) {
                               return std::abs(a - timestamp) < std::abs(b - timestamp);
                             });
  
  if (it == state_timestamps_.end()) {
    return std::nullopt;
  }
  
  const double closest_time = *it;
  const double tolerance = 0.1;
  
  if (std::abs(closest_time - timestamp) > tolerance) {
    return std::nullopt;
  }
  
  const size_t state_index = std::distance(state_timestamps_.begin(), it);
  return fgo::integration::StateHandle(state_index, closest_time);
}

} // namespace VIO
