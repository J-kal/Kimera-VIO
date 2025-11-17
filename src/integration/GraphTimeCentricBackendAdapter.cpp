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
//  GraphTimeCentricBackendAdapter implementation using PIMPL pattern
//

#include "kimera-vio/integration/GraphTimeCentricBackendAdapter.h"
#include <iostream>
#include <sstream>
#include <chrono>
#include <algorithm>
#include <cmath>

// ============================================================================
// PIMPL INTERFACE - Abstract base class for implementation
// ============================================================================

namespace VIO {

// Forward declaration of implementation base class
class GraphTimeCentricBackendAdapter::Impl {
 public:
  virtual ~Impl() = default;
  
  // Initialization
  virtual bool initialize() = 0;
  virtual bool isInitialized() const = 0;
  
  // State management
  virtual bool bufferNonKeyframeState(const Timestamp& timestamp,
                                     const gtsam::Pose3& pose,
                                     const gtsam::Vector3& velocity,
                                     const gtsam::imuBias::ConstantBias& bias) = 0;
  virtual void addKeyframeState(const Timestamp& timestamp,
                                const gtsam::Pose3& pose,
                                const gtsam::Vector3& velocity,
                                const gtsam::imuBias::ConstantBias& bias) = 0;
  
  // Optimization
  virtual bool optimizeGraph() = 0;
  virtual double getLastOptimizationTime() const = 0;
  
  // Result retrieval
  virtual std::optional<gtsam::Pose3> getOptimizedPoseAtTime(double timestamp) const = 0;
  virtual std::optional<gtsam::Vector3> getOptimizedVelocityAtTime(double timestamp) const = 0;
  virtual std::optional<gtsam::imuBias::ConstantBias> getOptimizedBiasAtTime(double timestamp) const = 0;
  virtual std::optional<gtsam::Matrix> getStateCovarianceAtTime(double timestamp) const = 0;
  virtual gtsam::Values getLastResult() = 0;
  
  // New methods
  virtual std::optional<gtsam::NavState> getStateAtTime(Timestamp timestamp) = 0;
  virtual std::optional<gtsam::NavState> getLatestState() = 0;
  virtual std::optional<gtsam::imuBias::ConstantBias> getLatestIMUBias() = 0;
  virtual std::optional<gtsam::Matrix> getStateCovariance(Timestamp timestamp) = 0;
  virtual std::optional<gtsam::Matrix> getLatestStateCovariance() = 0;
  
  // IMU handling
  virtual bool addIMUMeasurement(const ImuAccGyr& imu_measurement) = 0;
  virtual size_t addIMUMeasurements(const std::vector<ImuAccGyr>& imu_measurements) = 0;
  virtual bool addIMUTimestamps(const std::vector<double>& imu_timestamps) = 0;
  virtual bool preintegrateIMUBetweenStates(Timestamp t_i, Timestamp t_j) = 0;
  
  // Legacy methods
  virtual bool addKeyframeState(Timestamp timestamp, const gtsam::Pose3& pose_estimate) = 0;
  virtual bool addKeyframeState(Timestamp timestamp, const gtsam::NavState& nav_state) = 0;
  virtual bool addStateValues(unsigned long frame_id, double timestamp, const gtsam::NavState& navstate) = 0;
  
  // Statistics
  virtual size_t getNumStates() const = 0;
  virtual size_t getNumBufferedIMU() const = 0;
  virtual size_t getNumBufferedStates() const = 0;
  virtual std::string getStatistics() const = 0;
  
  // Helper methods
  virtual double timestampToSeconds(const Timestamp& timestamp) const = 0;
  virtual Timestamp secondsToTimestamp(double seconds) const = 0;
};

} // namespace VIO

// ============================================================================
// ENABLED IMPLEMENTATION - Full integration with online_fgo_core
// ============================================================================

#include "online_fgo_core/integration/KimeraIntegrationInterface.h"
#include "online_fgo_core/interface/ApplicationInterface.h"
#include "online_fgo_core/interface/LoggerInterface.h"
#include "online_fgo_core/interface/ParameterInterface.h"
#include "online_fgo_core/data/DataTypesFGO.h"

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
    bool hasParameter(const std::string& name) const override { return false; }
    bool getBool(const std::string& name, bool default_value) override { return default_value; }
    int getInt(const std::string& name, int default_value) override { return default_value; }
    double getDouble(const std::string& name, double default_value) override { return default_value; }
    std::string getString(const std::string& name, const std::string& default_value) override { return default_value; }
    std::vector<double> getDoubleArray(const std::string& name, const std::vector<double>& default_value) override { return default_value; }
    std::vector<int> getIntArray(const std::string& name, const std::vector<int>& default_value) override { return default_value; }
    std::vector<std::string> getStringArray(const std::string& name, const std::vector<std::string>& default_value) override { return default_value; }
    void setBool(const std::string& name, bool value) override {}
    void setInt(const std::string& name, int value) override {}
    void setDouble(const std::string& name, double value) override {}
    void setString(const std::string& name, const std::string& value) override {}
    void setDoubleArray(const std::string& name, const std::vector<double>& value) override {}
    void setIntArray(const std::string& name, const std::vector<int>& value) override {}
    void setStringArray(const std::string& name, const std::vector<std::string>& value) override {}
    void loadFromYAML(const std::string& filename) override {}
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

// Full implementation when adapter is enabled
class GraphTimeCentricBackendAdapterImpl : public GraphTimeCentricBackendAdapter::Impl {
public:
  GraphTimeCentricBackendAdapterImpl(const BackendParams& backend_params,
                                     const ImuParams& imu_params)
      : backend_params_(backend_params)
      , imu_params_(imu_params)
      , initialized_(false)
      , num_states_(0)
      , last_optimization_time_(0.0)
      , last_imu_timestamp_sec_(0.0) {
    LOG(INFO) << "GraphTimeCentricBackendAdapter: created (enabled implementation)";
    std::cout << "[Kimera-VIO] GraphTimeCentricBackendAdapter: ENABLED IMPLEMENTATION ACTIVE" << std::endl;
  }
  
  ~GraphTimeCentricBackendAdapterImpl() override {
    LOG(INFO) << "GraphTimeCentricBackendAdapter: destroyed (enabled implementation)";
  }
  
  bool initialize() override {
    if (initialized_) {
      LOG(WARNING) << "GraphTimeCentricBackendAdapter: already initialized";
      return true;
    }
    
    LOG(INFO) << "GraphTimeCentricBackendAdapter: initializing...";
    std::cout << "[Kimera-VIO] GraphTimeCentricBackendAdapter: INITIALIZING" << std::endl;
    
    try {
      // Create standalone application for testing
      standalone_app_ = std::make_unique<StandaloneApp>();
      
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
  
  bool isInitialized() const override {
    return initialized_;
  }
  
  bool bufferNonKeyframeState(const Timestamp& timestamp,
                             const gtsam::Pose3& pose,
                             const gtsam::Vector3& velocity,
                             const gtsam::imuBias::ConstantBias& bias) override {
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
  
  void addKeyframeState(const Timestamp& timestamp,
                       const gtsam::Pose3& pose,
                       const gtsam::Vector3& velocity,
                       const gtsam::imuBias::ConstantBias& bias) override {
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
  
  bool optimizeGraph() override {
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
        last_optimization_time_ = state_timestamps_.back();
        
        LOG(INFO) << "GraphTimeCentricBackendAdapter: optimization succeeded"
                  << " - optimized " << result.num_states << " states"
                  << ", time: " << result.optimization_time_ms << " ms";
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
  
  double getLastOptimizationTime() const override {
    return last_optimization_time_;
  }
  
  std::optional<gtsam::Pose3> getOptimizedPoseAtTime(double timestamp) const override {
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
  
  std::optional<gtsam::Vector3> getOptimizedVelocityAtTime(double timestamp) const override {
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
  
  std::optional<gtsam::imuBias::ConstantBias> getOptimizedBiasAtTime(double timestamp) const override {
    if (!initialized_ || !integration_interface_) {
      return std::nullopt;
    }
    
    auto state_handle = findStateHandleNearTimestamp(timestamp);
    if (!state_handle.has_value()) {
      return std::nullopt;
    }
    
    return integration_interface_->getOptimizedBias(state_handle.value());
  }
  
  std::optional<gtsam::Matrix> getStateCovarianceAtTime(double timestamp) const override {
    if (!initialized_ || !integration_interface_) {
      return std::nullopt;
    }
    
    auto state_handle = findStateHandleNearTimestamp(timestamp);
    if (!state_handle.has_value()) {
      return std::nullopt;
    }
    
    return integration_interface_->getStateCovariance(state_handle.value());
  }
  
  gtsam::Values getLastResult() override {
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
  
  std::optional<gtsam::NavState> getStateAtTime(Timestamp timestamp) override {
    const double timestamp_sec = timestampToSeconds(timestamp);
    auto state_handle = findStateHandleNearTimestamp(timestamp_sec);
    if (!state_handle.has_value()) {
      return std::nullopt;
    }
    return integration_interface_->getOptimizedState(state_handle.value());
  }
  
  std::optional<gtsam::NavState> getLatestState() override {
    if (!initialized_ || !integration_interface_) {
      return std::nullopt;
    }
    return integration_interface_->getLatestOptimizedState();
  }
  
  std::optional<gtsam::imuBias::ConstantBias> getLatestIMUBias() override {
    if (!initialized_ || !integration_interface_) {
      return std::nullopt;
    }
    return integration_interface_->getLatestOptimizedBias();
  }
  
  std::optional<gtsam::Matrix> getStateCovariance(Timestamp timestamp) override {
    const double timestamp_sec = timestampToSeconds(timestamp);
    auto state_handle = findStateHandleNearTimestamp(timestamp_sec);
    if (!state_handle.has_value()) {
      return std::nullopt;
    }
    return integration_interface_->getStateCovariance(state_handle.value());
  }
  
  std::optional<gtsam::Matrix> getLatestStateCovariance() override {
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
  
  bool addIMUMeasurement(const ImuAccGyr& imu_measurement) override {
    // ImuAccGyr is a 6x1 vector: [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
    // Note: This method cannot be used without timestamp - needs redesign
    LOG(WARNING) << "addIMUMeasurement called without timestamp - not implemented";
    return false;
  }
  
    size_t addIMUMeasurements(const std::vector<ImuAccGyr>& imu_measurements) override {
    if (!initialized_) {
      return 0;
    }
    
    // ImuAccGyr is a 6x1 vector without timestamps
    // This method cannot be used without timestamps - needs redesign
    LOG(WARNING) << "addIMUMeasurements called without timestamps - not implemented";
    return 0;
  }
  
  bool addIMUTimestamps(const std::vector<double>& /*imu_timestamps*/) override {
    LOG(WARNING) << "GraphTimeCentricBackendAdapter::addIMUTimestamps not implemented - use addIMUMeasurements instead";
    return true;
  }
  
  bool preintegrateIMUBetweenStates(Timestamp /*t_i*/, Timestamp /*t_j*/) override {
    LOG(INFO) << "GraphTimeCentricBackendAdapter::preintegrateIMUBetweenStates - "
              << "preintegration is handled automatically during optimization";
    return true;
  }
  
  bool addKeyframeState(Timestamp timestamp, const gtsam::Pose3& pose_estimate) override {
    gtsam::Vector3 velocity = gtsam::Vector3::Zero();
    gtsam::imuBias::ConstantBias bias;
    addKeyframeState(timestamp, pose_estimate, velocity, bias);
    return true;
  }
  
  bool addKeyframeState(Timestamp timestamp, const gtsam::NavState& nav_state) override {
    gtsam::imuBias::ConstantBias bias;
    addKeyframeState(timestamp, nav_state.pose(), nav_state.velocity(), bias);
    return true;
  }
  
  bool addStateValues(unsigned long /*frame_id*/, double timestamp, const gtsam::NavState& navstate) override {
    Timestamp kimera_timestamp = static_cast<Timestamp>(timestamp * 1e9);
    gtsam::imuBias::ConstantBias bias;
    addKeyframeState(kimera_timestamp, navstate.pose(), navstate.velocity(), bias);
    return true;
  }
  
  size_t getNumStates() const override {
    return num_states_;
  }
  
  size_t getNumBufferedIMU() const override {
    return 0; // IMU buffer not implemented in current design
  }
  
  size_t getNumBufferedStates() const override {
    std::lock_guard<std::mutex> lock(state_buffer_mutex_);
    return non_keyframe_buffer_.size();
  }
  
  std::string getStatistics() const override {
    std::ostringstream oss;
    oss << "GraphTimeCentricBackendAdapter Statistics (Enabled):\n";
    oss << "  Initialized: " << (initialized_ ? "yes" : "no") << "\n";
    oss << "  Num states: " << num_states_ << "\n";
    oss << "  Num buffered non-keyframes: " << getNumBufferedStates() << "\n";
    oss << "  Last optimization time: " << last_optimization_time_ << " s\n";
    return oss.str();
  }
  
  double timestampToSeconds(const Timestamp& timestamp) const override {
    return static_cast<double>(timestamp) / 1e9;
  }
  
  Timestamp secondsToTimestamp(double seconds) const override {
    return static_cast<Timestamp>(seconds * 1e9);
  }

private:
  // Parameters
  BackendParams backend_params_;
  ImuParams imu_params_;
  
  // Integration interface to online_fgo_core
  std::unique_ptr<fgo::integration::KimeraIntegrationInterface> integration_interface_;
  std::unique_ptr<StandaloneApp> standalone_app_;
  
  // State
  bool initialized_;
  size_t num_states_;
  double last_optimization_time_;
  double last_imu_timestamp_sec_;
  std::vector<double> state_timestamps_;
  
  // Buffering
  struct BufferedState {
    Timestamp timestamp;
    gtsam::Pose3 pose;
    gtsam::Vector3 velocity;
    gtsam::imuBias::ConstantBias bias;
    
    bool operator<(const BufferedState& other) const {
      return timestamp < other.timestamp;
    }
  };
  std::vector<BufferedState> non_keyframe_buffer_;
  mutable std::mutex state_buffer_mutex_;
  
  // Helper methods
  fgo::integration::KimeraIntegrationParams createIntegrationParams() const {
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
  
  std::optional<fgo::integration::StateHandle> findStateHandleNearTimestamp(double timestamp) const {
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
};

} // namespace VIO

// ============================================================================
// PUBLIC WRAPPER IMPLEMENTATION - Uses PIMPL pattern
// ============================================================================

namespace VIO {

// Constructor - creates the appropriate implementation
GraphTimeCentricBackendAdapter::GraphTimeCentricBackendAdapter(
    const BackendParams& backend_params,
    const ImuParams& imu_params)
    : pimpl_(std::make_unique<GraphTimeCentricBackendAdapterImpl>(backend_params, imu_params)) {
}

// Destructor - defined in .cpp for unique_ptr with forward-declared type
GraphTimeCentricBackendAdapter::~GraphTimeCentricBackendAdapter() = default;

// All public methods delegate to implementation
bool GraphTimeCentricBackendAdapter::initialize() {
  return pimpl_->initialize();
}

bool GraphTimeCentricBackendAdapter::isInitialized() const {
  return pimpl_->isInitialized();
}

bool GraphTimeCentricBackendAdapter::bufferNonKeyframeState(
    const Timestamp& timestamp,
    const gtsam::Pose3& pose,
    const gtsam::Vector3& velocity,
    const gtsam::imuBias::ConstantBias& bias) {
  return pimpl_->bufferNonKeyframeState(timestamp, pose, velocity, bias);
}

void GraphTimeCentricBackendAdapter::addKeyframeState(
    const Timestamp& timestamp,
    const gtsam::Pose3& pose,
    const gtsam::Vector3& velocity,
    const gtsam::imuBias::ConstantBias& bias) {
  pimpl_->addKeyframeState(timestamp, pose, velocity, bias);
}

bool GraphTimeCentricBackendAdapter::optimizeGraph() {
  return pimpl_->optimizeGraph();
}

bool GraphTimeCentricBackendAdapter::optimize(double /*timestep*/) {
  return pimpl_->optimizeGraph();
}

double GraphTimeCentricBackendAdapter::getLastOptimizationTime() const {
  return pimpl_->getLastOptimizationTime();
}

std::optional<gtsam::Pose3> GraphTimeCentricBackendAdapter::getOptimizedPoseAtTime(double timestamp) const {
  return pimpl_->getOptimizedPoseAtTime(timestamp);
}

std::optional<gtsam::Vector3> GraphTimeCentricBackendAdapter::getOptimizedVelocityAtTime(double timestamp) const {
  return pimpl_->getOptimizedVelocityAtTime(timestamp);
}

std::optional<gtsam::imuBias::ConstantBias> GraphTimeCentricBackendAdapter::getOptimizedBiasAtTime(double timestamp) const {
  return pimpl_->getOptimizedBiasAtTime(timestamp);
}

std::optional<gtsam::Matrix> GraphTimeCentricBackendAdapter::getStateCovarianceAtTime(double timestamp) const {
  return pimpl_->getStateCovarianceAtTime(timestamp);
}

gtsam::Values GraphTimeCentricBackendAdapter::getLastResult() {
  return pimpl_->getLastResult();
}

std::optional<gtsam::NavState> GraphTimeCentricBackendAdapter::getStateAtTime(Timestamp timestamp) {
  return pimpl_->getStateAtTime(timestamp);
}

std::optional<gtsam::NavState> GraphTimeCentricBackendAdapter::getLatestState() {
  return pimpl_->getLatestState();
}

std::optional<gtsam::imuBias::ConstantBias> GraphTimeCentricBackendAdapter::getLatestIMUBias() {
  return pimpl_->getLatestIMUBias();
}

std::optional<gtsam::Matrix> GraphTimeCentricBackendAdapter::getStateCovariance(Timestamp timestamp) {
  return pimpl_->getStateCovariance(timestamp);
}

std::optional<gtsam::Matrix> GraphTimeCentricBackendAdapter::getLatestStateCovariance() {
  return pimpl_->getLatestStateCovariance();
}

bool GraphTimeCentricBackendAdapter::addIMUMeasurement(const ImuAccGyr& imu_measurement) {
  return pimpl_->addIMUMeasurement(imu_measurement);
}

size_t GraphTimeCentricBackendAdapter::addIMUMeasurements(const std::vector<ImuAccGyr>& imu_measurements) {
  return pimpl_->addIMUMeasurements(imu_measurements);
}

bool GraphTimeCentricBackendAdapter::addIMUTimestamps(const std::vector<double>& imu_timestamps) {
  return pimpl_->addIMUTimestamps(imu_timestamps);
}

bool GraphTimeCentricBackendAdapter::preintegrateIMUBetweenStates(Timestamp t_i, Timestamp t_j) {
  return pimpl_->preintegrateIMUBetweenStates(t_i, t_j);
}

bool GraphTimeCentricBackendAdapter::addKeyframeState(Timestamp timestamp, const gtsam::Pose3& pose_estimate) {
  return pimpl_->addKeyframeState(timestamp, pose_estimate);
}

bool GraphTimeCentricBackendAdapter::addKeyframeState(Timestamp timestamp, const gtsam::NavState& nav_state) {
  return pimpl_->addKeyframeState(timestamp, nav_state);
}

bool GraphTimeCentricBackendAdapter::addStateValues(unsigned long frame_id, double timestamp, const gtsam::NavState& navstate) {
  return pimpl_->addStateValues(frame_id, timestamp, navstate);
}

size_t GraphTimeCentricBackendAdapter::getNumStates() const {
  return pimpl_->getNumStates();
}

size_t GraphTimeCentricBackendAdapter::getNumBufferedIMU() const {
  return pimpl_->getNumBufferedIMU();
}

size_t GraphTimeCentricBackendAdapter::getNumBufferedStates() const {
  return pimpl_->getNumBufferedStates();
}

std::string GraphTimeCentricBackendAdapter::getStatistics() const {
  return pimpl_->getStatistics();
}

double GraphTimeCentricBackendAdapter::timestampToSeconds(const Timestamp& timestamp) const {
  return pimpl_->timestampToSeconds(timestamp);
}

Timestamp GraphTimeCentricBackendAdapter::secondsToTimestamp(double seconds) const {
  return pimpl_->secondsToTimestamp(seconds);
}

} // namespace VIO
