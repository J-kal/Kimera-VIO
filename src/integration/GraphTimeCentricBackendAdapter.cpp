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
//

#include "kimera-vio/integration/GraphTimeCentricBackendAdapter.h"
#include <iostream>
#include <sstream>
#include <chrono>

#ifdef ENABLE_GRAPH_TIME_CENTRIC_ADAPTER
#include "online_fgo_core/integration/KimeraIntegrationInterface.h"
#include "online_fgo_core/interface/ApplicationInterface.h"
#include "online_fgo_core/interface/LoggerInterface.h"
#include "online_fgo_core/interface/ParameterInterface.h"
#include "online_fgo_core/data/DataTypesFGO.h"

// Standalone application implementation for testing
namespace {
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
    
    template<typename T>
    T getParameter(const std::string& name) const { return T(); }
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
#endif

namespace VIO {

GraphTimeCentricBackendAdapter::GraphTimeCentricBackendAdapter(
    const BackendParams& backend_params,
    const ImuParams& imu_params)
    : backend_params_(backend_params)
    , imu_params_(imu_params)
    , initialized_(false)
    , num_states_(0)
    , last_optimization_time_(0.0) {
  
  LOG(INFO) << "GraphTimeCentricBackendAdapter: created";
}

GraphTimeCentricBackendAdapter::~GraphTimeCentricBackendAdapter() {
  LOG(INFO) << "GraphTimeCentricBackendAdapter: destroyed";
}

// TODO(KIMERA_ADAPTER_5): Initialize the integration interface
bool GraphTimeCentricBackendAdapter::initialize() {
#ifdef ENABLE_GRAPH_TIME_CENTRIC_ADAPTER
  if (initialized_) {
    LOG(WARNING) << "GraphTimeCentricBackendAdapter: already initialized";
    return true;
  }
  
  LOG(INFO) << "GraphTimeCentricBackendAdapter: initializing...";
  
  try {
    // Create standalone application for testing
    app_ = std::make_shared<StandaloneApp>();
    
    // Create integration interface
    interface_ = std::make_shared<fgo::integration::KimeraIntegrationInterface>(app_);
    
    // Create integration parameters from Kimera params
    auto integration_params = createIntegrationParams();
    
    // Initialize the interface
    if (!interface_->initialize(integration_params)) {
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
#else
  LOG(WARNING) << "GraphTimeCentricBackendAdapter: ENABLE_GRAPH_TIME_CENTRIC_ADAPTER not defined";
  return false;
#endif
}

// Buffer non-keyframe state for later addition
bool GraphTimeCentricBackendAdapter::bufferNonKeyframeState(
    const Timestamp& timestamp,
    const gtsam::Pose3& pose,
    const gtsam::Vector3& velocity,
    const gtsam::imuBias::ConstantBias& bias) {
#ifdef ENABLE_GRAPH_TIME_CENTRIC_ADAPTER
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
#else
  LOG(WARNING) << "GraphTimeCentricBackendAdapter: ENABLE_GRAPH_TIME_CENTRIC_ADAPTER not defined";
  return false;
#endif
}

// TODO(KIMERA_ADAPTER_6): Add keyframe state to graph (creates timestamp-indexed state)
// UPDATED: Now processes buffered non-keyframes first
void GraphTimeCentricBackendAdapter::addKeyframeState(
    const Timestamp& timestamp,
    const gtsam::Pose3& pose,
    const gtsam::Vector3& velocity,
    const gtsam::imuBias::ConstantBias& bias) {
#ifdef ENABLE_GRAPH_TIME_CENTRIC_ADAPTER
  if (!initialized_) {
    LOG(ERROR) << "GraphTimeCentricBackendAdapter: not initialized, cannot add keyframe";
    return;
  }
  
  const double keyframe_timestamp_sec = timestampToSeconds(timestamp);
  
  LOG(INFO) << "GraphTimeCentricBackendAdapter: processing keyframe at t=" 
            << std::fixed << std::setprecision(6) << keyframe_timestamp_sec;
  
  try {
    // STEP 1: Process all buffered non-keyframe states in chronological order
    std::vector<BufferedState> states_to_add;
    {
      std::lock_guard<std::mutex> lock(state_buffer_mutex_);
      
      if (!non_keyframe_buffer_.empty()) {
        // Sort buffered states by timestamp
        std::sort(non_keyframe_buffer_.begin(), non_keyframe_buffer_.end());
        
        // Copy to local vector for processing
        states_to_add = non_keyframe_buffer_;
        
        LOG(INFO) << "GraphTimeCentricBackendAdapter: processing " 
                  << states_to_add.size() << " buffered non-keyframe states";
        
        // Clear the buffer
        non_keyframe_buffer_.clear();
      }
    }
    
    // Add buffered states (outside lock)
    for (const auto& buffered_state : states_to_add) {
      const double buffered_timestamp_sec = timestampToSeconds(buffered_state.timestamp);
      
      // Skip if timestamp is after the keyframe (shouldn't happen, but be safe)
      if (buffered_timestamp_sec >= keyframe_timestamp_sec) {
        LOG(WARNING) << "GraphTimeCentricBackendAdapter: skipping buffered state at t=" 
                     << buffered_timestamp_sec << " (after keyframe at t=" 
                     << keyframe_timestamp_sec << ")";
        continue;
      }
      
      gtsam::NavState nav_state(buffered_state.pose, buffered_state.velocity);
      auto state_handle = interface_->createStateAtTimestamp(
          buffered_timestamp_sec, nav_state, buffered_state.bias);
      
      if (state_handle.has_value()) {
        state_timestamps_.push_back(buffered_timestamp_sec);
        num_states_++;
        
        LOG(INFO) << "GraphTimeCentricBackendAdapter: added buffered state " 
                  << state_handle->state_index 
                  << " at timestamp " << buffered_timestamp_sec;
      } else {
        LOG(WARNING) << "GraphTimeCentricBackendAdapter: failed to add buffered state at t=" 
                     << buffered_timestamp_sec;
      }
    }
    
    // STEP 2: Add the keyframe state
    gtsam::NavState keyframe_nav_state(pose, velocity);
    auto keyframe_state_handle = interface_->createStateAtTimestamp(
        keyframe_timestamp_sec, keyframe_nav_state, bias);
    
    if (keyframe_state_handle.has_value()) {
      state_timestamps_.push_back(keyframe_timestamp_sec);
      num_states_++;
      
      LOG(INFO) << "GraphTimeCentricBackendAdapter: created keyframe state " 
                << keyframe_state_handle->state_index 
                << " at timestamp " << keyframe_state_handle->timestamp 
                << " (total states: " << num_states_ << ")";
    } else {
      LOG(WARNING) << "GraphTimeCentricBackendAdapter: failed to create keyframe state at t=" 
                   << keyframe_timestamp_sec;
    }
    
  } catch (const std::exception& e) {
    LOG(ERROR) << "GraphTimeCentricBackendAdapter: failed to add keyframe: " << e.what();
  }
#else
  LOG(WARNING) << "GraphTimeCentricBackendAdapter: ENABLE_GRAPH_TIME_CENTRIC_ADAPTER not defined";
#endif
}

// TODO(KIMERA_ADAPTER_7): Add IMU measurement to be used for preintegration
void GraphTimeCentricBackendAdapter::addIMUMeasurement(
    const Timestamp& timestamp,
    const gtsam::Vector3& linear_acceleration,
    const gtsam::Vector3& angular_velocity) {
#ifdef ENABLE_GRAPH_TIME_CENTRIC_ADAPTER
  if (!initialized_) {
    LOG(ERROR) << "GraphTimeCentricBackendAdapter: not initialized, cannot add IMU measurement";
    return;
  }
  
  const double timestamp_sec = timestampToSeconds(timestamp);
  
  try {
    // Add IMU measurement through the interface
    interface_->addIMUData(timestamp_sec, linear_acceleration, angular_velocity);
    
    // Log occasionally to avoid spam (every 10th measurement)
    static int imu_count = 0;
    if (++imu_count % 10 == 0) {
      LOG(INFO) << "GraphTimeCentricBackendAdapter: added IMU measurement " << imu_count 
                << " at t=" << std::fixed << std::setprecision(6) << timestamp_sec;
    }
    
  } catch (const std::exception& e) {
    LOG(ERROR) << "GraphTimeCentricBackendAdapter: failed to add IMU measurement: " << e.what();
  }
#else
  LOG(WARNING) << "GraphTimeCentricBackendAdapter: ENABLE_GRAPH_TIME_CENTRIC_ADAPTER not defined";
#endif
}

// TODO(KIMERA_ADAPTER_8): Trigger optimization of the factor graph
bool GraphTimeCentricBackendAdapter::optimizeGraph() {
#ifdef ENABLE_GRAPH_TIME_CENTRIC_ADAPTER
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
    auto result = interface_->optimize();
    
    if (result.success) {
      last_optimization_result_ = result;
      last_optimization_time_ = state_timestamps_.back();
      
      LOG(INFO) << "GraphTimeCentricBackendAdapter: optimization succeeded"
                << " - optimized " << result.num_optimized_states << " states"
                << ", error: " << result.final_error
                << ", iterations: " << result.num_iterations;
      return true;
    } else {
      LOG(ERROR) << "GraphTimeCentricBackendAdapter: optimization failed: " << result.error_message;
      return false;
    }
    
  } catch (const std::exception& e) {
    LOG(ERROR) << "GraphTimeCentricBackendAdapter: optimization failed with exception: " << e.what();
    return false;
  }
#else
  LOG(WARNING) << "GraphTimeCentricBackendAdapter: ENABLE_GRAPH_TIME_CENTRIC_ADAPTER not defined";
  return false;
#endif
}

// Legacy optimize() method for compatibility
bool GraphTimeCentricBackendAdapter::optimize(double /*timestep*/) {
  return optimizeGraph();
}

// TODO(KIMERA_ADAPTER_9): Get optimized pose at specific timestamp
std::optional<gtsam::Pose3> GraphTimeCentricBackendAdapter::getOptimizedPoseAtTime(double timestamp) const {
#ifdef ENABLE_GRAPH_TIME_CENTRIC_ADAPTER
  if (!initialized_ || !interface_) {
    LOG(ERROR) << "GraphTimeCentricBackendAdapter: not initialized";
    return std::nullopt;
  }
  
  // Find closest state handle
  auto state_handle = findStateHandleNearTimestamp(timestamp);
  if (!state_handle.has_value()) {
    LOG(WARNING) << "GraphTimeCentricBackendAdapter: no state found near t=" << timestamp;
    return std::nullopt;
  }
  
  // Get pose from interface
  return interface_->getOptimizedPose(state_handle.value());
#else
  LOG(WARNING) << "GraphTimeCentricBackendAdapter: ENABLE_GRAPH_TIME_CENTRIC_ADAPTER not defined";
  return std::nullopt;
#endif
}

// TODO(KIMERA_ADAPTER_10): Get optimized velocity at specific timestamp
std::optional<gtsam::Vector3> GraphTimeCentricBackendAdapter::getOptimizedVelocityAtTime(double timestamp) const {
#ifdef ENABLE_GRAPH_TIME_CENTRIC_ADAPTER
  if (!initialized_ || !interface_) {
    LOG(ERROR) << "GraphTimeCentricBackendAdapter: not initialized";
    return std::nullopt;
  }
  
  auto state_handle = findStateHandleNearTimestamp(timestamp);
  if (!state_handle.has_value()) {
    LOG(WARNING) << "GraphTimeCentricBackendAdapter: no state found near t=" << timestamp;
    return std::nullopt;
  }
  
  return interface_->getOptimizedVelocity(state_handle.value());
#else
  LOG(WARNING) << "GraphTimeCentricBackendAdapter: ENABLE_GRAPH_TIME_CENTRIC_ADAPTER not defined";
  return std::nullopt;
#endif
}

// TODO(KIMERA_ADAPTER_11): Get optimized IMU bias at specific timestamp
std::optional<gtsam::imuBias::ConstantBias> GraphTimeCentricBackendAdapter::getOptimizedBiasAtTime(double timestamp) const {
#ifdef ENABLE_GRAPH_TIME_CENTRIC_ADAPTER
  if (!initialized_ || !interface_) {
    LOG(ERROR) << "GraphTimeCentricBackendAdapter: not initialized";
    return std::nullopt;
  }
  
  auto state_handle = findStateHandleNearTimestamp(timestamp);
  if (!state_handle.has_value()) {
    LOG(WARNING) << "GraphTimeCentricBackendAdapter: no state found near t=" << timestamp;
    return std::nullopt;
  }
  
  return interface_->getOptimizedBias(state_handle.value());
#else
  LOG(WARNING) << "GraphTimeCentricBackendAdapter: ENABLE_GRAPH_TIME_CENTRIC_ADAPTER not defined";
  return std::nullopt;
#endif
}

// TODO(KIMERA_ADAPTER_12): Get state covariance at specific timestamp
std::optional<gtsam::Matrix> GraphTimeCentricBackendAdapter::getStateCovarianceAtTime(double timestamp) const {
#ifdef ENABLE_GRAPH_TIME_CENTRIC_ADAPTER
  if (!initialized_ || !interface_) {
    LOG(ERROR) << "GraphTimeCentricBackendAdapter: not initialized";
    return std::nullopt;
  }
  
  auto state_handle = findStateHandleNearTimestamp(timestamp);
  if (!state_handle.has_value()) {
    LOG(WARNING) << "GraphTimeCentricBackendAdapter: no state found near t=" << timestamp;
    return std::nullopt;
  }
  
  return interface_->getStateCovariance(state_handle.value());
#else
  LOG(WARNING) << "GraphTimeCentricBackendAdapter: ENABLE_GRAPH_TIME_CENTRIC_ADAPTER not defined";
  return std::nullopt;
#endif
}

// Legacy getLastResult() for compatibility
gtsam::Values GraphTimeCentricBackendAdapter::getLastResult() {
#ifdef ENABLE_GRAPH_TIME_CENTRIC_ADAPTER
  gtsam::Values values;
  
  if (!initialized_ || state_timestamps_.empty()) {
    return values;
  }
  
  // Reconstruct Values from optimized states
  try {
    for (const auto& timestamp : state_timestamps_) {
      auto state_handle = findStateHandleNearTimestamp(timestamp);
      if (!state_handle.has_value()) {
        continue;
      }
      
      auto pose = interface_->getOptimizedPose(state_handle.value());
      auto velocity = interface_->getOptimizedVelocity(state_handle.value());
      auto bias = interface_->getOptimizedBias(state_handle.value());
      
      if (pose.has_value() && velocity.has_value()) {
        // Create keys using state index
        gtsam::Key pose_key = gtsam::Symbol('x', state_handle->state_index).key();
        gtsam::Key vel_key = gtsam::Symbol('v', state_handle->state_index).key();
        gtsam::Key bias_key = gtsam::Symbol('b', state_handle->state_index).key();
        
        values.insert(pose_key, pose.value());
        values.insert(vel_key, velocity.value());
        if (bias.has_value()) {
          values.insert(bias_key, bias.value());
        }
      }
    }
  } catch (const std::exception& e) {
    LOG(ERROR) << "GraphTimeCentricBackendAdapter: failed to get last result: " << e.what();
  }
  
  return values;
#else
  return gtsam::Values();
#endif
}

// TODO(KIMERA_ADAPTER_13): Helper - Convert Kimera timestamp to seconds
double GraphTimeCentricBackendAdapter::timestampToSeconds(const Timestamp& timestamp) const {
  // Kimera timestamps are in nanoseconds
  return static_cast<double>(timestamp) / 1e9;
}

// TODO(KIMERA_ADAPTER_14): Helper - Create integration parameters from Kimera params
fgo::integration::IntegrationParameters GraphTimeCentricBackendAdapter::createIntegrationParams() const {
  fgo::integration::IntegrationParameters params;
  
  // Basic settings
  params.use_imu = true;
  params.use_gp_motion_prior = backend_params_.addBetweenStereoFactors_;
  params.optimize_on_add = false;  // We'll trigger optimization manually
  
  // Timestamp matching tolerance (100ms default)
  params.timestamp_tolerance = 0.1;
  
  // IMU parameters
  params.imu_params.gyroscope_noise_density = imu_params_.gyro_noise_;
  params.imu_params.accelerometer_noise_density = imu_params_.acc_noise_;
  params.imu_params.gyroscope_random_walk = imu_params_.gyro_walk_;
  params.imu_params.accelerometer_random_walk = imu_params_.acc_walk_;
  params.imu_params.integration_uncertainty = imu_params_.imu_integration_sigma_;
  
  // Gravity magnitude (if available)
  params.imu_params.gravity_magnitude = 9.81;  // TODO: get from params if available
  
  // GP motion prior settings (if enabled)
  if (params.use_gp_motion_prior) {
    params.gp_params.qc_model = "WhiteNoise";  // or get from backend_params
    params.gp_params.sigma_position = 0.1;     // TODO: configure from params
    params.gp_params.sigma_rotation = 0.05;
  }
  
  // Optimization settings
  params.optimization_params.max_iterations = 100;
  params.optimization_params.lambda_initial = 1e-5;
  params.optimization_params.lambda_factor = 10.0;
  params.optimization_params.relative_error_threshold = 1e-5;
  params.optimization_params.absolute_error_threshold = 1e-5;
  
  LOG(INFO) << "GraphTimeCentricBackendAdapter: created integration params:"
            << " use_imu=" << params.use_imu
            << ", use_gp=" << params.use_gp_motion_prior
            << ", tolerance=" << params.timestamp_tolerance;
  
  return params;
}

// TODO(KIMERA_ADAPTER_15): Helper - Find state handle near given timestamp
std::optional<fgo::integration::StateHandle> 
GraphTimeCentricBackendAdapter::findStateHandleNearTimestamp(double timestamp) const {
#ifdef ENABLE_GRAPH_TIME_CENTRIC_ADAPTER
  if (state_timestamps_.empty()) {
    return std::nullopt;
  }
  
  // Find closest timestamp
  auto it = std::min_element(state_timestamps_.begin(), state_timestamps_.end(),
                             [timestamp](double a, double b) {
                               return std::abs(a - timestamp) < std::abs(b - timestamp);
                             });
  
  if (it == state_timestamps_.end()) {
    return std::nullopt;
  }
  
  const double closest_time = *it;
  const double tolerance = 0.1;  // 100ms tolerance
  
  if (std::abs(closest_time - timestamp) > tolerance) {
    LOG(WARNING) << "GraphTimeCentricBackendAdapter: closest timestamp " << closest_time 
                 << " is too far from requested " << timestamp 
                 << " (diff: " << std::abs(closest_time - timestamp) << "s)";
    return std::nullopt;
  }
  
  // Create state handle with estimated index
  const size_t state_index = std::distance(state_timestamps_.begin(), it);
  
  fgo::integration::StateHandle handle;
  handle.timestamp = closest_time;
  handle.state_index = state_index;
  
  return handle;
#else
  return std::nullopt;
#endif
}

}  // namespace VIO
