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
#include <functional>
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
    const ImuParams& imu_params,
    SmootherUpdateCallback smoother_update_cb)
    : backend_params_(backend_params)
    , imu_params_(imu_params)
    , smoother_update_cb_(std::move(smoother_update_cb))
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
    // Map internal enum to the string values expected by online_fgo_core::GraphBase
    // GraphBase expects either "IncrementalFixedLag" (iSAM2-based) or
    // "BatchFixedLag" (Levenberg-Marquardt batch fixed-lag).
    switch (backend_params_.smootherType_) {
      case 0:
        // legacy "ISAM2" -> use the name online_fgo_core expects for iSAM2 fixed-lag
        smootherTypeString = "IncrementalFixedLag";
        break;
      case 1:
        // legacy "Batch" -> use the batch fixed-lag name
        smootherTypeString = "BatchFixedLag";
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
    state_timestamps_.clear();
    keyframe_state_handles_.clear();
    num_states_ = 0;

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

fgo::integration::StateHandle GraphTimeCentricBackendAdapter::addKeyframeState(
    const Timestamp& timestamp,
    FrameId frame_id,
    const gtsam::Pose3& pose,
    const gtsam::Vector3& velocity,
    const gtsam::imuBias::ConstantBias& bias) {
  fgo::integration::StateHandle handle;

  if (!initialized_) {
    LOG(ERROR) << "GraphTimeCentricBackendAdapter: not initialized, cannot add keyframe";
    return handle;
  }

  const double keyframe_timestamp_sec = timestampToSeconds(timestamp);

  LOG(INFO) << "GraphTimeCentricBackendAdapter: adding keyframe " << frame_id
            << " at t=" << std::fixed << std::setprecision(6)
            << keyframe_timestamp_sec;

  try {
    handle = integration_interface_->addKeyframeState(
        keyframe_timestamp_sec, pose, velocity, bias);

    if (handle.valid) {
      state_timestamps_.push_back(keyframe_timestamp_sec);
      num_states_++;
      keyframe_state_handles_[frame_id] = handle;

      LOG(INFO) << "GraphTimeCentricBackendAdapter: created keyframe state "
                << handle.index << " at timestamp " << handle.timestamp
                << " (frame_id=" << frame_id << ", total states: " << num_states_
                << ")";
    } else {
      LOG(WARNING)
          << "GraphTimeCentricBackendAdapter: failed to create keyframe state at t="
          << keyframe_timestamp_sec;
    }
  } catch (const std::exception& e) {
    LOG(ERROR) << "GraphTimeCentricBackendAdapter: failed to add keyframe: "
               << e.what();
  }

  return handle;
}

bool GraphTimeCentricBackendAdapter::addImuFactorBetween(
    const FrameId& previous_frame_id,
    const FrameId& current_frame_id,
    const ImuFrontend::PimPtr& pim) {
  if (!initialized_) {
    LOG(ERROR) << "GraphTimeCentricBackendAdapter: not initialized, cannot add IMU factor";
    return false;
  }

  if (!pim) {
    LOG(WARNING) << "GraphTimeCentricBackendAdapter: null PIM provided for IMU factor";
    return false;
  }

  auto prev_it = keyframe_state_handles_.find(previous_frame_id);
  auto curr_it = keyframe_state_handles_.find(current_frame_id);

  if (prev_it == keyframe_state_handles_.end() ||
      curr_it == keyframe_state_handles_.end()) {
    LOG(WARNING) << "GraphTimeCentricBackendAdapter: cannot add IMU factor between frame "
                 << previous_frame_id << " and " << current_frame_id
                 << " (state handle missing)";
    return false;
  }

  bool success = integration_interface_->addImuFactorBetween(
      prev_it->second,
      curr_it->second,
      pim);

  if (!success) {
    LOG(WARNING) << "GraphTimeCentricBackendAdapter: failed to add IMU factor between frame "
                 << previous_frame_id << " and " << current_frame_id;
    return false;
  }

  return true;
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
  
  if (!smoother_update_cb_) {
    LOG(ERROR) << "GraphTimeCentricBackendAdapter: smoother callback not set";
    return false;
  }

  fgo::integration::KimeraIntegrationInterface::IncrementalUpdatePacket packet;
  
  // CRITICAL DEBUG: Log before calling buildIncrementalUpdate
  
  LOG(INFO) << "GraphTimeCentricBackendAdapter: calling buildIncrementalUpdate...";
  
  bool build_result = integration_interface_->buildIncrementalUpdate(&packet);
  
  if (!build_result) {
    LOG(WARNING) << "GraphTimeCentricBackendAdapter: no incremental update available";
    return false;
  }

  // Debug contents of incremental update before handing to smoother.
  LOG(INFO) << "GraphTimeCentricBackendAdapter: incremental update packet - "
            << "factors=" << packet.factors.size()
            << ", values=" << packet.values.size()
            << ", key_timestamps=" << packet.key_timestamps.size();

  if (packet.factors.empty()) {
    LOG(WARNING) << "GraphTimeCentricBackendAdapter: packet.factors is empty before smoother call";
  }
  if (packet.values.empty()) {
    LOG(WARNING) << "GraphTimeCentricBackendAdapter: packet.values is empty before smoother call";
  }
  if (packet.key_timestamps.empty()) {
    LOG(WARNING) << "GraphTimeCentricBackendAdapter: packet.key_timestamps is empty before smoother call";
  }

  std::map<gtsam::Key, double> timestamps(packet.key_timestamps.begin(),
                                          packet.key_timestamps.end());

  Smoother::Result result;
  gtsam::FactorIndices delete_slots;

  bool status = smoother_update_cb_(
      &result, packet.factors, packet.values, timestamps, delete_slots);

  if (!status) {
    LOG(ERROR) << "GraphTimeCentricBackendAdapter: smoother update failed";
    return false;
  }

  integration_interface_->markIncrementalUpdateConsumed();

  if (!state_timestamps_.empty()) {
    last_optimization_time_ = state_timestamps_.back();
  }

  LOG(INFO) << "GraphTimeCentricBackendAdapter: optimization via backend smoother succeeded";
  return true;
}

bool GraphTimeCentricBackendAdapter::optimize(double /*timestep*/) {
  return optimizeGraph();
}

double GraphTimeCentricBackendAdapter::getLastOptimizationTime() const {
  return last_optimization_time_;
}


bool GraphTimeCentricBackendAdapter::addStateValues(unsigned long frame_id, double timestamp, const gtsam::NavState& navstate) {
  Timestamp kimera_timestamp = static_cast<Timestamp>(timestamp * 1e9);
  gtsam::imuBias::ConstantBias bias;
  ImuFrontend::PimPtr null_pim = nullptr;
  addKeyframeState(kimera_timestamp, FrameId(frame_id), navstate.pose(), navstate.velocity(), bias);
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

} // namespace VIO
