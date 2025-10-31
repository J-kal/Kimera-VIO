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
//  GraphTimeCentricBackendAdapter: Adapter to use GraphTimeCentric as VioBackend
//

#ifndef KIMERA_VIO_GRAPHTIMECENTRIC_BACKEND_ADAPTER_H
#define KIMERA_VIO_GRAPHTIMECENTRIC_BACKEND_ADAPTER_H

#pragma once

#include <memory>
#include <vector>
#include <deque>
#include <map>
#include <mutex>
#include <optional>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/navigation/NavState.h>
#include <gtsam/navigation/ImuBias.h>
#include <gtsam/nonlinear/Values.h>

#include "kimera-vio/backend/VioBackend-definitions.h"
#include "kimera-vio/backend/VioBackendParams.h"
#include "kimera-vio/imu-frontend/ImuFrontend-definitions.h"

// Forward declare to avoid hard dependency when not compiled in
namespace fgo {
  namespace integration {
    class KimeraIntegrationInterface;
    struct KimeraIntegrationParams;
    struct StateHandle;
    struct OptimizationResult;
  }
  namespace core {
    class ApplicationInterface;
  }
}

namespace VIO {

/**
 * @brief Adapter that allows Kimera VioBackend to use online_fgo_core's GraphTimeCentric
 * 
 * This adapter translates between Kimera's VioBackend interface and the GraphTimeCentric
 * graph from online_fgo_core. It handles:
 * - IMU measurement buffering and forwarding
 * - Timestamp conversion (Kimera Timestamp -> double seconds)
 * - State creation at keyframe timestamps
 * - Optimization triggering and result retrieval
 * - (Future) Visual factor translation
 * 
 * Design:
 * - Implements same interface as VioBackend operations
 * - Uses KimeraIntegrationInterface to communicate with GraphTimeCentric
 * - Maintains internal buffers for IMU and state timestamps
 * - Thread-safe for concurrent VioBackend calls
 * 
 * Usage in VioBackend:
 *   if (backend_params.use_graph_time_centric) {
 *     adapter_ = std::make_unique<GraphTimeCentricBackendAdapter>(...);
 *     adapter_->initialize(...);
 *   }
 */
class GraphTimeCentricBackendAdapter {
public:
  /**
   * @brief Constructor
   * @param backend_params Kimera backend parameters
   * @param imu_params IMU parameters
   */
  GraphTimeCentricBackendAdapter(const BackendParams& backend_params,
                                  const ImuParams& imu_params);

  /**
   * @brief Destructor
   */
  ~GraphTimeCentricBackendAdapter();

  // ========================================================================
  // INITIALIZATION
  // ========================================================================

  /**
   * @brief Initialize the adapter and underlying GraphTimeCentric
   * @param app_interface Application interface from online_fgo_core
   * @return true if initialization successful
   * 
   * TODO: Create KimeraIntegrationInterface with app_interface
   * TODO: Convert backend_params and imu_params to KimeraIntegrationParams
   * TODO: Call integration_interface_->initialize()
   */
  bool initialize(fgo::core::ApplicationInterface* app_interface);

  /**
   * @brief Initialize with default standalone application
   * @return true if initialization successful
   * 
   * TODO: Create standalone ApplicationInterface for testing
   */
  bool initializeStandalone();

  /**
   * @brief Check if adapter is initialized
   * @return true if ready to use
   */
  bool isInitialized() const { return initialized_; }

  // ========================================================================
  // STATE MANAGEMENT
  // ========================================================================

  /**
   * @brief Buffer a non-keyframe state for later addition
   * 
   * Called by VioBackend when a frame is NOT selected as keyframe but should
   * still be added to the graph. These frames are buffered and added in batch
   * when the next keyframe is seen.
   * 
   * @param timestamp Frame timestamp
   * @param pose Pose estimate
   * @param velocity Velocity estimate
   * @param bias IMU bias estimate
   * @return true if successfully buffered
   */
  bool bufferNonKeyframeState(
      const Timestamp& timestamp,
      const gtsam::Pose3& pose,
      const gtsam::Vector3& velocity,
      const gtsam::imuBias::ConstantBias& bias);

  /**
   * @brief Add a keyframe state at specified timestamp
   * 
   * Called by VioBackend when a new keyframe is created. This creates
   * a state in GraphTimeCentric at the keyframe timestamp.
   * ALSO processes any buffered non-keyframe states in chronological order.
   * 
   * @param timestamp Keyframe timestamp (Kimera Timestamp type)
   * @param pose Pose estimate for the keyframe
   * @param velocity Velocity estimate
   * @param bias IMU bias estimate
   * @return true if state successfully created
   * 
   * Implementation:
   * 1. Process all buffered non-keyframes (sorted by timestamp)
   * 2. Add the keyframe state
   * 3. Clear the buffer
   */
  void addKeyframeState(
      const Timestamp& timestamp,
      const gtsam::Pose3& pose,
      const gtsam::Vector3& velocity,
      const gtsam::imuBias::ConstantBias& bias);

  /**
   * @brief Legacy interface - add a keyframe with Pose only
   * @param timestamp Keyframe timestamp
   * @param pose_estimate Initial pose estimate for the keyframe
   * @return true if state successfully created
   */
  bool addKeyframeState(Timestamp timestamp, const gtsam::Pose3& pose_estimate);

  /**
   * @brief Legacy interface - add a keyframe with full NavState estimate
   * @param timestamp Keyframe timestamp
   * @param nav_state Initial NavState (pose + velocity)
   * @return true if state successfully created
   */
  bool addKeyframeState(Timestamp timestamp, const gtsam::NavState& nav_state);

  /**
   * @brief Add state from frame_id and timestamp (legacy interface)
   * @param frame_id Frame identifier
   * @param timestamp Timestamp in seconds
   * @param navstate Initial NavState
   * @return true if successfully added
   */
  bool addStateValues(unsigned long frame_id, double timestamp, const gtsam::NavState& navstate);

  // ========================================================================
  // IMU HANDLING
  // ========================================================================

  /**
   * @brief Add a single IMU measurement
   * 
   * Called by VioBackend for each IMU measurement. Measurements are buffered
   * and forwarded to GraphTimeCentric.
   * 
   * @param imu_measurement IMU acc/gyro measurement
   * @return true if successfully buffered
   * 
   * TODO: Extract timestamp, accel, gyro from ImuAccGyr struct
   * TODO: Compute dt from previous measurement
   * TODO: Call integration_interface_->addIMUData()
   * TODO: Buffer measurement for preintegration between states
   */
  bool addIMUMeasurement(const ImuAccGyr& imu_measurement);

  /**
   * @brief Add multiple IMU measurements in batch
   * @param imu_measurements Vector of IMU measurements
   * @return Number of measurements successfully added
   * 
   * TODO: Batch process and call integration_interface_->addIMUDataBatch()
   */
  size_t addIMUMeasurements(const std::vector<ImuAccGyr>& imu_measurements);

  /**
   * @brief Add IMU timestamps (legacy interface)
   * @param imu_timestamps Vector of timestamps in seconds
   * @return true if successfully added
   */
  bool addIMUTimestamps(const std::vector<double>& imu_timestamps);

  /**
   * @brief Preintegrate IMU measurements between two timestamps
   * 
   * This explicitly requests preintegration between two states.
   * Used when states are not consecutive in time.
   * 
   * @param t_i Start timestamp
   * @param t_j End timestamp
   * @return true if preintegration successful
   * 
   * TODO: Extract IMU measurements from buffer in time range [t_i, t_j]
   * TODO: Call GraphTimeCentric IMU factor creation methods
   */
  bool preintegrateIMUBetweenStates(Timestamp t_i, Timestamp t_j);

  // ========================================================================
  // OPTIMIZATION
  // ========================================================================

  /**
   * @brief Trigger graph optimization
   * 
   * Called by VioBackend when optimization should be performed (e.g., on keyframe).
   * 
   * @return true if optimization successful
   * 
   * TODO: Call integration_interface_->optimize()
   * TODO: Check OptimizationResult for success
   * TODO: Update internal state tracking
   * TODO: Trigger callbacks if optimization failed
   */
  bool optimizeGraph();

  /**
   * @brief Optimize with timestamp (legacy interface)
   * @param timestep Current timestamp
   * @return true if optimization successful
   */
  bool optimize(double timestep);

  /**
   * @brief Get the optimization time from last optimize call
   * @return Optimization time in seconds
   */
  double getLastOptimizationTime() const { return last_optimization_time_; }

  // ========================================================================
  // RESULT RETRIEVAL
  // ========================================================================

  /**
   * @brief Get optimized NavState at specific timestamp
   * @param timestamp Timestamp to query
   * @return NavState if available at that time
   * 
   * TODO: Find state handle for timestamp (within tolerance)
   * TODO: Call integration_interface_->getOptimizedState()
   */
  std::optional<gtsam::NavState> getStateAtTime(Timestamp timestamp);

  /**
   * @brief Get latest optimized NavState
   * @return Most recent NavState after optimization
   * 
   * TODO: Call integration_interface_->getLatestOptimizedState()
   */
  std::optional<gtsam::NavState> getLatestState();

  /**
   * @brief Get latest optimized IMU bias
   * @return Most recent bias estimate
   * 
   * TODO: Call integration_interface_->getLatestOptimizedBias()
   */
  std::optional<gtsam::imuBias::ConstantBias> getLatestIMUBias();

  /**
   * @brief Get state covariance at timestamp
   * @param timestamp Timestamp to query
   * @return Covariance matrix if available
   * 
   * TODO: Find state handle and call integration_interface_->getStateCovariance()
   */
  std::optional<gtsam::Matrix> getStateCovariance(Timestamp timestamp);

  /**
   * @brief Get latest state covariance
   * @return Covariance for most recent state
   */
  std::optional<gtsam::Matrix> getLatestStateCovariance();

  /**
   * @brief Get last optimization result as gtsam::Values (legacy interface)
   * @return GTSAM values with all optimized variables
   */
  gtsam::Values getLastResult();

  // ========================================================================
  // STATISTICS AND DIAGNOSTICS
  // ========================================================================

  /**
   * @brief Get number of states in graph
   * @return State count
   */
  size_t getNumStates() const { return num_states_; }

  /**
   * @brief Get number of IMU measurements buffered
   * @return IMU buffer size
   */
  size_t getNumBufferedIMU() const {
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    return imu_buffer_.size();
  }

  /**
   * @brief Get number of non-keyframe states buffered
   * @return Non-keyframe buffer size
   */
  size_t getNumBufferedStates() const {
    std::lock_guard<std::mutex> lock(state_buffer_mutex_);
    return non_keyframe_buffer_.size();
  }

  /**
   * @brief Get statistics string for logging
   * @return Formatted statistics string
   */
  std::string getStatistics() const;

private:
  // ========================================================================
  // MEMBER VARIABLES
  // ========================================================================

  // Parameters
  BackendParams backend_params_;
  ImuParams imu_params_;

  // Integration interface to online_fgo_core
  std::unique_ptr<fgo::integration::KimeraIntegrationInterface> integration_interface_;

  // Application interface (owned by caller, not this class)
  fgo::core::ApplicationInterface* app_interface_ = nullptr;
  std::unique_ptr<fgo::core::ApplicationInterface> standalone_app_;  // For standalone mode

  // Initialization flag
  bool initialized_ = false;

  // IMU measurement buffer
  std::deque<ImuAccGyr> imu_buffer_;
  mutable std::mutex buffer_mutex_;

  // Non-keyframe buffer structure
  struct BufferedState {
    Timestamp timestamp;
    gtsam::Pose3 pose;
    gtsam::Vector3 velocity;
    gtsam::imuBias::ConstantBias bias;
    
    // For sorting by timestamp
    bool operator<(const BufferedState& other) const {
      return timestamp < other.timestamp;
    }
  };
  
  // Buffer for non-keyframe states (to be added when next keyframe arrives)
  std::vector<BufferedState> non_keyframe_buffer_;
  mutable std::mutex state_buffer_mutex_;

  // State tracking
  std::map<Timestamp, fgo::integration::StateHandle> keyframe_states_;
  std::vector<double> state_timestamps_;  // All timestamps (keyframes + non-keyframes) in order
  size_t num_states_ = 0;

  // Timing
  double last_optimization_time_ = 0.0;
  Timestamp last_imu_timestamp_{0};

  // Last optimization result (for getLastResult)
  gtsam::Values last_result_;

  // ========================================================================
  // HELPER METHODS
  // ========================================================================

  /**
   * @brief Convert Kimera Timestamp to double seconds
   * @param timestamp Kimera timestamp
   * @return Time in seconds
   * 
   * TODO: Implement based on Kimera's Timestamp type
   * TODO: Handle nanoseconds or microseconds conversion
   */
  double timestampToSeconds(Timestamp timestamp) const;

  /**
   * @brief Convert double seconds to Kimera Timestamp
   * @param seconds Time in seconds
   * @return Kimera timestamp
   * 
   * TODO: Reverse conversion of timestampToSeconds
   */
  Timestamp secondsToTimestamp(double seconds) const;

  /**
   * @brief Create KimeraIntegrationParams from Kimera parameters
   * @return Integration parameters
   * 
   * TODO: Map BackendParams and ImuParams to KimeraIntegrationParams
   * TODO: Set smoother_lag, GP prior settings, etc.
   */
  fgo::integration::KimeraIntegrationParams createIntegrationParams() const;

  /**
   * @brief Find state handle closest to timestamp
   * @param timestamp Target timestamp
   * @param tolerance Max time difference in seconds
   * @return State handle if found
   * 
   * TODO: Search keyframe_states_ for nearest timestamp
   */
  std::optional<fgo::integration::StateHandle> findStateHandleNearTimestamp(
      Timestamp timestamp, double tolerance = 0.001) const;
};

} // namespace VIO

#endif // KIMERA_VIO_GRAPHTIMECENTRIC_BACKEND_ADAPTER_H
