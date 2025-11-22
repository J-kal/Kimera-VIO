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
   * @return true if initialization successful
   */
  bool initialize();

  /**
   * @brief Check if adapter is initialized
   * @return true if ready to use
   */
  bool isInitialized() const;

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
   * @brief Add single IMU measurement to be used for preintegration
   * @param timestamp Timestamp of the measurement
   * @param linear_acceleration Raw accelerometer measurement
   * @param angular_velocity Raw gyroscope measurement
   */
  bool addIMUMeasurement(const ImuAccGyr& imu_measurement);

  /**
   * @brief Add multiple IMU measurements in batch
   * @param imu_measurements Vector of IMU measurements
   * @return Number of measurements successfully added
   */
  size_t addIMUMeasurements(const std::vector<ImuAccGyr>& imu_measurements);

  /**
   * @brief Add multiple IMU measurements in batch from separate timestamp and
   * measurement matrices.
   * @param timestamps Matrix of timestamps
   * @param measurements Matrix of accelerometer and gyroscope measurements
   * @return true if successfully added
   */
  bool addIMUMeasurements(const ImuStampS& timestamps,
                          const ImuAccGyrS& measurements);

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
   * NOTE: Preintegration is handled automatically by GraphTimeCentric.
   * This method is currently a no-op.
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
   * NOTE: Optimization failure callbacks not yet implemented
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
  double getLastOptimizationTime() const;

  // ========================================================================
  // RESULT RETRIEVAL
  // ========================================================================

  /**
   * @brief Get optimized NavState at specific timestamp
   * @param timestamp Timestamp to query
   * @return NavState if available at that time
   */
  std::optional<gtsam::NavState> getStateAtTime(Timestamp timestamp);

  /**
   * @brief Get optimized Pose3 at specific timestamp (seconds)
   * @param timestamp Timestamp in seconds
   * @return Pose3 if available
   */
  std::optional<gtsam::Pose3> getOptimizedPoseAtTime(double timestamp) const;

  /**
   * @brief Get optimized velocity at specific timestamp (seconds)
   * @param timestamp Timestamp in seconds
   * @return Velocity vector if available
   */
  std::optional<gtsam::Vector3> getOptimizedVelocityAtTime(double timestamp) const;

  /**
   * @brief Get optimized IMU bias at specific timestamp (seconds)
   * @param timestamp Timestamp in seconds
   * @return IMU bias if available
   */
  std::optional<gtsam::imuBias::ConstantBias> getOptimizedBiasAtTime(double timestamp) const;

  /**
   * @brief Get state covariance at specific timestamp (seconds)
   * @param timestamp Timestamp in seconds
   * @return Covariance matrix if available
   */
  std::optional<gtsam::Matrix> getStateCovarianceAtTime(double timestamp) const;

  /**
   * @brief Get latest optimized NavState
   * @return Most recent NavState after optimization
   */
  std::optional<gtsam::NavState> getLatestState();

  /**
   * @brief Get latest optimized IMU bias
   * @return Most recent bias estimate
   */
  std::optional<gtsam::imuBias::ConstantBias> getLatestIMUBias();

  /**
   * @brief Get state covariance at timestamp
   * @param timestamp Timestamp to query
   * @return Covariance matrix if available
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
  size_t getNumStates() const;

  /**
   * @brief Get number of IMU measurements buffered
   * @return IMU buffer size
   */
  size_t getNumBufferedIMU() const;

  /**
   * @brief Get number of non-keyframe states buffered
   * @return Non-keyframe buffer size
   */
  size_t getNumBufferedStates() const;

  /**
   * @brief Get statistics string for logging
   * @return Formatted statistics string
   */
  std::string getStatistics() const;
  
  // ========================================================================
  // HELPER METHODS (public for timestamp conversion)
  // ========================================================================
  
  /**
   * @brief Convert Kimera Timestamp to double seconds
   * @param timestamp Kimera timestamp (nanoseconds)
   * @return Time in seconds
   */
  double timestampToSeconds(const Timestamp& timestamp) const;

  /**
   * @brief Convert double seconds to Kimera Timestamp
   * @param seconds Time in seconds
   * @return Kimera timestamp (nanoseconds)
   */
  Timestamp secondsToTimestamp(double seconds) const;

private:
  // Parameters
  BackendParams backend_params_;
  ImuParams imu_params_;
  
  // Integration interface to online_fgo_core (forward declared)
  std::unique_ptr<fgo::integration::KimeraIntegrationInterface> integration_interface_;
  std::unique_ptr<fgo::core::ApplicationInterface> standalone_app_;
  
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
  fgo::integration::KimeraIntegrationParams createIntegrationParams() const;
  std::optional<fgo::integration::StateHandle> findStateHandleNearTimestamp(double timestamp) const;
};

} // namespace VIO

#endif // KIMERA_VIO_GRAPHTIMECENTRIC_BACKEND_ADAPTER_H
