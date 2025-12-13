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
#include <map>
#include <unordered_map>
#include <mutex>
#include <functional>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/StereoPoint2.h>
#include <gtsam/geometry/Cal3_S2Stereo.h>
#include <gtsam/navigation/NavState.h>
#include <gtsam/navigation/ImuBias.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam_unstable/slam/SmartStereoProjectionPoseFactor.h>

#include "kimera-vio/backend/VioBackend-definitions.h"
#include "kimera-vio/backend/VioBackendParams.h"
#include "kimera-vio/imu-frontend/ImuFrontend-definitions.h"
#include "online_fgo_core/integration/KimeraIntegrationInterface.h"

namespace VIO {

// Import OmegaAtState from online_fgo_core for use in this adapter
using fgo::integration::OmegaAtState;

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
  using SmootherUpdateCallback = std::function<bool(
      Smoother::Result*,
      const gtsam::NonlinearFactorGraph&,
      const gtsam::Values&,
      const std::map<gtsam::Key, double>&,
      const gtsam::FactorIndices&)>;

  /**
   * @brief Constructor
   * @param backend_params Kimera backend parameters
   * @param imu_params IMU parameters
   */
  GraphTimeCentricBackendAdapter(const BackendParams& backend_params,
                                 const ImuParams& imu_params,
                                 SmootherUpdateCallback smoother_update_cb = SmootherUpdateCallback());

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
   * @brief Set pointer to VioBackend's smoother (for ISAM2 result access)
   * @param smoother Pointer to VioBackend's smoother
   */
  void setSmoother(Smoother* smoother);

  /**
   * @brief Set pointer to VioBackend instance (for state extraction via updateStates)
   * @param vio_backend Pointer to VioBackend instance
   */
  void setVioBackend(class VioBackend* vio_backend);

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
   * @brief Add a keyframe state at specified timestamp with preintegrated IMU
   * 
   * Called by VioBackend when a new keyframe is created. This creates
   * a state in GraphTimeCentric at the keyframe timestamp and stores
   * the preintegrated IMU measurements (pim_) for later factor creation.
   * 
   * @param timestamp Keyframe timestamp (Kimera Timestamp type)
   * @param pose Pose estimate for the keyframe
   * @param velocity Velocity estimate
   * @param bias IMU bias estimate
   * @param pim Preintegrated IMU measurements from last keyframe to this one
   * 
   * Implementation:
   * 1. Create state at keyframe timestamp
   * 2. Store PIM for later IMU factor creation
   */
  fgo::integration::StateHandle addKeyframeState(
      const Timestamp& timestamp,
      FrameId frame_id,
      const gtsam::Pose3& pose,
      const gtsam::Vector3& velocity,
      const gtsam::imuBias::ConstantBias& bias);

  fgo::integration::StateHandle bootstrapInitialState(
      const Timestamp& timestamp,
      FrameId frame_id,
      const gtsam::Pose3& pose,
      const gtsam::Vector3& velocity,
      const gtsam::imuBias::ConstantBias& bias);

  bool addImuFactorBetween(const FrameId& previous_frame_id,
                           const FrameId& current_frame_id,
                           const ImuFrontend::PimPtr& pim);

  // ========================================================================
  // VISUAL FACTOR INTERFACE
  // ========================================================================

  /**
   * @brief Set stereo camera calibration and smart factor params for visual factors
   * @param stereo_cal Stereo camera calibration
   * @param B_Pose_leftCam Body to left camera transformation
   * @param smart_noise Pre-initialized smart factor noise model (from VioBackend)
   * @param smart_params Pre-initialized smart factor params (from VioBackend)
   */
  void setStereoCalibration(const gtsam::Cal3_S2Stereo::shared_ptr& stereo_cal,
                            const gtsam::Pose3& B_Pose_leftCam,
                            const gtsam::SharedNoiseModel& smart_noise,
                            const gtsam::SmartStereoProjectionParams& smart_params);
  
  /**
   * @brief Set GP motion prior parameters
   * @param gp_qc_model Pre-initialized Qc noise model for GP priors (from VioBackend)
   * @param gp_ad_matrix Singer model acceleration damping matrix
   * @param gp_acc_prior_noise Prior noise for acceleration state (Full variants)
   * 
   * This follows the same pattern as setStereoCalibration - VioBackend creates
   * all parameters and passes them pre-initialized to ensure consistency.
   */
  void setGPPriorParams(const gtsam::SharedNoiseModel& gp_qc_model,
                        const gtsam::Matrix6& gp_ad_matrix,
                        const gtsam::SharedNoiseModel& gp_acc_prior_noise);
  
  /**
   * @brief Add IMU factor with omega (angular velocity) for full GP priors
   * @param from_id Previous keyframe ID
   * @param to_id Current keyframe ID  
   * @param pim Preintegrated IMU measurements
   * @param omega_from OmegaAtState at the from_id state (contains bias-corrected omega)
   * @param omega_to OmegaAtState at the to_id state (contains bias-corrected omega)
   * @return True if successful
   * 
   * OmegaAtState encapsulates bias-corrected angular velocity computed from
   * gyroscope measurements: omega = gyro_meas - gyro_bias
   */
  bool addImuFactorWithOmega(FrameId from_id, 
                             FrameId to_id,
                             const gtsam::PreintegrationType& pim,
                             const OmegaAtState& omega_from,
                             const OmegaAtState& omega_to);

  /**
   * @brief Add stereo visual measurements for a keyframe
   * @param frame_id Current keyframe ID
   * @param stereo_measurements Vector of (landmark_id, stereo_point) pairs
   * @return Number of landmarks processed
   */
  size_t addStereoMeasurements(FrameId frame_id,
                               const StereoMeasurements& stereo_measurements);


  /**
   * @brief Add state from frame_id and timestamp (legacy interface)
   * @param frame_id Frame identifier
   * @param timestamp Timestamp in seconds
   * @param navstate Initial NavState
   * @return true if successfully added
   */
  bool addStateValues(unsigned long frame_id, double timestamp, const gtsam::NavState& navstate);


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
  SmootherUpdateCallback smoother_update_cb_;
  
  // Pointer to VioBackend's smoother for accessing ISAM2 result (slot tracking)
  Smoother* vio_backend_smoother_;
  
  // Pointer to VioBackend instance for calling updateStates() (state extraction)
  class VioBackend* vio_backend_;
  
  // State
  bool initialized_;
  size_t num_states_;
  double last_optimization_time_;
  double last_imu_timestamp_sec_;
  std::vector<double> state_timestamps_;
  std::unordered_map<FrameId, fgo::integration::StateHandle> keyframe_state_handles_;
  int optimization_iteration_;  // Counter for factor graph debug logging
  
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
};

} // namespace VIO

#endif // KIMERA_VIO_GRAPHTIMECENTRIC_BACKEND_ADAPTER_H
