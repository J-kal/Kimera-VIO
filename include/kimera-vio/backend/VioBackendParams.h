/* ----------------------------------------------------------------------------
 * Copyright 2017, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Luca Carlone, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file   VioBackendParams.h
 * @brief  Class parsing the parameters for the VIO's Backend from a YAML file.
 * @author Antoni Rosinol, Luca Carlone
 */

#pragma once

#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <gtsam/base/Vector.h>
#include <gtsam/nonlinear/ISAM2Params.h>
#include <gtsam/slam/SmartFactorParams.h>

#include <glog/logging.h>

#include "kimera-vio/common/VioNavState.h"
#include "kimera-vio/pipeline/PipelineParams.h"
#include "kimera-vio/utils/Macros.h"
#include "kimera-vio/utils/YamlParser.h"

namespace VIO {

/** \struct Backend Output Params
 * \brief Params controlling what the Backend outputs.
 */
struct BackendOutputParams {
 public:
  BackendOutputParams(
      const bool& output_map_lmk_ids_to_3d_points_in_time_horizon,
      const int& min_num_obs_for_lmks_in_time_horizon,
      const bool& output_lmk_id_to_lmk_type_map)
      : output_map_lmk_ids_to_3d_points_in_time_horizon_(
            output_map_lmk_ids_to_3d_points_in_time_horizon),
        min_num_obs_for_lmks_in_time_horizon_(
            min_num_obs_for_lmks_in_time_horizon),
        output_lmk_id_to_lmk_type_map_(output_lmk_id_to_lmk_type_map) {}
  ~BackendOutputParams() = default;

 public:
  //! Whether to output the map from lmk ids to actual lmk 3D positions for
  //! those landmarks that are in the time-horizon of the Backend optimization.
  bool output_map_lmk_ids_to_3d_points_in_time_horizon_ = false;
  //! Minimum number of observations for a landmark to be included in the
  //! output of the map from landmark ids to actual landmark 3D positions.
  int min_num_obs_for_lmks_in_time_horizon_ = 4u;
  //! Whether to output as well the type of lmk id (smart, projection, etc).
  //! This is typically used for visualization, to display lmks with different
  //! colors depending on their type.
  bool output_lmk_id_to_lmk_type_map_ = false;
};

/**
 * @brief The PoseGuessSource enum determines which pose is used as initial
 * guess for the keyframe pose.
 */
enum class PoseGuessSource {
  IMU = 0,
  MONO = 1,
  STEREO = 2,
  PNP = 3,
  EXTERNAL_ODOM = 4,
};

class BackendParams : public PipelineParams {
 public:
  KIMERA_POINTER_TYPEDEFS(BackendParams);
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  BackendParams();
  virtual ~BackendParams() = default;

 public:
  virtual bool equals(const BackendParams& vp2, double tol = 1e-8) const;
  void print() const override;
  bool parseYAML(const std::string& filepath) override;

  // Set parameters for ISAM 2 incremental smoother.
  static void setIsam2Params(const BackendParams& vio_params,
                             gtsam::ISAM2Params* isam_param);

 protected:
  bool equals(const PipelineParams& obj) const override {
    const auto& rhs = static_cast<const BackendParams&>(obj);
    return equals(rhs, 1e-8);
  }
  bool parseYAMLVioBackendParams(const YamlParser& yaml_parser);
  bool equalsVioBackendParams(const BackendParams& vp2,
                              double tol = 1e-8) const;
  void printVioBackendParams() const;

 public:
  //! Initialization params
  // TODO(Toni): make an enum class...
  int autoInitialize_ = 0;
  double initialPositionSigma_ = 0.00001;
  double initialRollPitchSigma_ = 10.0 / 180.0 * M_PI;
  double initialYawSigma_ = 0.1 / 180.0 * M_PI;
  double initialVelocitySigma_ = 1e-3;
  double initialAccBiasSigma_ = 0.1;
  double initialGyroBiasSigma_ = 0.01;
  /// Only used if autoInitialize set to false.
  VioNavState initial_ground_truth_state_ = VioNavState();
  bool roundOnAutoInitialize_ = false;

  //! Smart factor params
  gtsam::LinearizationMode linearizationMode_ = gtsam::HESSIAN;
  gtsam::DegeneracyMode degeneracyMode_ = gtsam::ZERO_ON_DEGENERACY;
  double smartNoiseSigma_ = 3.0;
  double rankTolerance_ = 1.0;
  //! max distance to triangulate point in meters
  double landmarkDistanceThreshold_ = 20.0;
  //! max acceptable reprojection error // before tuning: 3
  double outlierRejection_ = 8.0;
  double retriangulationThreshold_ = 1.0e-3;

  bool addBetweenStereoFactors_ = true;

  // Inverse of variance
  double betweenRotationPrecision_ = 0.0;
  double betweenTranslationPrecision_ = 1 / (0.1 * 0.1);

  //! iSAM params
  double relinearizeThreshold_ = 1.0e-2;
  double relinearizeSkip_ = 1.0;
  double nr_states_ = 30.0;
  int numOptimize_ = 2;
  double wildfire_threshold_ = 0.001;
  bool useDogLeg_ = false;

  //! No Motion params
  double zero_velocity_precision_ = 1000;
  double no_motion_position_precision_ = 1000;
  double no_motion_rotation_precision_ = 10000;
  double constant_vel_precision_ = 100;

  //! Source of the initial guess for the keyframe pose
  PoseGuessSource pose_guess_source_ = PoseGuessSource::IMU;
  double mono_translation_scale_factor_ = 0.1;
  // Runtime toggle to use Graph Time Centric backend adapter
  bool use_graph_time_centric_ = true;
  
  // Add GP motion priors between consecutive keyframe states (GraphTimeCentric only)
  // Supports multiple GP model types (see gp_model_type)
  bool add_gp_motion_priors_ = false;
  
  // GP model type - all non-Full types can be combined without interference!
  // 0 = WNOA: White Noise on Acceleration (simplest, uses omega only)
  // 1 = WNOJ: White Noise on Jerk (uses omega + acceleration measurements)
  // 2 = WNOJFull: WNOJ with acceleration as optimized state (not implemented)
  // 3 = Singer: Exponentially decaying acceleration model
  // 4 = SingerFull: Singer with acceleration as optimized state (not implemented)
  // 5 = WNOA+WNOJ: Both WNOA and WNOJ simultaneously
  // 6 = WNOA+Singer: WNOA plus Singer decay model
  // 7 = WNOA+WNOJ+Singer: All three - maximum smoothing
  int gp_model_type_ = 0;  // Default: WNOA
  
  // Qc variance for GP priors (process noise / trajectory smoothness)
  // Lower values = smoother trajectory, higher values = more flexibility
  double qc_gp_trans_var_ = 1.0;   // Translation variance (m^2/s^4)
  double qc_gp_rot_var_ = 0.1;     // Rotation variance (rad^2/s^4)
  
  // Singer model acceleration damping (ad matrix diagonal)
  // Controls exponential decay: da/dt = -ad * a + noise
  // Higher = faster decay to zero acceleration
  double ad_trans_ = 1.0;          // Translation damping (1/s)
  double ad_rot_ = 2.0;            // Rotation damping (1/s)
  
  // Full variant parameters (for future WNOJFull/SingerFull)
  double initial_acc_sigma_trans_ = 0.5;   // m/s^2
  double initial_acc_sigma_rot_ = 0.1;     // rad/s^2
  
  // Omega measurement prior sigma (for GP motion priors)
  // Controls the tightness of angular velocity measurement constraints
  // Lower = tighter constraint (more trust in gyro), Higher = looser constraint
  // Typical range: 0.005-0.05 rad/s
  double omega_measurement_sigma_ = 0.005;  // rad/s (conservative gyro noise at keyframe rate)
  
  // Smoother lag calculation parameter
  // Expected average keyframe rate in Hz (used to convert nr_states to smoother lag in seconds)
  // Typical values: 5.0 for ~0.2s keyframe intervals, 10.0 for ~0.1s intervals
  double keyframe_rate_hz_ = 5.0;  // Default: 5Hz (0.2s keyframe interval)
  
  //! Debug params
  // Enable comprehensive factor graph logging when optimization fails
  // Outputs factor keys, .g2o files, and .dot graph visualizations
  bool enable_factor_graph_debug_logging_ = true;
  std::string factor_graph_debug_save_dir_ = "debug_run_logs";  // Directory for saved files
  int factor_graph_debug_save_interval_ = 1;  // Save after every N optimizations (1 = every time)
  bool factor_graph_debug_include_smart_factors_ = true;  // Include SmartStereoProjectionFactors in debug output
  int smootherType_ = 2;  // 0: ISAM2, 1: FixedLagSmoother, 2: Hybrid
};

}  // namespace VIO
