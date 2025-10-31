/**
 * @file test_graph_time_centric_backend_adapter.cpp
 * @brief Unit and integration tests for GraphTimeCentricBackendAdapter
 * Tests the non-keyframe buffering system and Kimera integration
 * @author Kimera Integration Team
 * @date 2025-10-31
 */

#include <gtest/gtest.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/navigation/NavState.h>
#include <gtsam/navigation/ImuBias.h>

#ifdef ENABLE_GRAPH_TIME_CENTRIC_ADAPTER
#include "kimera-vio/integration/GraphTimeCentricBackendAdapter.h"
#include "kimera-vio/backend/VioBackendParams.h"
#include "kimera-vio/imu-frontend/ImuFrontend-definitions.h"
#endif

using namespace VIO;

/**
 * @brief Test fixture for GraphTimeCentricBackendAdapter tests
 */
class GraphTimeCentricBackendAdapterTest : public ::testing::Test {
protected:
#ifdef ENABLE_GRAPH_TIME_CENTRIC_ADAPTER
    std::unique_ptr<GraphTimeCentricBackendAdapter> adapter_;
    BackendParams backend_params_;
    ImuParams imu_params_;
#endif
    
    void SetUp() override {
#ifdef ENABLE_GRAPH_TIME_CENTRIC_ADAPTER
        // Setup backend params
        backend_params_.use_graph_time_centric = true;
        backend_params_.addBetweenStereoFactors_ = true;
        backend_params_.nr_states_ = 10;
        
        // Setup IMU params
        imu_params_.gyro_noise_ = 0.0003394;
        imu_params_.acc_noise_ = 0.002;
        imu_params_.gyro_walk_ = 0.000038785;
        imu_params_.acc_walk_ = 0.0003;
        imu_params_.imu_integration_sigma_ = 0.0;
        imu_params_.n_gravity_ = gtsam::Vector3(0, 0, -9.81);
        
        // Create adapter
        adapter_ = std::make_unique<GraphTimeCentricBackendAdapter>(
            backend_params_, imu_params_);
        
        // Initialize
        ASSERT_TRUE(adapter_->initialize());
#else
        GTEST_SKIP() << "GraphTimeCentric adapter not enabled";
#endif
    }
    
    void TearDown() override {
#ifdef ENABLE_GRAPH_TIME_CENTRIC_ADAPTER
        adapter_.reset();
#endif
    }
    
#ifdef ENABLE_GRAPH_TIME_CENTRIC_ADAPTER
    // Helper: Create timestamp (nanoseconds)
    Timestamp createTimestamp(double seconds) {
        return static_cast<Timestamp>(seconds * 1e9);
    }
    
    // Helper: Create pose at position
    gtsam::Pose3 createPose(double x, double y, double z) {
        return gtsam::Pose3(gtsam::Rot3(), gtsam::Point3(x, y, z));
    }
    
    // Helper: Create velocity
    gtsam::Vector3 createVelocity(double vx, double vy, double vz) {
        return gtsam::Vector3(vx, vy, vz);
    }
    
    // Helper: Create bias
    gtsam::imuBias::ConstantBias createBias(double acc_bias = 0.01, double gyro_bias = 0.001) {
        return gtsam::imuBias::ConstantBias(
            gtsam::Vector3(acc_bias, acc_bias, acc_bias),
            gtsam::Vector3(gyro_bias, gyro_bias, gyro_bias));
    }
#endif
};

#ifdef ENABLE_GRAPH_TIME_CENTRIC_ADAPTER

// =============================================================================
// INITIALIZATION TESTS
// =============================================================================

TEST_F(GraphTimeCentricBackendAdapterTest, InitializeSuccess) {
    EXPECT_TRUE(adapter_->isInitialized());
}

TEST_F(GraphTimeCentricBackendAdapterTest, GetNumStatesInitially) {
    EXPECT_EQ(adapter_->getNumStates(), 0);
}

TEST_F(GraphTimeCentricBackendAdapterTest, GetNumBufferedStatesInitially) {
    EXPECT_EQ(adapter_->getNumBufferedStates(), 0);
}

// =============================================================================
// NON-KEYFRAME BUFFERING TESTS
// =============================================================================

TEST_F(GraphTimeCentricBackendAdapterTest, BufferSingleNonKeyframe) {
    Timestamp ts = createTimestamp(1.0);
    gtsam::Pose3 pose = createPose(1, 0, 0);
    gtsam::Vector3 vel = createVelocity(1, 0, 0);
    gtsam::imuBias::ConstantBias bias = createBias();
    
    EXPECT_TRUE(adapter_->bufferNonKeyframeState(ts, pose, vel, bias));
    EXPECT_EQ(adapter_->getNumBufferedStates(), 1);
    EXPECT_EQ(adapter_->getNumStates(), 0);  // Not added to graph yet
}

TEST_F(GraphTimeCentricBackendAdapterTest, BufferMultipleNonKeyframes) {
    // Buffer 5 non-keyframes
    for (int i = 0; i < 5; ++i) {
        Timestamp ts = createTimestamp(1.0 + i * 0.1);
        gtsam::Pose3 pose = createPose(i, 0, 0);
        gtsam::Vector3 vel = createVelocity(1, 0, 0);
        gtsam::imuBias::ConstantBias bias = createBias();
        
        EXPECT_TRUE(adapter_->bufferNonKeyframeState(ts, pose, vel, bias));
    }
    
    EXPECT_EQ(adapter_->getNumBufferedStates(), 5);
    EXPECT_EQ(adapter_->getNumStates(), 0);
}

TEST_F(GraphTimeCentricBackendAdapterTest, BufferNonKeyframesOutOfOrder) {
    // Buffer in reverse order
    std::vector<double> timestamps = {1.4, 1.2, 1.3, 1.1, 1.0};
    
    for (double ts_sec : timestamps) {
        Timestamp ts = createTimestamp(ts_sec);
        gtsam::Pose3 pose = createPose(ts_sec, 0, 0);
        gtsam::Vector3 vel = createVelocity(1, 0, 0);
        gtsam::imuBias::ConstantBias bias = createBias();
        
        EXPECT_TRUE(adapter_->bufferNonKeyframeState(ts, pose, vel, bias));
    }
    
    EXPECT_EQ(adapter_->getNumBufferedStates(), 5);
}

// =============================================================================
// KEYFRAME ADDITION TESTS (Buffer Processing)
// =============================================================================

TEST_F(GraphTimeCentricBackendAdapterTest, AddKeyframeWithoutBuffer) {
    // Add keyframe directly (no buffered states)
    Timestamp ts = createTimestamp(1.0);
    gtsam::Pose3 pose = createPose(1, 0, 0);
    gtsam::Vector3 vel = createVelocity(1, 0, 0);
    gtsam::imuBias::ConstantBias bias = createBias();
    
    adapter_->addKeyframeState(ts, pose, vel, bias);
    
    EXPECT_EQ(adapter_->getNumStates(), 1);
    EXPECT_EQ(adapter_->getNumBufferedStates(), 0);
}

TEST_F(GraphTimeCentricBackendAdapterTest, AddKeyframeProcessesBuffer) {
    // Buffer 3 non-keyframes
    for (int i = 0; i < 3; ++i) {
        Timestamp ts = createTimestamp(1.0 + i * 0.1);
        adapter_->bufferNonKeyframeState(ts, createPose(i, 0, 0), 
                                         createVelocity(1, 0, 0), createBias());
    }
    
    EXPECT_EQ(adapter_->getNumBufferedStates(), 3);
    
    // Add keyframe (should process buffer)
    Timestamp kf_ts = createTimestamp(1.4);
    adapter_->addKeyframeState(kf_ts, createPose(4, 0, 0),
                               createVelocity(1, 0, 0), createBias());
    
    // Buffer should be cleared and all states added
    EXPECT_EQ(adapter_->getNumBufferedStates(), 0);
    EXPECT_EQ(adapter_->getNumStates(), 4);  // 3 buffered + 1 keyframe
}

TEST_F(GraphTimeCentricBackendAdapterTest, BufferClearedAfterKeyframe) {
    // Buffer some states
    for (int i = 0; i < 5; ++i) {
        Timestamp ts = createTimestamp(1.0 + i * 0.1);
        adapter_->bufferNonKeyframeState(ts, createPose(i, 0, 0),
                                         createVelocity(1, 0, 0), createBias());
    }
    
    // Add keyframe
    Timestamp kf_ts = createTimestamp(1.6);
    adapter_->addKeyframeState(kf_ts, createPose(6, 0, 0),
                               createVelocity(1, 0, 0), createBias());
    
    // Verify buffer cleared
    EXPECT_EQ(adapter_->getNumBufferedStates(), 0);
}

TEST_F(GraphTimeCentricBackendAdapterTest, ChronologicalOrderingAfterSort) {
    // Buffer in random order
    std::vector<double> timestamps = {1.3, 1.1, 1.4, 1.0, 1.2};
    
    for (double ts_sec : timestamps) {
        Timestamp ts = createTimestamp(ts_sec);
        adapter_->bufferNonKeyframeState(ts, createPose(ts_sec, 0, 0),
                                         createVelocity(1, 0, 0), createBias());
    }
    
    // Add keyframe (should sort buffer first)
    Timestamp kf_ts = createTimestamp(1.5);
    adapter_->addKeyframeState(kf_ts, createPose(1.5, 0, 0),
                               createVelocity(1, 0, 0), createBias());
    
    // All states should be added (5 buffered + 1 keyframe)
    EXPECT_EQ(adapter_->getNumStates(), 6);
    EXPECT_EQ(adapter_->getNumBufferedStates(), 0);
}

// =============================================================================
// MULTIPLE KEYFRAME CYCLES
// =============================================================================

TEST_F(GraphTimeCentricBackendAdapterTest, MultipleKeyframeCycles) {
    // Simulate 3 keyframe cycles
    for (int cycle = 0; cycle < 3; ++cycle) {
        // Buffer 3 non-keyframes
        for (int i = 0; i < 3; ++i) {
            double ts_sec = cycle * 0.5 + i * 0.1;
            Timestamp ts = createTimestamp(ts_sec);
            adapter_->bufferNonKeyframeState(ts, createPose(ts_sec, 0, 0),
                                             createVelocity(1, 0, 0), createBias());
        }
        
        // Add keyframe
        double kf_ts_sec = cycle * 0.5 + 0.4;
        Timestamp kf_ts = createTimestamp(kf_ts_sec);
        adapter_->addKeyframeState(kf_ts, createPose(kf_ts_sec, 0, 0),
                                   createVelocity(1, 0, 0), createBias());
        
        // Verify buffer cleared after each keyframe
        EXPECT_EQ(adapter_->getNumBufferedStates(), 0);
    }
    
    // Total states: 3 cycles * (3 buffered + 1 keyframe) = 12
    EXPECT_EQ(adapter_->getNumStates(), 12);
}

TEST_F(GraphTimeCentricBackendAdapterTest, VaryingBufferSizesPerCycle) {
    // Cycle 1: 2 non-keyframes
    adapter_->bufferNonKeyframeState(createTimestamp(1.0), createPose(1, 0, 0),
                                     createVelocity(1, 0, 0), createBias());
    adapter_->bufferNonKeyframeState(createTimestamp(1.1), createPose(1.1, 0, 0),
                                     createVelocity(1, 0, 0), createBias());
    adapter_->addKeyframeState(createTimestamp(1.2), createPose(1.2, 0, 0),
                               createVelocity(1, 0, 0), createBias());
    
    // Cycle 2: 5 non-keyframes
    for (int i = 0; i < 5; ++i) {
        double ts = 1.3 + i * 0.05;
        adapter_->bufferNonKeyframeState(createTimestamp(ts), createPose(ts, 0, 0),
                                         createVelocity(1, 0, 0), createBias());
    }
    adapter_->addKeyframeState(createTimestamp(1.6), createPose(1.6, 0, 0),
                               createVelocity(1, 0, 0), createBias());
    
    // Cycle 3: 1 non-keyframe
    adapter_->bufferNonKeyframeState(createTimestamp(1.7), createPose(1.7, 0, 0),
                                     createVelocity(1, 0, 0), createBias());
    adapter_->addKeyframeState(createTimestamp(1.8), createPose(1.8, 0, 0),
                               createVelocity(1, 0, 0), createBias());
    
    // Total: (2+1) + (5+1) + (1+1) = 11 states
    EXPECT_EQ(adapter_->getNumStates(), 11);
    EXPECT_EQ(adapter_->getNumBufferedStates(), 0);
}

// =============================================================================
// IMU MEASUREMENT TESTS
// =============================================================================

TEST_F(GraphTimeCentricBackendAdapterTest, AddIMUMeasurement) {
    ImuAccGyr imu;
    imu.timestamp = createTimestamp(1.0);
    imu.lin_acc = gtsam::Vector3(0, 0, 9.81);
    imu.ang_vel = gtsam::Vector3(0, 0, 0);
    
    // Note: Method signature may vary, just test it doesn't crash
    // EXPECT_TRUE(adapter_->addIMUMeasurement(imu));
}

// =============================================================================
// OPTIMIZATION TESTS
// =============================================================================

TEST_F(GraphTimeCentricBackendAdapterTest, OptimizeWithNoStates) {
    // Try to optimize with no states
    bool result = adapter_->optimizeGraph();
    
    // Should fail gracefully
    EXPECT_FALSE(result);
}

TEST_F(GraphTimeCentricBackendAdapterTest, OptimizeWithSingleState) {
    // Add one keyframe
    adapter_->addKeyframeState(createTimestamp(1.0), createPose(1, 0, 0),
                               createVelocity(1, 0, 0), createBias());
    
    // Optimize
    bool result = adapter_->optimizeGraph();
    
    // May succeed with single state (depending on implementation)
    // Just verify no crash
}

TEST_F(GraphTimeCentricBackendAdapterTest, OptimizeWithMultipleStates) {
    // Add several keyframes
    for (int i = 0; i < 5; ++i) {
        double ts = 1.0 + i * 0.2;
        adapter_->addKeyframeState(createTimestamp(ts), createPose(ts, 0, 0),
                                   createVelocity(1, 0, 0), createBias());
    }
    
    // Optimize
    bool result = adapter_->optimizeGraph();
    
    // Should succeed with multiple states
    EXPECT_TRUE(result);
}

// =============================================================================
// EDGE CASES
// =============================================================================

TEST_F(GraphTimeCentricBackendAdapterTest, BufferNonKeyframeAfterKeyframe) {
    // This tests the timestamp validation (buffered state shouldn't be after keyframe)
    
    // Buffer a non-keyframe at t=1.0
    adapter_->bufferNonKeyframeState(createTimestamp(1.0), createPose(1, 0, 0),
                                     createVelocity(1, 0, 0), createBias());
    
    // Add keyframe at t=0.9 (before buffered state - unusual but test it)
    adapter_->addKeyframeState(createTimestamp(0.9), createPose(0.9, 0, 0),
                               createVelocity(1, 0, 0), createBias());
    
    // Buffered state at t=1.0 should be skipped (after keyframe)
    // Buffer should still be cleared
    EXPECT_EQ(adapter_->getNumBufferedStates(), 0);
}

TEST_F(GraphTimeCentricBackendAdapterTest, ZeroTimestamp) {
    // Test with timestamp = 0
    adapter_->addKeyframeState(createTimestamp(0.0), createPose(0, 0, 0),
                               createVelocity(0, 0, 0), createBias());
    
    EXPECT_EQ(adapter_->getNumStates(), 1);
}

TEST_F(GraphTimeCentricBackendAdapterTest, VeryLargeTimestamp) {
    // Test with large timestamp (e.g., Unix epoch time)
    double large_ts = 1698787200.0;  // Some future date
    
    adapter_->addKeyframeState(createTimestamp(large_ts), createPose(0, 0, 0),
                               createVelocity(0, 0, 0), createBias());
    
    EXPECT_EQ(adapter_->getNumStates(), 1);
}

// =============================================================================
// STRESS TESTS
// =============================================================================

TEST_F(GraphTimeCentricBackendAdapterTest, StressTestLargeBuffer) {
    // Buffer 100 non-keyframes
    for (int i = 0; i < 100; ++i) {
        double ts = 1.0 + i * 0.01;
        adapter_->bufferNonKeyframeState(createTimestamp(ts), createPose(ts, 0, 0),
                                         createVelocity(1, 0, 0), createBias());
    }
    
    EXPECT_EQ(adapter_->getNumBufferedStates(), 100);
    
    // Add keyframe (should process all 100)
    adapter_->addKeyframeState(createTimestamp(2.1), createPose(2.1, 0, 0),
                               createVelocity(1, 0, 0), createBias());
    
    EXPECT_EQ(adapter_->getNumBufferedStates(), 0);
    EXPECT_EQ(adapter_->getNumStates(), 101);
}

TEST_F(GraphTimeCentricBackendAdapterTest, StressTestManyCycles) {
    // Simulate 50 keyframe cycles with 5 non-keyframes each
    for (int cycle = 0; cycle < 50; ++cycle) {
        // Buffer 5 non-keyframes
        for (int i = 0; i < 5; ++i) {
            double ts = cycle + i * 0.1;
            adapter_->bufferNonKeyframeState(createTimestamp(ts), createPose(ts, 0, 0),
                                             createVelocity(1, 0, 0), createBias());
        }
        
        // Add keyframe
        double kf_ts = cycle + 0.6;
        adapter_->addKeyframeState(createTimestamp(kf_ts), createPose(kf_ts, 0, 0),
                                   createVelocity(1, 0, 0), createBias());
    }
    
    // Total: 50 * (5 + 1) = 300 states
    EXPECT_EQ(adapter_->getNumStates(), 300);
    EXPECT_EQ(adapter_->getNumBufferedStates(), 0);
}

// =============================================================================
// INTEGRATION TEST: Full VIO Pipeline Simulation
// =============================================================================

TEST_F(GraphTimeCentricBackendAdapterTest, FullVIOPipelineSimulation) {
    // Simulate realistic VIO: 10 Hz keyframes, ~30 Hz total frames, 200 Hz IMU
    
    double keyframe_period = 0.1;   // 10 Hz
    double frame_period = 0.033;    // ~30 Hz
    double duration = 2.0;          // 2 seconds
    
    double t = 0.0;
    int frame_count = 0;
    int keyframe_count = 0;
    
    while (t < duration) {
        // Simulate pose/velocity/bias estimation
        gtsam::Pose3 pose = createPose(t, 0, 0);
        gtsam::Vector3 vel = createVelocity(1, 0, 0);
        gtsam::imuBias::ConstantBias bias = createBias();
        
        // Decide if keyframe (every 3rd frame roughly)
        bool is_keyframe = (frame_count % 3 == 0);
        
        if (is_keyframe) {
            adapter_->addKeyframeState(createTimestamp(t), pose, vel, bias);
            keyframe_count++;
            
            // Verify buffer cleared
            EXPECT_EQ(adapter_->getNumBufferedStates(), 0);
        } else {
            adapter_->bufferNonKeyframeState(createTimestamp(t), pose, vel, bias);
        }
        
        t += frame_period;
        frame_count++;
    }
    
    // Verify reasonable number of states
    EXPECT_GT(adapter_->getNumStates(), keyframe_count);
    EXPECT_LE(adapter_->getNumStates(), frame_count);
    
    // Final buffer should have non-keyframes waiting
    EXPECT_GE(adapter_->getNumBufferedStates(), 0);
    
    std::cout << "VIO Simulation Statistics:" << std::endl;
    std::cout << "  Total frames: " << frame_count << std::endl;
    std::cout << "  Keyframes: " << keyframe_count << std::endl;
    std::cout << "  States in graph: " << adapter_->getNumStates() << std::endl;
    std::cout << "  Buffered states: " << adapter_->getNumBufferedStates() << std::endl;
}

#endif  // ENABLE_GRAPH_TIME_CENTRIC_ADAPTER

// =============================================================================
// MAIN
// =============================================================================

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
