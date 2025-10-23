// GraphTimeCentricBackendAdapter.h - thin adapter to call into GNSSFGO GraphTimeCentricKimera

#pragma once

#include <gtsam/nonlinear/Values.h>
#include <vector>

namespace kimera {
namespace integration {

class GraphTimeCentricBackendAdapter {
public:
    GraphTimeCentricBackendAdapter();
    ~GraphTimeCentricBackendAdapter();

    bool initializeAdapter();

    bool addStateValues(unsigned long frame_id, double timestamp, const gtsam::NavState &navstate);

    bool addIMUTimestamps(const std::vector<double> &imu_timestamps);

    bool optimize(double timestep);

    gtsam::Values getLastResult();

private:
    // Opaque pointer to implementation (keeps header light)
    struct Impl;
    Impl *impl_ = nullptr;
};

} // namespace integration
} // namespace kimera
