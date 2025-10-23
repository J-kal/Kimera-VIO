#include "kimera-vio/integration/GraphTimeCentricBackendAdapter.h"
#include <iostream>

using namespace kimera::integration;

struct GraphTimeCentricBackendAdapter::Impl {
    Impl() {}
};

GraphTimeCentricBackendAdapter::GraphTimeCentricBackendAdapter()
: impl_(new Impl())
{}

GraphTimeCentricBackendAdapter::~GraphTimeCentricBackendAdapter()
{
    delete impl_;
}

bool GraphTimeCentricBackendAdapter::initializeAdapter()
{
#ifdef ENABLE_GRAPH_TIME_CENTRIC_ADAPTER
    // In monorepo builds this would link against gnssFGO symbols directly.
    return true;
#else
    std::cerr << "GraphTimeCentric adapter not built in.\n";
    return false;
#endif
}

bool GraphTimeCentricBackendAdapter::addStateValues(unsigned long /*frame_id*/, double /*timestamp*/, const gtsam::NavState &/*navstate*/)
{
#ifdef ENABLE_GRAPH_TIME_CENTRIC_ADAPTER
    // translate and call into GraphTimeCentricKimera when available
    return true;
#else
    return false;
#endif
}

bool GraphTimeCentricBackendAdapter::addIMUTimestamps(const std::vector<double> &imu_timestamps)
{
#ifdef ENABLE_GRAPH_TIME_CENTRIC_ADAPTER
    // call into GraphTimeCentricKimera
    (void)imu_timestamps;
    return true;
#else
    return false;
#endif
}

bool GraphTimeCentricBackendAdapter::optimize(double /*timestep*/)
{
#ifdef ENABLE_GRAPH_TIME_CENTRIC_ADAPTER
    return true;
#else
    return false;
#endif
}

gtsam::Values GraphTimeCentricBackendAdapter::getLastResult()
{
#ifdef ENABLE_GRAPH_TIME_CENTRIC_ADAPTER
    return gtsam::Values();
#else
    return gtsam::Values();
#endif
}
