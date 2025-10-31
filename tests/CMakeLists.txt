# Tests for Kimera-VIO integration with online_fgo_core
cmake_minimum_required(VERSION 3.16)

# Find GTest
find_package(GTest REQUIRED)
include(GoogleTest)

# GraphTimeCentric Backend Adapter Test (conditional)
if(ENABLE_GRAPH_TIME_CENTRIC_ADAPTER)
    message(STATUS "Adding GraphTimeCentric backend adapter test")
    
    add_executable(testGraphTimeCentricBackendAdapter
        testGraphTimeCentricBackendAdapter.cpp
    )
    
    target_include_directories(testGraphTimeCentricBackendAdapter PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
        ${GTEST_INCLUDE_DIRS}
    )
    
    target_link_libraries(testGraphTimeCentricBackendAdapter PRIVATE
        kimera_vio
        GTest::gtest
        GTest::gtest_main
        gtsam
        gtsam_unstable
    )
    
    # Link to online_fgo_core if available
    if(TARGET online_fgo_core)
        target_link_libraries(testGraphTimeCentricBackendAdapter PRIVATE
            online_fgo_core
        )
    endif()
    
    gtest_discover_tests(testGraphTimeCentricBackendAdapter
        DISCOVERY_MODE PRE_TEST
    )
    
    message(STATUS "  - testGraphTimeCentricBackendAdapter configured")
else()
    message(STATUS "ENABLE_GRAPH_TIME_CENTRIC_ADAPTER not set, skipping adapter tests")
endif()

# Add custom target to run all Kimera integration tests
if(ENABLE_GRAPH_TIME_CENTRIC_ADAPTER)
    add_custom_target(run_kimera_integration_tests
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -R "testGraphTimeCentricBackendAdapter"
        DEPENDS testGraphTimeCentricBackendAdapter
        COMMENT "Running Kimera integration tests"
    )
endif()
