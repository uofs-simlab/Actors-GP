cmake_minimum_required(VERSION 3.5.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED true)

project(ignition_actor VERSION 1.0.0)

find_package(Threads REQUIRED)

# CAF (C++ Actor Framework)
# release: the default was <CAF prefix> written out; it is now taken from the environment
set(CAF_ROOT "$ENV{CAF_ROOT}" CACHE PATH "CAF installation root (-DCAF_ROOT or the CAF_ROOT environment variable)")
find_path(CAF_INCLUDE_DIR caf/all.hpp HINTS "${CAF_ROOT}/include")
find_library(CAF_CORE_LIB caf_core HINTS "${CAF_ROOT}/lib64" "${CAF_ROOT}/lib")
find_library(CAF_IO_LIB caf_io HINTS "${CAF_ROOT}/lib64" "${CAF_ROOT}/lib")

if(NOT CAF_INCLUDE_DIR OR NOT CAF_CORE_LIB OR NOT CAF_IO_LIB)
    message(FATAL_ERROR
        "Could not find CAF. Set -DCAF_ROOT=<path> to the CAF install prefix.")
endif()

message(STATUS "Found CAF: ${CAF_CORE_LIB}")

# ignition_actor: CAF-based critical-amplitude sweep
# (no deal.II needed at link)
add_executable(ignition_actor
    src/radial_ignition_actor.cc
    src/server.cc
    src/gp_advisor_process.cc
    src/client.cc
    src/worker.cc)

target_include_directories(ignition_actor PRIVATE "${CAF_INCLUDE_DIR}")

target_link_libraries(ignition_actor
    ${CAF_CORE_LIB}
    ${CAF_IO_LIB}
    Threads::Threads)

set_target_properties(ignition_actor PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin")

set_target_properties(ignition_actor PROPERTIES
    INSTALL_RPATH "${CAF_ROOT}/lib64:${CAF_ROOT}/lib"
    BUILD_WITH_INSTALL_RPATH TRUE)
