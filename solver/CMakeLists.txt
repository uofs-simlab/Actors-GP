# FitzHugh-Nagumo solver of the study, built on deal.II.
cmake_minimum_required(VERSION 3.5.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED true)

find_package(deal.II 9.4.0 REQUIRED
    HINTS ${DEAL_II_DIR} $ENV{DEAL_II_DIR})

if (NOT (${DEAL_II_WITH_UMFPACK}
    AND ${DEAL_II_WITH_COMPLEX_VALUES}
    AND ${DEAL_II_WITH_PETSC}))
    message(FATAL_ERROR "This solver requires deal.II configured with UMFPACK, complex values, and PETSc.")
endif()

deal_II_initialize_cached_variables()

project(fhn_solver VERSION 1.0.0)

# Two builds of the same source: the 2D wave-segment solver and the 1D radial solver.
foreach(target parallel_bdf_fhn radial_fhn_bdf)
    add_executable(${target} "parallel_bdf_fhn.cc")
    deal_II_setup_target(${target})
    set_target_properties(${target} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin")
endforeach()
target_compile_definitions(radial_fhn_bdf PRIVATE IGNITION_RADIAL_BDF)
