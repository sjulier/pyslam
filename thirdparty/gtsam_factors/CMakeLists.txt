cmake_minimum_required(VERSION 3.10)
project(gtsam_factors)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(SET_RPATH               ON  CACHE BOOL   "Set RPATH instead of the default RUNPATH") # for solving the linking issue with g2o binaries (see the related node in the compiler options section)
set(WITH_PYTHON_INTERP_CHECK OFF CACHE BOOL  "Checking python interpreter") # to be activated when called within virtual python environment 
# PYSLAM_MARCH: the CPU the code is compiled for (the value of -march). The default, "native", is
# this machine's CPU. All of pySLAM's native modules and GTSAM must use the same value: they exchange
# Eigen objects, whose alignment depends on it. Prebuilt binaries use a common baseline (x86-64-v3).
if(NOT DEFINED PYSLAM_MARCH)
    if(DEFINED ENV{PYSLAM_MARCH} AND NOT "$ENV{PYSLAM_MARCH}" STREQUAL "")
        set(PYSLAM_MARCH "$ENV{PYSLAM_MARCH}")
    else()
        set(PYSLAM_MARCH "native")
    endif()
endif()
message(STATUS "PYSLAM_MARCH: ${PYSLAM_MARCH}")

set(WITH_MARCH_NATIVE OFF  CACHE BOOL  "Build with \"-march native\"")

# In order to set RPATH (instead of RUNPATH) for g2o and other libraries linking issue.
if(LINUX)
  if(SET_RPATH)
    set(MY_FLAGS "${MY_FLAGS} -Wl,--disable-new-dtags")
  endif()
endif()

if(APPLE)
  set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
  set(THREADS_PREFER_PTHREAD_FLAG TRUE)
  find_package(Threads REQUIRED)
endif()

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -O3 -fPIC -DNDEBUG ${MY_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fPIC -DNDEBUG ${MY_FLAGS}")

if (WITH_MARCH_NATIVE)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=${PYSLAM_MARCH}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=${PYSLAM_MARCH}")
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CMAKE_CXX_STANDARD: ${CMAKE_CXX_STANDARD}")
message(STATUS "CXX flags: ${CMAKE_CXX_FLAGS}")

# eigen 
find_package(Eigen3 CONFIG REQUIRED)
# Get the include directory from the target (modern Eigen3 uses targets)
if(TARGET Eigen3::Eigen)
    get_target_property(EIGEN3_INCLUDE_DIR Eigen3::Eigen INTERFACE_INCLUDE_DIRECTORIES)
    message(STATUS "eigen include: ${EIGEN3_INCLUDE_DIR}")
else()
    # Fallback for older Eigen3 installations
    message(STATUS "eigen include: ${EIGEN3_INCLUDE_DIR}")
endif()

# gtsam
# Only accept the local GTSAM built by scripts/install_gtsam.sh. Without NO_DEFAULT_PATH,
# CMake silently falls back to system copies (e.g. /usr/local, /opt/homebrew, $CONDA_PREFIX).
# GTSAM_DIR and GTSAM_INCLUDE_DIR are cache entries, so clear them in case a previous
# configure resolved them to a system copy.
get_filename_component(GTSAM_LOCAL_ROOT "${PROJECT_SOURCE_DIR}/../gtsam_local" ABSOLUTE)
set(GTSAM_CONFIG_DIR "${GTSAM_LOCAL_ROOT}/install/lib/cmake/GTSAM")
unset(GTSAM_DIR CACHE)
unset(GTSAM_INCLUDE_DIR CACHE)
find_package(GTSAM CONFIG PATHS "${GTSAM_CONFIG_DIR}" NO_DEFAULT_PATH)
if(NOT GTSAM_FOUND)
  message(FATAL_ERROR "Local GTSAM not found in ${GTSAM_CONFIG_DIR}. Run scripts/install_gtsam.sh first.")
endif()
message(STATUS "gtsam config: ${GTSAM_DIR} (version ${GTSAM_VERSION})")
message(STATUS "gtsam include: ${GTSAM_INCLUDE_DIR}")
# Imported targets pass their include dirs as -isystem, which cancels a -I for the same dir, and
# Apple clang searches /usr/local/include before -isystem dirs. So a GTSAM in /usr/local would
# shadow the local headers. Keep the local GTSAM include dir a regular -I (CMake >= 3.25).
set_property(TARGET gtsam PROPERTY SYSTEM FALSE)
message(STATUS "gtsam lib dirs: ${GTSAM_LIBRARY_DIRS}")
message(STATUS "gtsam libs: ${GTSAM_LIBS}")



# if (APPLE)
#   find_package(TBB REQUIRED)
#   if (TBB_FOUND)
#     message(STATUS "TBB found")
#     if (NOT TBB_LIBRARIES)
#       set(TBB_LIBRARIES TBB::tbb)
#     endif()
#   endif()
#   message(STATUS "TBB include: ${TBB_INCLUDE_DIRS}")
#   message(STATUS "TBB libs: ${TBB_LIBRARIES}")
#   message(STATUS "TBB library dirs: ${TBB_LIBRARY_DIRS}")
# endif()

# the following 2 lines are added to correctly detect the python version 
if(WITH_PYTHON_INTERP_CHECK)
  message(STATUS "WITH_PYTHON_INTERP_CHECK: ${WITH_PYTHON_INTERP_CHECK}")
  find_package(PythonInterp) 
  #find_package(PythonLibs)
  message(STATUS "PythonInterp: ${PythonInterp}")
  #message(STATUS "PythonLibs: ${PythonLibs}")
endif()

# Put the local GTSAM headers first so that other include dirs (e.g. /usr/local/include
# pulled in via Python, Eigen, Boost or TBB) cannot shadow them with a different GTSAM version.
include_directories(BEFORE ${GTSAM_INCLUDE_DIR})
include_directories(
    ${PYTHON_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${TBB_INCLUDE_DIRS}
)

# link_directories( 
#   ${GTSAM_LIBRARY_DIRS}
# )

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

add_subdirectory(pybind11)

pybind11_add_module(
  gtsam_factors 
  gtsam_factors_module.cpp 
  similarity.h
  switchable_robust_noise_model.h
  optimizers.h 
  resectioning.h
  weighted_projection_factors.h
)
target_link_libraries(gtsam_factors PRIVATE
  gtsam
  ${TBB_LIBRARIES}
)

# # Install header files
# install(FILES
#   similarity.h
#   switchable_robust_noise_model.h
#   optimizers.h
#   resectioning.h
#   weighted_projection_factors.h
#   DESTINATION include/gtsam_factors
# )



