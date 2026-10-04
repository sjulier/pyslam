cmake_minimum_required(VERSION 3.4...3.18)
project(pydbow2)

set(CPP_STANDARD_VERSION "17" CACHE STRING "Desired C++ standard version") 

# Set the C++ standard 
set(CMAKE_CXX_STANDARD ${CPP_STANDARD_VERSION})
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING
      "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
      FORCE)
endif()

# PYSLAM_MARCH: the CPU the code is compiled for (the value of -march). The default, "native", is
# this machine's CPU. All of pySLAM's native modules and GTSAM must use the same value: they exchange
# Eigen objects, whose alignment depends on it. Prebuilt binaries use a common baseline (x86-64-v3).
if(NOT DEFINED PYSLAM_MARCH)
    if(DEFINED ENV{PYSLAM_MARCH} AND NOT "$ENV{PYSLAM_MARCH}" STREQUAL "")
        set(PYSLAM_MARCH "$ENV{PYSLAM_MARCH}")
    else()
        # the value recorded for the native modules already in this checkout (thirdparty/.pyslam_march,
        # written e.g. when prebuilt modules are installed), so that a module rebuilt later matches them
        set(PYSLAM_MARCH "native")
        set(_pyslam_dir "${CMAKE_CURRENT_SOURCE_DIR}")
        foreach(_pyslam_level RANGE 8)
            if(EXISTS "${_pyslam_dir}/thirdparty/.pyslam_march")
                file(STRINGS "${_pyslam_dir}/thirdparty/.pyslam_march" PYSLAM_MARCH LIMIT_COUNT 1)
                break()
            endif()
            get_filename_component(_pyslam_dir "${_pyslam_dir}" DIRECTORY)
        endforeach()
    endif()
endif()
message(STATUS "PYSLAM_MARCH: ${PYSLAM_MARCH}")

set(BUILD_WITH_MARCH_NATIVE ON CACHE BOOL "Build with -march=${PYSLAM_MARCH}")

LIST(APPEND CMAKE_MODULE_PATH 
    ${PROJECT_SOURCE_DIR}/cmake
)

# Generate file compile_commands.json in our build folder: it contains the full command line to compile individual source files
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")

if(BUILD_WITH_MARCH_NATIVE)
    if("${CMAKE_SYSTEM_NAME}" MATCHES "Linux")
        message(STATUS "Build with -march=${PYSLAM_MARCH}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=${PYSLAM_MARCH}")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=${PYSLAM_MARCH}")
    else()
        include(CheckCXXCompilerFlag)
        unset(COMPILER_SUPPORTS_MARCH_NATIVE CACHE)
        CHECK_CXX_COMPILER_FLAG(-march=${PYSLAM_MARCH} COMPILER_SUPPORTS_MARCH_NATIVE)
        if(COMPILER_SUPPORTS_MARCH_NATIVE AND NOT APPLE)
            message(STATUS "Build with -march=${PYSLAM_MARCH}")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=${PYSLAM_MARCH}")
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=${PYSLAM_MARCH}")
        endif()
    endif()
endif() 

add_subdirectory(modules/pybind11)
include("${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/pyslam_pybind11.cmake")
pyslam_use_bundled_pybind11("${CMAKE_CURRENT_SOURCE_DIR}/modules/pybind11")

find_package(OpenCV REQUIRED)

find_package(Boost REQUIRED COMPONENTS serialization filesystem)
message(STATUS "Boost_VERSION: ${Boost_VERSION}")
message(STATUS "Boost includes: ${Boost_INCLUDE_DIRS}")

set(DBOW2_DIR ${PROJECT_SOURCE_DIR}/modules/dbow2)
set(DBoW2_INCLUDE_DIRS ${DBOW2_DIR}/)
if(APPLE)
    set(DBoW2_LIB ${DBOW2_DIR}/lib/libDBoW2.dylib)
else()
    set(DBoW2_LIB ${DBOW2_DIR}/lib/libDBoW2.so)
endif()

include_directories(
    ${DBoW2_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
)
link_directories( 
    ${DBoW2_LIB}
)

if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.18)
    # When building a Python extension we should use the Module component to
    # avoid pulling in an extra libpython runtime (which can crash on import).
    # Development.Module requires CMake 3.18+
    find_package(Python3 COMPONENTS Interpreter Development.Module NumPy REQUIRED)
else()
    find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)
endif()
message(STATUS "Python3_EXECUTABLE: ${Python3_EXECUTABLE}")
message(STATUS "Python3_LIBRARIES: ${Python3_LIBRARIES}")

# Force bundled pybind11 to use the same interpreter
set(PYTHON_EXECUTABLE ${Python3_EXECUTABLE})

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

pybind11_add_module(pydbow2 
    src/py_dbow2.cpp
    src/ndarray_converter.cpp
)

target_link_libraries(pydbow2
PRIVATE
    ${OpenCV_LIBS}
    ${Boost_LIBRARIES}
    ${DBoW2_LIB}
    Python3::Module
    Python3::NumPy
)

target_compile_definitions(pydbow2 
PRIVATE 
    VERSION_INFO=${EXAMPLE_VERSION_INFO}
)
