# Copyright (c) 2016 Intel Corporation

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#      http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required (VERSION 3.1 FATAL_ERROR)
include(CheckCXXCompilerFlag)

# Register Intel helper modules.
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/utils/build/cmake/modules")
include(IntelHelpers)

# ======================================================================================================

# Writes debug message in project (it will be presented only if debug is enabled).
#
# @param text Text to present.
function(cldnn_dmsg text)
  if(CLDNN__CMAKE_DEBUG)
    message(STATUS "[clDNN][D] ${text}")
  endif()
endfunction()

# ======================================================================================================
# ==================================== BUILD CONFIGURATIONS (part 1) ===================================
# ======================================================================================================
# The section must be before project definition which is a point for configurations generation
# for multi-configuration generators.

# Available configuration types:
# Two standard configurations.
set(CMAKE_CONFIGURATION_TYPES
    "Debug"
    "Release"
  )
set(CMAKE_CONFIGURATION_TYPES ${CMAKE_CONFIGURATION_TYPES} CACHE INTERNAL "Available build configurations.")

# ======================================================================================================
# ======================================================================================================
# ======================================================================================================

# Name of project (helper constant variable).
set(CLDNN__PROJ_NAME "clDNN")

# Default languages: C, C++.
project("${CLDNN__PROJ_NAME}")


# ======================================================================================================
# ==================================== BUILD CONFIGURATIONS (part 2) ===================================
# ======================================================================================================
# The section must be after project definition and its supported languages because otherwise
# used variables are not available yet.

#intel_custom_build_add("<custom config>" "Release") # No custom configuration at the momeent.
# Populating global property with list of debug configurations.
set_property(GLOBAL PROPERTY DEBUG_CONFIGURATIONS "Debug")
# Use solution folders.
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# ======================================================================================================
# ======================================================================================================
# ======================================================================================================


# ======================================================================================================
# ====================================== HELPER CONSTANT VARIABLES =====================================
# ======================================================================================================

# Path which points to directory with common dependencies (internal and 3-rd party).
set(CLDNN__COMMON_DIR   "${CMAKE_CURRENT_SOURCE_DIR}/common")

# Path which points to directory with binaries for Intel OpenCL SDK ICD (Installable Client Driver).
set(CLDNN__IOCL_ICD_DIR "${CLDNN__COMMON_DIR}/intel_ocl_icd")

# Build targets settings.

# Path which points to default root directory for compilation output.
set(CLDNN_BUILD__DEFAULT_OUT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/build/out")

# Prefix for all targets in internal pass.
set(CLDNN_BUILD__PROJ_NAME_PREFIX "")

# ======================================================================================================
# ======================================================================================================
# ======================================================================================================


# Expressing CMake setting for current build configuration as option and providing a way to correct it (treat as case-insensitive).
if(DEFINED CMAKE_BUILD_TYPE)
  string(TOLOWER "${CMAKE_BUILD_TYPE}" __CLDNN_BuildType)
  string(STRIP "${__CLDNN_BuildType}" __CLDNN_BuildType)

  if(__CLDNN_BuildType MATCHES "^release$")
    set(__CLDNN_BuildType "Release")
  elseif(__CLDNN_BuildType MATCHES "^debug$")
    set(__CLDNN_BuildType "Debug")
  else()
    set(__CLDNN_BuildType "Debug")
    if(NOT (MSVC OR DEFINED XCODE_VERSION))
      message(WARNING "[clDNN] CMAKE_BUILD_TYPE: Unknown build configuration. The following configurations are available: ${CMAKE_CONFIGURATION_TYPES}.
          The \"${__CLDNN_BuildType}\" configuration will be used.

          This value has meaning only for single-configuration generators (like Make). It will be ignored for MSVC/XCode."
        )
    endif()
  endif()
else()
  set(__CLDNN_BuildType "Debug")
  if(NOT (MSVC OR DEFINED XCODE_VERSION))
    message(WARNING "[clDNN] CMAKE_BUILD_TYPE: No build configuration specified. The following configurations are available: ${CMAKE_CONFIGURATION_TYPES}.
        The \"${__CLDNN_BuildType}\" configuration will be used.

        This value has meaning only for single-configuration generators (like Make). It will be ignored for MSVC/XCode."
      )
  endif()
endif()
set(CMAKE_BUILD_TYPE "${__CLDNN_BuildType}")
unset(__CLDNN_BuildType)

# ======================================================================================================

# Detecting, setting and validating target architecture for compilation.
intel_arch_detect(__CLDNN_DetectedArch_Target __CLDNN_DetectedArch_Host)
intel_arch_validate(__CLDNN_DetectedArchValid_Target "${__CLDNN_DetectedArch_Target}")
unset(__CLDNN_DetectedArch_Host)

if(DEFINED CLDNN__ARCHITECTURE_TARGET)
  intel_arch_normalize(__CLDNN_Arch_Target "${CLDNN__ARCHITECTURE_TARGET}")
elseif(__CLDNN_DetectedArchValid_Target)
  set(__CLDNN_Arch_Target "${__CLDNN_DetectedArch_Target}")
  message("[clDNN] CLDNN__ARCHITECTURE_TARGET: Target architecture is not specified. Trying to deduce it from context.")
else()
  message(FATAL_ERROR "[clDNN] CLDNN__ARCHITECTURE_TARGET: Target architecture is not specified and cannot be deduced from context.
      Please specify one, e.g. Windows32, Linux64, Android32, Darwin32, ..."
    )
endif()
set(CLDNN__ARCHITECTURE_TARGET "${__CLDNN_Arch_Target}")
unset(__CLDNN_Arch_Target)
unset(__CLDNN_DetectedArchValid_Target)

intel_arch_validate(__CLDNN_ArchValid_Target "${CLDNN__ARCHITECTURE_TARGET}")
if(NOT __CLDNN_ArchValid_Target)
  message(FATAL_ERROR "[clDNN] CLDNN__ARCHITECTURE_TARGET: Target architecture \"${CLDNN__ARCHITECTURE_TARGET}\" is invalid.
      Please specify correct one, e.g. Windows32, Linux64, Android32, Darwin32, ..."
    )
endif()
unset(__CLDNN_ArchValid_Target)

# ======================================================================================================

# Detecting all versions of added Intel OpenCL SDK ICDs in the project.
if(NOT EXISTS "${CLDNN__IOCL_ICD_DIR}")
  if(NOT CLDNN__IOCL_ICD_USE_EXTERNAL)
    message(FATAL_ERROR "[clDNN] Root for Intel OpenCL SDK ICDs is invalid or does not exist.
      Please make sure that ICDs are added into \"${CLDNN__IOCL_ICD_DIR}\".")
  endif()
endif()

file(GLOB __CLDNN_IOclIcdVersionPaths RELATIVE "${CLDNN__IOCL_ICD_DIR}" "${CLDNN__IOCL_ICD_DIR}/[0-9rRvV]*")
set(__CLDNN_IOclIcdVersions "")
set(__CLDNN_IOclIcdPaths    "")
foreach(__CLDNN_IOclIcdVersionPath ${__CLDNN_IOclIcdVersionPaths})
  if(__CLDNN_IOclIcdVersionPath MATCHES "^[rRvV]?([0-9]+([\\._][0-9]+([\\._][0-9]+([\\._][0-9]+)?)?)?)")
    string(REPLACE "_" "." __CLDNN_IOclIcdVersion "${CMAKE_MATCH_1}")
    set(__CLDNN_IOclIcdPath "${CLDNN__IOCL_ICD_DIR}/${__CLDNN_IOclIcdVersionPath}")

    set(__CLDNN_IOclIcdPathValid TRUE)
    file(GLOB __CLDNN_IOclIcdTargets "${__CLDNN_IOclIcdPath}/*/include/CL/opencl.h")
    list(LENGTH __CLDNN_IOclIcdTargets __CLDNN_IOclIcdTargetsCount)
    if(__CLDNN_IOclIcdTargetsCount LESS 1)
        message(WARNING "[clDNN] ICD for Intel OpenCL SDK at \"${__CLDNN_IOclIcdPath}\" does not contain required header files. It will be omitted.")
        set(__CLDNN_IOclIcdPathValid FALSE)
    endif()
    unset(__CLDNN_IOclIcdTargets)
    unset(__CLDNN_IOclIcdTargetsCount)

    if(__CLDNN_IOclIcdPathValid)
      list(APPEND __CLDNN_IOclIcdVersions "${__CLDNN_IOclIcdVersion}")
      list(APPEND __CLDNN_IOclIcdPaths    "${__CLDNN_IOclIcdPath}")
    endif()
  endif()
endforeach()
unset(__CLDNN_IOclIcdVersionPaths)
unset(__CLDNN_IOclIcdVersionPath)
unset(__CLDNN_IOclIcdVersion)
unset(__CLDNN_IOclIcdPath)
unset(__CLDNN_IOclIcdPathValid)

# Selecting highest version of ICD as default version.
set(__CLDNN_IOclIcdDefaultVersion "0")
foreach(__CLDNN_IOclIcdVersion ${__CLDNN_IOclIcdVersions})
  if(__CLDNN_IOclIcdVersion VERSION_GREATER __CLDNN_IOclIcdDefaultVersion)
    set(__CLDNN_IOclIcdDefaultVersion "${__CLDNN_IOclIcdVersion}")
  endif()
endforeach()
unset(__CLDNN_IOclIcdVersion)

if(__CLDNN_IOclIcdDefaultVersion VERSION_EQUAL "0")
  if(NOT CLDNN__IOCL_ICD_USE_EXTERNAL)
    message(FATAL_ERROR "[clDNN] Intel OpenCL SDK ICD required for build cannot be localized.")
  endif()
endif()


# ======================================================================================================
# ============================================ CMAKE OPTIONS ===========================================
# ======================================================================================================

# Current build configuration (only for single-configuration generators).
set(CMAKE_BUILD_TYPE "${CMAKE_BUILD_TYPE}" CACHE STRING "Current build configuration (only for single-configuration generators)." FORCE)
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS ${CMAKE_CONFIGURATION_TYPES})

# ======================================================================================================

# Target architecture for compilation.
set(CLDNN__ARCHITECTURE_TARGET "${CLDNN__ARCHITECTURE_TARGET}" CACHE STRING "Target architecture for compilation." FORCE)
if(MSVC)
  set_property(CACHE CLDNN__ARCHITECTURE_TARGET PROPERTY STRINGS "Windows32" "Windows64")
elseif(DEFINED XCODE_VERSION)
  set_property(CACHE CLDNN__ARCHITECTURE_TARGET PROPERTY STRINGS "Darwin32" "Darwin64")
endif()

# ======================================================================================================

# Output directory path where the final libraries, examples and tests will be stored.
set(CLDNN__OUTPUT_DIR "${CLDNN_BUILD__DEFAULT_OUT_ROOT}/${CLDNN__ARCHITECTURE_TARGET}/${CMAKE_CFG_INTDIR}" CACHE PATH "Output directory path where the final libraries, examples and tests will be stored.")

# ======================================================================================================

# Intel OpenCL SDK ICD: Version of ICD used to build clDNN.
set(CLDNN__IOCL_ICD_VERSION "${__CLDNN_IOclIcdDefaultVersion}" CACHE STRING "Intel OpenCL SDK ICD: Version of ICD used to build clDNN framework.")
mark_as_advanced(CLDNN__IOCL_ICD_VERSION)
set_property(CACHE CLDNN__IOCL_ICD_VERSION PROPERTY STRINGS ${__CLDNN_IOclIcdVersions})

# ======================================================================================================

set(CLDNN__IOCL_ICD_USE_EXTERNAL ON CACHE BOOL "Intel OpenCL SDK ICD: Try to use externally-intalled Intel OpenCL SDK ICD.")
mark_as_advanced(CLDNN__IOCL_ICD_USE_EXTERNAL)

# ======================================================================================================

# Include and build: Examples of usage of clDNN framework.
set(CLDNN__INCLUDE_EXAMPLES ON CACHE BOOL "Include and build: clDNN framework's usage examples.")
mark_as_advanced(CLDNN__INCLUDE_EXAMPLES)

# ======================================================================================================

# Include and build: Tests (unit tests and small acceptance tests) for clDNN framework.
set(CLDNN__INCLUDE_TESTS ON CACHE BOOL "Include and build: clDNN framework's tests.")
mark_as_advanced(CLDNN__INCLUDE_TESTS)

# ======================================================================================================

# Run (requires CLDNN__INCLUDE_TESTS to be true): Tests (unit tests and small acceptance tests) for clDNN framework.
set(CLDNN__RUN_TESTS ON CACHE BOOL "Run: clDNN framework's tests.")
mark_as_advanced(CLDNN__RUN_TESTS)

# ======================================================================================================

# Compile / Link: Allow unsafe binary size optimizations.
set(CLDNN__COMPILE_LINK_ALLOW_UNSAFE_SIZE_OPT ON CACHE BOOL "Compile / Link: Allow unsafe binary size optimizations.")
mark_as_advanced(CLDNN__COMPILE_LINK_ALLOW_UNSAFE_SIZE_OPT)

# ======================================================================================================

# CMake: Enables debug trace messages in adapter project.
set(CLDNN__CMAKE_DEBUG OFF CACHE BOOL "CMake: Enables debug trace messages in clDNN project.")
mark_as_advanced(CLDNN__CMAKE_DEBUG)

# ======================================================================================================
# ======================================================================================================
# ======================================================================================================


# Selecting Intel OpenCL SDK version and path.
if (CLDNN__IOCL_ICD_USE_EXTERNAL)
  if(NOT DEFINED ENV{INTELOCLSDKROOT})
    message(FATAL_ERROR "[clDNN] CLDNN__IOCL_ICD_USE_EXTERNAL: Usage of external Intel OpenCL SDK ICD was selected, but the INTELOCLSDKROOT environment variable cannot be found.
      Make sure that SDK is installed on the host.")
  endif()
  if(NOT EXISTS "$ENV{INTELOCLSDKROOT}/include/CL/opencl.h")
    message(FATAL_ERROR "[clDNN] CLDNN__IOCL_ICD_USE_EXTERNAL: Usage of external Intel OpenCL SDK ICD was selected, but the INTELOCLSDKROOT environment variable points to invalid directory.
      Make sure that SDK is installed on the host.")
  endif()
  set(CLDNN__IOCL_ICD_ROOT "$ENV{INTELOCLSDKROOT}")
else()
  list(FIND __CLDNN_IOclIcdVersions "${CLDNN__IOCL_ICD_VERSION}" __CLDNN_IOclIcdVersionIdx)
  if(__CLDNN_IOclIcdVersionIdx LESS 0)
    message(FATAL_ERROR "[clDNN] CLDNN__IOCL_ICD_VERSION: Selected version of Intel OpenCL SDK ICD cannot be found.
      Please use one of following values: ${__CLDNN_IOclIcdVersions}.")
  endif()
  list(GET __CLDNN_IOclIcdPaths ${__CLDNN_IOclIcdVersionIdx} CLDNN__IOCL_ICD_ROOT)
endif()

set(CLDNN__IOCL_ICD_ROOT "${CLDNN__IOCL_ICD_ROOT}" CACHE INTERNAL "Path to Intel OpenCL SDK ICD used to build clDNN framework.")

# Select SDK subdirectories with headers, binaries and libraries (based on architecture, cpu, generator and SDK type).
intel_arch_get_os(__CLDNN_TargetOs "${CLDNN__ARCHITECTURE_TARGET}")
string(TOLOWER "${__CLDNN_TargetOs}" __CLDNN_TargetOs)

intel_arch_get_cpu(__CLDNN_TargetCpu "${CLDNN__ARCHITECTURE_TARGET}")
string(TOLOWER "${__CLDNN_TargetCpu}" __CLDNN_TargetCpu)

if(CMAKE_CFG_INTDIR STREQUAL ".")
  set(__CLDNN_TargetConfig "${CMAKE_BUILD_TYPE}")
else()
  set(__CLDNN_TargetConfig "${CMAKE_CFG_INTDIR}")
endif()

if(__CLDNN_TargetCpu STREQUAL "32")
  set(__CLDNN_TargetCpuDir "x86")
elseif(__CLDNN_TargetCpu STREQUAL "64")
  set(__CLDNN_TargetCpuDir "x64")
else()
  set(__CLDNN_TargetCpuDir "${__CLDNN_TargetCpu}")
endif()

if(CLDNN__IOCL_ICD_USE_EXTERNAL)
  set(CLDNN__IOCL_ICD_INCDIRS "${CLDNN__IOCL_ICD_ROOT}/include" CACHE INTERNAL "Paths to interface headers for Intel OpenCL SDK ICD.")
  set(CLDNN__IOCL_ICD_LIBDIRS "${CLDNN__IOCL_ICD_ROOT}/lib/${__CLDNN_TargetCpuDir}" CACHE INTERNAL "Paths to static libraries for Intel OpenCL SDK ICD.")

  set(__CLDNN_FindLibSuffixes ${CMAKE_FIND_LIBRARY_SUFFIXES})
  set(CMAKE_FIND_LIBRARY_SUFFIXES "${CMAKE_SHARED_LIBRARY_SUFFIX}")
  find_library(CLDNN_FIND__IOCL_ICD_SHLPATH
    NAMES OpenCL
    PATHS "${CLDNN__IOCL_ICD_ROOT}"
          "${CLDNN__IOCL_ICD_ROOT}/bin"
          "${CLDNN__IOCL_ICD_ROOT}/bin/${__CLDNN_TargetCpuDir}"
          "${CLDNN__IOCL_ICD_ROOT}/lib"
          "${CLDNN__IOCL_ICD_ROOT}/lib/${__CLDNN_TargetCpuDir}"
    NO_CMAKE_FIND_ROOT_PATH
  )
  set(CMAKE_FIND_LIBRARY_SUFFIXES ${__CLDNN_FindLibSuffixes})

  if(CLDNN_FIND__IOCL_ICD_SHLPATH)
    get_filename_component(__CLDNN_IOclIcdSharedLibDir "${CLDNN_FIND__IOCL_ICD_SHLPATH}" DIRECTORY)
    set(CLDNN__IOCL_ICD_SHLDIRS "${__CLDNN_IOclIcdSharedLibDir}" CACHE INTERNAL "Paths to shared libraries for Intel OpenCL SDK ICD.")
  else()
    message(FATAL_ERROR "[clDNN] CLDNN__IOCL_ICD_USE_EXTERNAL: Usage of external Intel OpenCL SDK ICD was selected, but the ICD shared library cannot be located.
      Make sure that SDK is installed on the host.")
  endif()

  unset(CLDNN_FIND__IOCL_ICD_SHLPATH CACHE)
  unset(__CLDNN_FindLibSuffixes)
  unset(__CLDNN_IOclIcdSharedLibDir)
else()
  set(CLDNN__IOCL_ICD_INCDIRS "${CLDNN__IOCL_ICD_ROOT}/${__CLDNN_TargetOs}/include" CACHE INTERNAL "Paths to interface headers for Intel OpenCL SDK ICD.")
  set(CLDNN__IOCL_ICD_LIBDIRS "${CLDNN__IOCL_ICD_ROOT}/${__CLDNN_TargetOs}/${__CLDNN_TargetConfig}/lib/${__CLDNN_TargetCpuDir}" CACHE INTERNAL "Paths to static libraries for Intel OpenCL SDK ICD.")
  set(CLDNN__IOCL_ICD_SHLDIRS "${CLDNN__IOCL_ICD_ROOT}/${__CLDNN_TargetOs}/${__CLDNN_TargetConfig}/bin/${__CLDNN_TargetCpuDir}" CACHE INTERNAL "Paths to shared libraries for Intel OpenCL SDK ICD.")
endif()

unset(__CLDNN_IOclIcdVersions)
unset(__CLDNN_IOclIcdPaths)
unset(__CLDNN_IOclIcdDefaultVersion)
unset(__CLDNN_IOclIcdVersionIdx)
unset(__CLDNN_TargetOs)
unset(__CLDNN_TargetCpu)
unset(__CLDNN_TargetConfig)
unset(__CLDNN_TargetCpuDir)


# ============================================= Status =================================================

# Display status.
message(STATUS "[clDNN] ======================== ${CLDNN__PROJ_NAME} Project =======================")
message(STATUS "[clDNN] Build type:       ${CMAKE_BUILD_TYPE} (for single-configuration generators)")
message(STATUS "[clDNN] Av. build types:  ${CMAKE_CONFIGURATION_TYPES} (for multi-configuration generators)")
message(STATUS "[clDNN]")
message(STATUS "[clDNN] Output directory:")
message(STATUS "[clDNN]    \"${CLDNN__OUTPUT_DIR}\"")
message(STATUS "[clDNN] Architecture:")
message(STATUS "[clDNN]  - target: ${CLDNN__ARCHITECTURE_TARGET}    (detected: ${__CLDNN_DetectedArch_Target})")
message(STATUS "[clDNN]")
message(STATUS "[clDNN]")
message(STATUS "[clDNN] Advanced:")
if (CLDNN__IOCL_ICD_USE_EXTERNAL)
  message(STATUS "[clDNN]  - ICD version used to build:  N/A (installed externally)")
else()
  message(STATUS "[clDNN]  - ICD version used to build:  ${CLDNN__IOCL_ICD_VERSION}")
endif()
message(STATUS "[clDNN]")
message(STATUS "[clDNN]  - Include/Build examples:     ${CLDNN__INCLUDE_EXAMPLES}")
message(STATUS "[clDNN]  - Include/Build tests:        ${CLDNN__INCLUDE_TESTS}")
message(STATUS "[clDNN]")
message(STATUS "[clDNN]  - Run tests:                  ${CLDNN__RUN_TESTS}")
message(STATUS "[clDNN]")
message(STATUS "[clDNN]  - Allow unsafe size opts:     ${CLDNN__COMPILE_LINK_ALLOW_UNSAFE_SIZE_OPT}")
message(STATUS "[clDNN]  - CMake debug trace:          ${CLDNN__CMAKE_DEBUG}")
message(STATUS "[clDNN]")
message(STATUS "[clDNN]")
message(STATUS "[clDNN] ICD:")
message(STATUS "[clDNN]  - Root:           ${CLDNN__IOCL_ICD_ROOT}")
message(STATUS "[clDNN]     + Headers:     ${CLDNN__IOCL_ICD_INCDIRS}")
message(STATUS "[clDNN]     + Static libs: ${CLDNN__IOCL_ICD_LIBDIRS}")
message(STATUS "[clDNN]     + Shared libs: ${CLDNN__IOCL_ICD_SHLDIRS}")
message(STATUS "[clDNN] =============================================================================")
unset(__CLDNN_DetectedArch_Target)

return()

if(NOT(UNIX))
  set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
  set (LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
  set (CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin)
  set (LIBRARY_OUTPUT_PATH ${LIBRARY_OUTPUT_DIRECTORY}) # compatibility issue: linux uses LIBRARY_OUTPUT_PATH, windows uses LIBRARY_OUTPUT_DIRECTORY
else ()
  set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin/${CMAKE_BUILD_TYPE})
  set (LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin/${CMAKE_BUILD_TYPE})
  set (CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin/${CMAKE_BUILD_TYPE})
  set (LIBRARY_OUTPUT_PATH ${LIBRARY_OUTPUT_DIRECTORY})
endif()

message (STATUS "PROJECT ............................... " ${PROJECT_NAME})
message (STATUS "CMAKE_BINARY_DIR ...................... " ${CMAKE_BINARY_DIR})
message (STATUS "CMAKE_SOURCE_DIR ...................... " ${CMAKE_SOURCE_DIR})
message (STATUS "CMAKE_GENERATOR ....................... " ${CMAKE_GENERATOR})
message (STATUS "CMAKE_C_COMPILER_ID ................... " ${CMAKE_C_COMPILER_ID})
message (STATUS "CMAKE_CXX_COMPILER_ID ................. " ${CMAKE_CXX_COMPILER_ID})

# To Enable C++11 gcc (Linux compiler) require additional flag to be added
if(UNIX)
  CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
  CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
  if(COMPILER_SUPPORTS_CXX11)
    add_definitions(-std=c++11)
  elseif(COMPILER_SUPPORTS_CXX0X)
    add_definitions(-std=c++0x)
  elseif ((NOT(COMPILER_SUPPORTS_CXX11)) AND (NOT(COMPILER_SUPPORTS_CXX0X)))
    message (FATAL_ERROR "CMake BREAK: Compiler must support C++11")
  endif()
  if( ${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU") # GCC specific flags
    add_definitions(-mavx)
    add_definitions(-mavx2)
    add_definitions(-mfma)
    add_definitions(-ffast-math)
    if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 4.9 OR CMAKE_CXX_COMPILER_VERSION VERSION_EQUAL 4.9)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fstack-protector-strong")
    else()
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fstack-protector")
    endif()
  endif()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIE -fPIC -fno-operator-names -Wformat -Wformat-security -Wall")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -D_FORTIFY_SOURCE=2")
  
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -z noexecstack -z relro -z now")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pie -z noexecstack -z relro -z now")
else()
  if (MSVC) #MSVC specific flags
    add_definitions(/sdl)
  endif()
endif()

if(UNIX)
  CHECK_CXX_COMPILER_FLAG("-fvisibility=hidden" COMPILER_SUPPORTS_VISIBILITY)
  if (COMPILER_SUPPORTS_VISIBILITY)
    #add_definitions(-fvisibility=hidden) todo: should be hidden? if so define default visibiliti explicite for each funtion
    add_definitions(-fvisibility=default) 
    endif(COMPILER_SUPPORTS_VISIBILITY)
endif(UNIX)

include_directories(${CMAKE_SOURCE_DIR}/api)
include_directories(${CMAKE_SOURCE_DIR}/xbyak)

# Sub-directories where more CMakeLists.txt exist
add_subdirectory(src)
add_subdirectory(examples)
add_subdirectory(tests)
