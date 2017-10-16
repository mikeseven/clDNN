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

if(CLDNN__INCLUDE_GENERIC_SAMPLE)
  message(STATUS "[clDNN] InferenceEngine Submodule start =============================================")

  # Keep compiler flags before changing
  set(KEEP_OLD_CXX_FLAGS ${CMAKE_CXX_FLAGS})
  set(KEEP_OLD_C_FLAGS ${CMAKE_C_FLAGS})

  # Reset the build flags to the initial set
  set(CMAKE_CXX_FLAGS ${INITIAL_CXX_FLAGS})
  set(CMAKE_C_FLAGS ${INITIAL_C_FLAGS})
  if(WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Zi")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Zi")
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
    set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -fPIC")
  endif()
  
  set(IE_SUBMODULE_IN_CLDNN ON)

  # Remove some parts from the IE build
  set(MKL_DNN_ON_WIN32 OFF)
  set(ENABLE_MKL_DNN OFF)
  set(ENABLE_MYRIAD OFF)
  set(ENABLE_DLIA OFF)
  set(ENABLE_OPENVX OFF)
  set(ENABLE_GNA OFF)
  set(ENABLE_TESTS OFF)
  set(ENABLE_PERFORMANCE_TESTS OFF)
  set(ENABLE_OMP OFF)
  set(ENABLE_INTEL_OMP OFF)
  set(BUILD_VALIDATION_APP OFF)
  set(OS_FOLDER ON CACHE BOOL "os folder" FORCE)

  add_subdirectory(dpd_vcp_dl-scoring_engine)
  add_subdirectory(generic_sample)

  # Restore the old compiler flags
  set(CMAKE_CXX_FLAGS ${KEEP_OLD_CXX_FLAGS})
  set(CMAKE_C_FLAGS ${KEEP_OLD_C_FLAGS})

  message(STATUS "[clDNN] InferenceEngine Submodule end ===============================================")
endif()
