# Copyright (c) 2018 Intel Corporation

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#      http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# =================================== Custom pre- and post-steps =======================================

set(__CLDNN_FindLibSuffixes ${CMAKE_FIND_LIBRARY_SUFFIXES})
set(CMAKE_FIND_LIBRARY_SUFFIXES "${CMAKE_SHARED_LIBRARY_SUFFIX}")
find_library(CLDNN_FIND__FREEIMAGE_SHLPATH
  NAMES FreeImage
  PATHS ${CLDNN__FREEIMAGE_LIBDIRS}
  NO_CMAKE_FIND_ROOT_PATH
)
set(CMAKE_FIND_LIBRARY_SUFFIXES ${__CLDNN_FindLibSuffixes})

# =================================== Link targets and dependencies ====================================

add_subdirectory(common)
add_subdirectory(examples)
add_subdirectory(examples_training)
add_subdirectory(microbench)

# =================================== Custom pre- and post-steps =======================================

unset(CLDNN_FIND__FREEIMAGE_SHLPATH CACHE)
unset(__CLDNN_FindLibSuffixes)

# ======================================================================================================
