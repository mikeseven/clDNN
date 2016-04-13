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

#!/bin/bash

BUILD_DIR=${1:-"UnixMk"}

echo Creating echo Creating Unix Makefiles...

# Following environment variables can customize build system:
# USE_SDE_EMULATION -- Whether to use SDE emulation posible values: on, off  (lack of varable is equal to off)
# RUN_ULTS_OFFLINE --  whether to seprate ULTs execution for building process. Possible values: on, off (lack of variable is equal to off
# 
# Examples:
# RUN_ULTS_OFFLINE=on ./create_unixmake_gcc.sh       # This will make ULTs running offline
# WITH_CAFFE=on ./create_unixmake_gcc.sh             # This will build caffe tests

cmake -G"Unix Makefiles" -B"./${BUILD_DIR}/Release" --no-warn-unused-cli -DRUN_ULTS_OFFLINE:BOOL=${RUN_ULTS_OFFLINE} -DWITH_CAFFE:BOOL=${WITH_CAFFE} -DUSE_SDE_EMULATION:BOOL=${USE_SDE_EMULATION} -DCMAKE_BUILD_TYPE:STRING="Release" -DCMAKE_CONFIGURATION_TYPES:STRING="Release" -H"."
cmake -G"Unix Makefiles" -B"./${BUILD_DIR}/Debug" --no-warn-unused-cli -DRUN_ULTS_OFFLINE:BOOL=${RUN_ULTS_OFFLINE} -DWITH_CAFFE:BOOL=${WITH_CAFFE} -DUSE_SDE_EMULATION:BOOL=${USE_SDE_EMULATION} -DCMAKE_BUILD_TYPE:STRING="Debug" -DCMAKE_CONFIGURATION_TYPES:STRING="Debug" -H"."-G"Unix Makefiles" -B"./${BUILD_DIR}/Debug" --no-warn-unused-cli -DRUN_ULTS_OFFLINE:BOOL=${RUN_ULTS_OFFLINE} -DUSE_SDE_EMULATION:BOOL=${USE_SDE_EMULATION} -DCMAKE_BUILD_TYPE:STRING="Debug" -DCMAKE_CONFIGURATION_TYPES:STRING="Debug" -H"."

echo Done.
