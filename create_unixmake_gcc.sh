#!/bin/bash
# Copyright (c) 2016, Intel Corporation
# NeuralIA 

echo Creating echo Creating Unix Makefiles...

# Following environment variables can customize build system:
# USE_SDE_EMULATION -- Whether to use SDE emulation posible values: on, off  (lack of varable is equal to off)
# RUN_ULTS_OFFLINE --  whether to seprate ULTs execution for building process. Possible values: on, off (lack of variable is equal to off
# 
# Examples:
# RUN_ULTS_OFFLINE=on ./create_unixmax_gcc.sh       # This will make ULTs running offline

cmake -G"Unix Makefiles" -B"./build_release" --no-warn-unused-cli -DRUN_ULTS_OFFLINE:BOOL=${RUN_ULTS_OFFLINE} -DUSE_SDE_EMULATION:BOOL=${USE_SDE_EMULATION} -DCMAKE_BUILD_TYPE:STRING="Release" -DCMAKE_CONFIGURATION_TYPES:STRING="Release" -H"."
cmake -G"Unix Makefiles" -B"./build_debug" --no-warn-unused-cli -DRUN_ULTS_OFFLINE:BOOL=${RUN_ULTS_OFFLINE} -DUSE_SDE_EMULATION:BOOL=${USE_SDE_EMULATION} -DCMAKE_BUILD_TYPE:STRING="Debug" -DCMAKE_CONFIGURATION_TYPES:STRING="Debug" -H"."-G"Unix Makefiles" -B"./build_debug" --no-warn-unused-cli -DRUN_ULTS_OFFLINE:BOOL=${RUN_ULTS_OFFLINE} -DUSE_SDE_EMULATION:BOOL=${USE_SDE_EMULATION} -DCMAKE_BUILD_TYPE:STRING="Debug" -DCMAKE_CONFIGURATION_TYPES:STRING="Debug" -H"."

echo Done.
