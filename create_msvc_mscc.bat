@REM Copyright (c) 2016, Intel Corporation
@REM NeuralIA 

@echo off
echo Creating Microsoft Visual Studio 14 2015 Win64 files...

cmake -G"Visual Studio 14 2015 Win64" -B".\MSVC\Debug" --no-warn-unused-cli -DCMAKE_BUILD_TYPE:STRING="Debug" -DCMAKE_CONFIGURATION_TYPES:STRING="Debug" -DCMAKE_SUPPRESS_REGENERATION:BOOL=ON -H"."
cmake -G"Visual Studio 14 2015 Win64" -B".\MSVC\Release" --no-warn-unused-cli -DCMAKE_BUILD_TYPE:STRING="Release" -DCMAKE_CONFIGURATION_TYPES:STRING="Release" -DCMAKE_SUPPRESS_REGENERATION:BOOL=ON -H"."
echo Done.
pause
