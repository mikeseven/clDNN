@REM Copyright (c) 2016 Intel Corporation

@REM Licensed under the Apache License, Version 2.0 (the "License");
@REM you may not use this file except in compliance with the License.
@REM You may obtain a copy of the License at

@REM      http://www.apache.org/licenses/LICENSE-2.0

@REM Unless required by applicable law or agreed to in writing, software
@REM distributed under the License is distributed on an "AS IS" BASIS,
@REM WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
@REM See the License for the specific language governing permissions and
@REM limitations under the License.

@echo off
echo Creating Microsoft Visual Studio 14 2015 Win64 files...

cmake -G"Visual Studio 14 2015 Win64" -B".\MSVC\Debug" --no-warn-unused-cli -DCMAKE_BUILD_TYPE:STRING="Debug" -DCMAKE_CONFIGURATION_TYPES:STRING="Debug" -DCMAKE_SUPPRESS_REGENERATION:BOOL=ON -H"."
cmake -G"Visual Studio 14 2015 Win64" -B".\MSVC\Release" --no-warn-unused-cli -DCMAKE_BUILD_TYPE:STRING="Release" -DCMAKE_CONFIGURATION_TYPES:STRING="Release" -DCMAKE_SUPPRESS_REGENERATION:BOOL=ON -H"."
echo Done.
pause
