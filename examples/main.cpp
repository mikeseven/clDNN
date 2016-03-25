/*
// Copyright (c) 2016 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

#include "api/neural.h"

#if defined _WIN32
#   include "os_windows.h"
#else
#   include "os_linux.h"
#endif

#include <iostream>

#include "convolution.h" //todo remove

int main( int argc, char* argv[ ] )
{
    // RAII for loading library, device initialization and opening interface 0
    scoped_library library("neuralIA"+dynamic_library_extension);
    extern void example_convolution_forward();

    try{
        example_convolution_forward();
    } catch (std::exception &e) {
        std::cerr << e.what();
    } catch(...) {
        std::cerr << "Unknown exceptions.";
    }

    return 0;
}
