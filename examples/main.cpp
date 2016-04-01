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
#include <iostream>

#ifdef _MSC_VER
namespace{
    extern "C" DLL_SYM void _cdecl nn_init();
    extern "C" DLL_SYM void _cdecl nn_exit();

    struct init_openmkl_t {
        init_openmkl_t() {
            nn_init();
        }
        ~init_openmkl_t() {
            nn_exit();
        }
    };

    init_openmkl_t init_openmkl;
}
#endif // _MSC_VER

int main( int argc, char* argv[ ] )
{
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
