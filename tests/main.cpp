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

#include "gtest/gtest.h"
#include "dll.h"

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
    int result;
    {
        ::testing::InitGoogleTest(&argc, argv);
#if _WIN32
        // Safety cleanup.
        system("where /q umdh && del pre_cpu.txt");
        system("where /q umdh && del post_cpu.txt");
        system("where /q umdh && del memdiff_cpu.txt");

        // Get first snapshot.
        system("where /q umdh && umdh -pn:tests.exe -f:pre_cpu.txt");
#endif

        result = RUN_ALL_TESTS();
    }

#if _WIN32
    // Get second snapshot.
    system("where /q umdh && umdh -pn:tests.exe -f:post_cpu.txt");

    // Prepare memory diff.
    system("where /q umdh && umdh pre_cpu.txt post_cpu.txt -f:memdiff_cpu.txt");

    // Cleanup.
    system("where /q umdh && del pre_cpu.txt");
    system("where /q umdh && del post_cpu.txt");
#endif

    return result;
}