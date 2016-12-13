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

#ifdef _MSC_VER

#include <new>
#include <iostream>

#include "api/cldnn_defs.h"

using void_fun_ptr = void (__cdecl *)();

#pragma warning(disable : 4075)

#pragma section(".nn_init$a", read, write)
__declspec(allocate(".nn_init$a")) void_fun_ptr nn_cl_seg_start = reinterpret_cast<void_fun_ptr>(1);
#pragma section(".nn_init$b", read, write)
__declspec(allocate(".nn_init$b")) void_fun_ptr nn_cl_seg_zero = nullptr;
#pragma section(".nn_init$z", read, write)
__declspec(allocate(".nn_init$z")) void_fun_ptr nn_cl_seg_end = reinterpret_cast<void_fun_ptr>(1);

int nn_init_seg_add(void_fun_ptr fp) {
    static auto ptr = &nn_cl_seg_start;
    ptr[0] = fp;
    ptr[1] = nullptr;
    ++ptr;
    return 0;
}

#pragma init_seg(".nn_init$m", nn_init_seg_add)

extern "C" DLL_SYM void _cdecl nn_init() {
    for (auto fp=(&nn_cl_seg_start)+1; fp<&nn_cl_seg_end; ++fp) if (*fp) (*fp)();
}
extern "C" DLL_SYM void _cdecl nn_exit() {
    for (auto fp=&nn_cl_seg_start; *fp; ++fp) (*fp)();
}

#endif // _MSC_VER
