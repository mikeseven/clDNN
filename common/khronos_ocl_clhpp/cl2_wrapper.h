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


///
/// \file This file wraps cl2.hpp and disables temporary some warnings that this header can emit.
///

#ifndef CL2_WRAPPER_H_
#define CL2_WRAPPER_H_

// Check for C++.
#ifndef __cplusplus
    #error This header can be used in C++ only.
#endif

// Check for compiler and change specific diagnostics.
#if defined _MSC_VER
    #pragma warning(push)
    #pragma warning(disable: 4018)
#elif defined __GNUC__
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignore "-Wsign-compare"
#elif defined __clang__
    #pragma clang diagnostic push
    #pragma clang diagnostic ignore "-Wsign-compare"
#else
    #pragma message("Unknown compiler. No changes in diagnostics will be done.")
#endif


#include "cl2.hpp"


// Restore specific diagnostics.
#if defined _MSC_VER
    #pragma warning(pop)
#elif defined __GNUC__
    #pragma GCC diagnostic pop
#elif defined __clang__
    #pragma clang diagnostic pop
#endif

#endif // CL2_WRAPPER_H_