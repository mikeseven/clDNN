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

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include <functional>
#include <string>

// exporting symbols form dynamic library
#ifdef EXPORT_NEURAL_SYMBOLS
#   if defined(_MSC_VER)
//  Microsoft
#      define DLL_SYM __declspec(dllexport)
#   elif defined(__GNUC__)
//  GCC
#      define DLL_SYM __attribute__((visibility("default")))
#   else
#      define DLL_SYM
#      pragma warning Unknown dynamic link import/export semantics.
#   endif
#else //import dll
#   if defined(_MSC_VER)
//  Microsoft
#      define DLL_SYM __declspec(dllimport)
#   elif defined(__GNUC__)
//  GCC
#      define DLL_SYM
#   else
#      define DLL_SYM
#      pragma warning Unknown dynamic link import/export semantics.
#   endif
#endif

namespace {
// There is no portable half precision floating point support.
// Using wrapped integral type with the same size and alignment restrictions.
class half_impl
{
public:
    half_impl() = default;
    explicit half_impl(uint16_t data) : _data(data) {}
    operator uint16_t() const { return _data; }

private:
    uint16_t _data;
};
}
// Use complete implementation if necessary.
#if defined HALF_HALF_HPP
typedef half half_t;
#else
typedef half_impl half_t;
#endif

#define CLDNN_SUCCESS  0
#define CLDNN_ERROR   -1
#define CLDNN_UNSUPPORTED -2

namespace cldnn {

typedef int32_t status_t;

#define CLDNN_THROW(msg, status) throw std::runtime_error(msg);

template<class T>
T create_obj(std::string err_msg, std::function<typename T::impl_type*(status_t*)> func)
{
    status_t status;
    auto impl = func(&status);
    if (!impl)
        CLDNN_THROW(err_msg, status);
    return T(impl);
}


}

