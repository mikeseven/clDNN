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
#include "cldnn.h"

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

namespace cldnn {

using status_t = ::cldnn_status;

#define CLDNN_THROW(msg, status) throw std::runtime_error(msg);

template<class T>
T check_status(std::string err_msg, std::function<T(status_t*)> func)
{
    status_t status;
    auto result = func(&status);
    if (status != CLDNN_SUCCESS)
        CLDNN_THROW(err_msg, status);
    return result;
}

template<>
inline void check_status<void>(std::string err_msg, std::function<void(status_t*)> func)
{
    status_t status;
    func(&status);
    if (status != CLDNN_SUCCESS)
        CLDNN_THROW(err_msg, status);
}

#define CLDNN_API_CLASS(the_class) static_assert(std::is_standard_layout<the_class>::value, #the_class " has to be 'standart layout' class");

}

template<typename T>
typename std::enable_if<std::is_integral<T>::value, T>::type align_to(T size, size_t align) {
    return static_cast<T>((size % align == 0) ? size : size - size % align + align);
}

template<typename T>
typename std::enable_if<std::is_integral<T>::value, T>::type pad_to(T size, size_t align) {
    return static_cast<T>((size % align == 0) ? 0 : align - size % align);
}

template<typename T>
typename std::enable_if<std::is_integral<T>::value, bool>::type is_aligned_to(T size, size_t align)
{
    return !(size % align);
}
