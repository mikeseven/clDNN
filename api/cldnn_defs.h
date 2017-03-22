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

/*! @mainpage clDNN Documentation
* @section intro Introduction
* Compute Library for Deep Neural Networks (clDNN) is a middle-ware software
*  for accelerating DNN inference on Intel &reg; HD and Iris&trade; Pro Graphics.
* @section example Example MNIST network
* @include example_cldnn.cpp
*/

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <functional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

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

/// @addtogroup cpp_api C++ API
/// @{

/// @defgroup cpp_error Error Handling
/// @{

using status_t = ::cldnn_status;

/// @brief clDNN specific exception type.
class error : public std::runtime_error
{
public:
    explicit error(const std::string& _Message, status_t status = CLDNN_ERROR)
        : runtime_error(_Message)
        , _status(status)
    {
    }

    explicit error(const char* _Message, status_t status = CLDNN_ERROR)
        : runtime_error(_Message)
        , _status(status)
    {
    }

    /// @brief Returns clDNN status code.
    const status_t& status() const { return _status; }
private:
    status_t _status;
};

#define CLDNN_THROW(msg, status) throw cldnn::error(msg, status);

template<class T>
T check_status(std::string err_msg, std::function<T(status_t*)> func)
{
    status_t status = CLDNN_SUCCESS;
    auto result = func(&status);
    
    if (status != CLDNN_SUCCESS)
        CLDNN_THROW(err_msg, status);
    return result;
}

template<>
inline void check_status<void>(std::string err_msg, std::function<void(status_t*)> func)
{
    status_t status = CLDNN_SUCCESS;
    func(&status);
    if (status != CLDNN_SUCCESS)
        CLDNN_THROW(err_msg, status);
}

/// @}

/// @cond CPP_HELPERS

/// @defgroup cpp_helpers Helpers
/// @{

#define CLDNN_API_CLASS(the_class) static_assert(std::is_standard_layout<the_class>::value, #the_class " has to be 'standart layout' class");


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

/// Computes ceil(@p val / @p divider) on unsigned integral numbers.
///
/// Computes division of unsigned integral numbers and rounds result up to full number (ceiling).
/// The function works for unsigned integrals only. Signed integrals are converted to corresponding
/// unsigned ones.
///
/// @tparam T1   Type of @p val. Type must be integral (SFINAE).
/// @tparam T2   Type of @p divider. Type must be integral (SFINAE).
///
/// @param val       Divided value. If value is signed, it will be converted to corresponding unsigned type.
/// @param divider   Divider value. If value is signed, it will be converted to corresponding unsigned type.
///
/// @return   Result of ceil(@p val / @p divider). The type of result is determined as if in normal integral
///           division, except each operand is converted to unsigned type if necessary.
template <typename T1, typename T2>
constexpr auto ceil_div(T1 val, T2 divider)
    -> typename std::enable_if<std::is_integral<T1>::value && std::is_integral<T2>::value,
                               decltype(std::declval<typename std::make_unsigned<T1>::type>() / std::declval<typename std::make_unsigned<T2>::type>())>::type
{
    typedef typename std::make_unsigned<T1>::type UT1;
    typedef typename std::make_unsigned<T2>::type UT2;
    typedef decltype(std::declval<UT1>() / std::declval<UT2>()) RetT;

    return static_cast<RetT>((static_cast<UT1>(val) + static_cast<UT2>(divider) - 1U) / static_cast<UT2>(divider));
}

/// Rounds @p val to nearest multiply of @p rounding that is greater or equal to @p val.
///
/// The function works for unsigned integrals only. Signed integrals are converted to corresponding
/// unsigned ones.
///
/// @tparam T1       Type of @p val. Type must be integral (SFINAE).
/// @tparam T2       Type of @p rounding. Type must be integral (SFINAE).
///
/// @param val        Value to round up. If value is signed, it will be converted to corresponding unsigned type.
/// @param rounding   Rounding value. If value is signed, it will be converted to corresponding unsigned type.
///
/// @return   @p val rounded up to nearest multiply of @p rounding. The type of result is determined as if in normal integral
///           division, except each operand is converted to unsigned type if necessary.
template <typename T1, typename T2>
constexpr auto round_up_to(T1 val, T2 rounding)
    -> typename std::enable_if<std::is_integral<T1>::value && std::is_integral<T2>::value,
                               decltype(std::declval<typename std::make_unsigned<T1>::type>() / std::declval<typename std::make_unsigned<T2>::type>())>::type
{
    typedef typename std::make_unsigned<T1>::type UT1;
    typedef typename std::make_unsigned<T2>::type UT2;
    typedef decltype(std::declval<UT1>() / std::declval<UT2>()) RetT;

    return static_cast<RetT>(ceil_div(val, rounding) * static_cast<UT2>(rounding));
}

///
/// \brief Converts C API float array to std::vector<float>
///
inline std::vector<float> float_arr_to_vector(const cldnn_float_arr& arr)
{
    std::vector<float> result(arr.size);
    for (size_t i = 0; i < arr.size; i++)
    {
        result[i] = arr.data[i];
    }
    return result;
}


///
/// \brief Converts std::vector<float> to C API float_array
///
inline cldnn_float_arr float_vector_to_arr(const std::vector<float>& stor)
{
    return { stor.data(), stor.size() };
}

/// @}

/// @endcond

/// @}
}
