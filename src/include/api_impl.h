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
#include "api/cldnn.h"
#include <functional>

#define API_CAST(api_type, impl_type) \
inline api_type api_cast(impl_type* value) { return reinterpret_cast<api_type>(value); } \
inline impl_type* api_cast(api_type value) { return reinterpret_cast<impl_type*>(value); }


template<typename T>
T exception_handler(cldnn_status default_error, cldnn_status* status, const T& default_result, std::function<T()> func)
{
    //NOTE for implementer: status should not be modified after successful func() call
    try
    {
        if (status)
            *status = CLDNN_SUCCESS;
        return func();
    }
    catch (...)
    {
        if (status)
            *status = default_error;
#ifndef NDEBUG
        static_cast<void>(default_result);
        throw;
#else
        return default_result;
#endif
    }
}

inline void exception_handler(cldnn_status default_error, cldnn_status* status, std::function<void()> func)
{
    //NOTE for implementer: status should not be modified after successful func() call
    try
    {
        if (status)
            *status = CLDNN_SUCCESS;
        func();
    }
    catch (...)
    {
        if (status)
            *status = default_error;
#ifndef NDEBUG
        throw;
#endif
    }
}
