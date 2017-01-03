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
#include <cstdint>
#include "cldnn_defs.h"
#include "compounds.h"

namespace cldnn
{
enum class engine_types { ocl };

struct engine_configuration
{
    uint32_t enable_profiling;
    uint32_t enable_debugging;
    string_ref compiler_options;
    engine_configuration(bool profiling = false, bool debug = false, string_ref options = "")
        :enable_profiling(profiling), enable_debugging(debug), compiler_options(options) {}
};

class engine_impl;
struct engine
{
    engine(const engine_configuration& configuration = engine_configuration())
        :engine(engine_types::ocl, 0, configuration)
    {}

    engine(engine_types type, uint32_t engine_num, const engine_configuration& configuration = engine_configuration())
        :_impl(check_status<engine_impl*>("failed to create engine", [&](status_t* status)
              {
                  return create_engine_impl(type, engine_num, &configuration, status);
              }))
    {}

    DLL_SYM engine(const engine& other);
    DLL_SYM engine& operator=(const engine& other);
    DLL_SYM ~engine();
    friend bool operator==(const engine& lhs, const engine& rhs) { return lhs._impl == rhs._impl; }
    friend bool operator!=(const engine& lhs, const engine& rhs) { return !(lhs == rhs); }

    static uint32_t engine_count(engine_types type)
    {
        status_t status;
        auto result = engine_count_impl(type, &status);
        if (status != CLDNN_SUCCESS)
            CLDNN_THROW("engine_count failed", status);
        return result;
    }

    DLL_SYM engine_types engine_type() noexcept;

    engine_impl* get() const { return _impl; }

private:
    friend struct network;
    friend struct memory;
    friend struct event;
    DLL_SYM engine(engine_impl* impl) : _impl(impl)
    {
        if (_impl == nullptr) throw std::invalid_argument("implementation pointer should not be null");
    }
    engine_impl* _impl;
    DLL_SYM static uint32_t engine_count_impl(engine_types type, status_t* status) noexcept;
    DLL_SYM static engine_impl* create_engine_impl(engine_types type, uint32_t engine_num, const engine_configuration* configuration, status_t* status) noexcept;
};
API_CLASS(engine)
}
