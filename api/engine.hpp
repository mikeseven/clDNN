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
    string_ref compiler_options;
    engine_configuration(bool profiling = false, string_ref options = "")
        :enable_profiling(profiling), compiler_options(options) {}
};

struct engine_info
{
    enum configurations : uint32_t
    {
        GT0 = 0,
        GT1,
        GT1_5,
        GT2,
        GT3,
        GT4,
        GT_UNKNOWN,
        GT_COUNT
    };

    configurations configuration;
    uint32_t cores_count;
    uint32_t core_frequency;

    uint64_t max_work_group_size;
    uint64_t max_local_mem_size;

    // Flags (for layout compatibility fixed size types are used).
    uint8_t supports_fp16;
    uint8_t supports_fp16_denorms;
};

typedef struct engine_impl* cldnn_engine_t;
struct engine
{
    engine(const engine_configuration& configuration = engine_configuration())
        :engine(engine_types::ocl, 0, configuration)
    {}

    engine(engine_types type, uint32_t engine_num, const engine_configuration& configuration = engine_configuration())
        :_impl(check_status<cldnn_engine_t>("failed to create engine", [&](status_t* status)
              {
                  return create_engine_impl(type, engine_num, &configuration, status);
              }))
    {}

    engine(const engine& other) :_impl(other._impl)
    {
        retain_engine(_impl);
    }

    engine& operator=(const engine& other)
    {
        if (_impl == other._impl) return *this;
        release_engine(_impl);
        _impl = other._impl;
        retain_engine(_impl);
        return *this;
    }

    ~engine()
    {
        release_engine(_impl);
    }

    friend bool operator==(const engine& lhs, const engine& rhs) { return lhs._impl == rhs._impl; }
    friend bool operator!=(const engine& lhs, const engine& rhs) { return !(lhs == rhs); }

    static uint32_t engine_count(engine_types type)
    {
        return check_status<uint32_t>("engine_count failed", [=](status_t* status)
        {
            return engine_count_impl(type, status);
        });
    }

    engine_info get_info() const
    {
        engine_info result;
        check_status("get engine info failed", get_info_impl(_impl, &result));
        return result;
    }

    engine_types engine_type() const { return get_engine_type(_impl); }

    cldnn_engine_t get() const { return _impl; }

private:
    friend struct network;
    friend struct memory;
    friend struct event;
    engine(cldnn_engine_t impl) : _impl(impl)
    {
        if (_impl == nullptr) throw std::invalid_argument("implementation pointer should not be null");
    }
    cldnn_engine_t _impl;
    DLL_SYM static uint32_t engine_count_impl(engine_types type, status_t* status);
    DLL_SYM static cldnn_engine_t create_engine_impl(engine_types type, uint32_t engine_num, const engine_configuration* configuration, status_t* status);
    DLL_SYM static void retain_engine(cldnn_engine_t engine);
    DLL_SYM static void release_engine(cldnn_engine_t engine);
    DLL_SYM static status_t get_info_impl(cldnn_engine_t engine, engine_info* info);
    DLL_SYM static engine_types get_engine_type(cldnn_engine_t engine);
};
API_CLASS(engine)
}
