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
#include "cldnn_defs.h"

namespace cldnn
{

enum class engine_types : int32_t
{
    ocl = cldnn_engine_ocl
};

struct engine_configuration
{
    const bool enable_profiling;
    const std::string compiler_options;
    engine_configuration(bool profiling = false, const std::string& options = std::string())
        :enable_profiling(profiling), compiler_options(options) {}

    engine_configuration(const cldnn_engine_configuration& c_conf)
        :enable_profiling(c_conf.enable_profiling != 0), compiler_options(c_conf.compiler_options){}

    operator ::cldnn_engine_configuration() const
    {
        return{ enable_profiling, compiler_options.c_str() };
    }
};

using engine_info = ::cldnn_engine_info;

struct engine
{
    engine(const engine_configuration& configuration = engine_configuration())
        :engine(engine_types::ocl, 0, configuration)
    {}

    engine(engine_types type, uint32_t engine_num, const engine_configuration& configuration = engine_configuration())
        :_impl(check_status<::cldnn_engine>("failed to create engine", [&](status_t* status)
              {
                  cldnn_engine_configuration conf = configuration;
                  return cldnn_create_engine(static_cast<int32_t>(type), engine_num, &conf, status);
              }))
    {}

    // TODO add move construction/assignment
    engine(const engine& other) :_impl(other._impl)
    {
        retain();
    }

    engine& operator=(const engine& other)
    {
        if (_impl == other._impl) return *this;
        release();
        _impl = other._impl;
        retain();
        return *this;
    }

    ~engine()
    {
        release();
    }

    friend bool operator==(const engine& lhs, const engine& rhs) { return lhs._impl == rhs._impl; }
    friend bool operator!=(const engine& lhs, const engine& rhs) { return !(lhs == rhs); }

    static uint32_t engine_count(engine_types type)
    {
        return check_status<uint32_t>("engine_count failed", [=](status_t* status)
        {
            return cldnn_get_engine_count(static_cast<int32_t>(type), status);
        });
    }

    engine_info get_info() const
    {
        return check_status<engine_info>("engine_count failed", [=](status_t* status)
        {
            return cldnn_get_engine_info(_impl, status);
        });
    }

    engine_types get_type() const
    {
        return check_status<engine_types>("engine_count failed", [=](status_t* status)
        {
            return static_cast<engine_types>(cldnn_get_engine_type(_impl, status));
        });
    }

    ::cldnn_engine get() const { return _impl; }

private:
    friend struct network;
    friend struct memory;
    friend struct event;
    engine(::cldnn_engine impl) : _impl(impl)
    {
        if (_impl == nullptr) throw std::invalid_argument("implementation pointer should not be null");
    }
    ::cldnn_engine _impl;

    void retain()
    {
        check_status<void>("retain engine failed", [=](status_t* status) { cldnn_retain_engine(_impl, status); });
    }
    void release()
    {
        check_status<void>("release engine failed", [=](status_t* status) { cldnn_release_engine(_impl, status); });
    }
};
CLDNN_API_CLASS(engine)
}
