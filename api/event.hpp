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
#include "engine.hpp"
#include "profiling.hpp"
#include <algorithm>

namespace cldnn
{
typedef struct event_impl* cldnn_event_t;
struct event
{
    struct profiling_interval_ref
    {
        string_ref name;
        uint64_t nanoseconds;
    };

    static event create_user_event(const engine& engine)
    {
        return check_status<cldnn_event_t>("create user event failed", [&](status_t* status) { return create_user_event_impl(engine.get(), status); });
    }
    
    event(const event& other) : _impl(other._impl)
    {
        retain_event(_impl);
    }
    
    event& operator=(const event& other)
    {
        if (_impl == other._impl) return *this;
        release_event(_impl);
        _impl = other._impl;
        retain_event(_impl);
        return *this;
    }

    ~event()
    {
        release_event(_impl);
    }

    friend bool operator==(const event& lhs, const event& rhs) { return lhs._impl == rhs._impl; }
    friend bool operator!=(const event& lhs, const event& rhs) { return !(lhs == rhs); }

    void wait() const { check_status("wait event failed", wait_impl(_impl)); }
    void set() const { check_status("set event failed", set_impl(_impl)); }

    typedef void(*event_handler)(void*);
    void set_event_handler(event_handler handler, void* param) const
    {
        check_status("set event handler failed", add_event_handler_impl(_impl, handler, param));
    }

    std::vector<instrumentation::profiling_interval> get_profiling_info() const
    {
        using namespace instrumentation;
        wait();
        array_ref<profiling_interval_ref> profiling_info_ref = check_status<array_ref<profiling_interval_ref>>("network execute failed", [&](status_t* status) { return get_profiling_impl(_impl, status); });
        std::vector<profiling_interval> result(profiling_info_ref.size());
        std::transform(
            std::begin(profiling_info_ref),
            std::end(profiling_info_ref),
            std::begin(result),
            [](const profiling_interval_ref& ref) -> profiling_interval
            {
                return{
                    ref.name,
                    std::make_shared<profiling_period_basic>(std::chrono::nanoseconds(ref.nanoseconds))
                };
            }
        );
        return result;
    }

    cldnn_event_t get() const { return _impl; }
private:
    friend struct network;
    event(cldnn_event_t impl) : _impl(impl)
    {
        if (_impl == nullptr) throw std::invalid_argument("implementation pointer should not be null");
    }

    cldnn_event_t _impl;
    DLL_SYM static cldnn_event_t create_user_event_impl(cldnn_engine_t engine, status_t* status);
    DLL_SYM static void retain_event(cldnn_event_t event);
    DLL_SYM static void release_event(cldnn_event_t event);
    DLL_SYM static status_t wait_impl(cldnn_event_t event);
    DLL_SYM static status_t set_impl(cldnn_event_t event);
    DLL_SYM static status_t add_event_handler_impl(cldnn_event_t event, event_handler handler, void* param);
    DLL_SYM static array_ref<profiling_interval_ref> get_profiling_impl(cldnn_event_t event, status_t* status);
};
API_CLASS(event)
}
