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
class event_impl;
struct event
{
    struct profiling_interval_ref
    {
        string_ref name;
        uint64_t nanoseconds;
    };

    static event create_user_event(const engine& engine)
    {
        return check_status<event_impl*>("create user event failed", [&](status_t* status) { return create_user_event_impl(engine, status); });
    }
    DLL_SYM event(const event& other);
    DLL_SYM event& operator=(const event& other);
    DLL_SYM ~event();
    friend bool operator==(const event& lhs, const event& rhs) { return lhs._impl == rhs._impl; }
    friend bool operator!=(const event& lhs, const event& rhs) { return !(lhs == rhs); }

    void wait() const { check_status("wait event failed", wait_impl()); }
    void set() { check_status("set event failed", set_impl()); }

    typedef void(*event_handler)(void*);
    void set_event_handler(event_handler handler, void* param)
    {
        check_status("set event handler failed", add_event_handler_impl(handler, param));
    }

    std::vector<instrumentation::profiling_interval> get_profiling_info() const
    {
        using namespace instrumentation;
        wait();
        array_ref<profiling_interval_ref> profiling_info_ref = check_status<array_ref<profiling_interval_ref>>("network execute failed", [&](status_t* status) { return get_profiling_impl(status); });
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

    event_impl* get() const { return _impl; }
private:
    friend struct network;
    event(event_impl* impl) : _impl(impl) {}
    event_impl* _impl;
    DLL_SYM static event_impl* create_user_event_impl(const engine& engine, status_t* status) noexcept;
    DLL_SYM status_t wait_impl() const noexcept;
    DLL_SYM status_t set_impl() noexcept;
    DLL_SYM status_t add_event_handler_impl(event_handler handler, void* param) noexcept;
    DLL_SYM array_ref<profiling_interval_ref> get_profiling_impl(status_t* status) const noexcept;
};
API_CLASS(event)
}
