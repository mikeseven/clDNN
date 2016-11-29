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
class event_impl;
struct event
{
    typedef event_impl impl_type;
    DLL_SYM event(const event& other);
    DLL_SYM event& operator=(const event& other);
    DLL_SYM ~event();
    friend bool operator==(const event& lhs, const event& rhs) { return lhs._impl == rhs._impl; }
    friend bool operator!=(const event& lhs, const event& rhs) { return !(lhs == rhs); }

    DLL_SYM void wait();
    DLL_SYM void reset();
    DLL_SYM void set();
    typedef void(*event_handler)(void*);
    DLL_SYM void on_event(event_handler handler, void* param);

    event_impl* implementation() const { return _impl; }
private:
    friend struct network;
    event(event_impl* impl) : _impl(impl) {}
    event_impl* _impl;
};
}
