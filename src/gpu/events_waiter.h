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
#include "api/CPP/profiling.hpp"
#include "ocl_user_event.h"
#include "ocl_toolkit.h"

namespace cldnn { namespace gpu {
class events_waiter : public context_holder
{
public:
    explicit events_waiter(std::shared_ptr<gpu_toolkit> context) : context_holder(context)
    {}

    event_impl::ptr run(const std::vector<event_impl::ptr>& dependencies)
    {
        if (dependencies.size() == 0)
        {
            auto ev = new gpu::user_event(cl::UserEvent(context()->context()), true);
            return{ ev, false };
        }

        return context()->enqueue_marker(dependencies);
    }
};
}}