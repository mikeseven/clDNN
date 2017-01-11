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
#include "api/profiling.hpp"
#include "event_impl.h"

namespace neural { namespace gpu {
class events_waiter: public context_holder
{
public:
    explicit events_waiter(std::shared_ptr<gpu_toolkit> context) : context_holder(context){}
    cldnn::refcounted_obj_ptr<cldnn::event_impl> run(const std::vector<cldnn::refcounted_obj_ptr<cldnn::event_impl>>& dependencies)
    {
        if(dependencies.size() == 0)
        {
            cldnn::refcounted_obj_ptr<cldnn::event_impl> result(new cldnn::user_event_gpu(cl::UserEvent( context()->context() )), false);
            result->set();
            return result;
        }
        cl::Event end_event;
        std::vector<cl::Event> events;
        events.reserve(dependencies.size());
        for(auto& dependency : dependencies)
        {
            events.emplace_back(dependency->get());
        }

        if (context()->get_configuration().enable_profiling) {
            instrumentation::timer<> pre_enqueue_timer;
            auto pre_enqueue_time = pre_enqueue_timer.uptime();
            //TODO cl::CommandQueue::enqueueMarkerWithWaitList() should be const
            const_cast<cl::CommandQueue&>(context()->queue()).enqueueMarkerWithWaitList(&events, &end_event);
            end_event.wait();
            context()->report_profiling({ "events_waiter",
            {
                { "pre-enqueue", std::make_shared<instrumentation::profiling_period_basic>(pre_enqueue_time) },
                { "submission",  std::make_shared<profiling_period_event>(end_event, CL_PROFILING_COMMAND_QUEUED, CL_PROFILING_COMMAND_SUBMIT) },
                { "starting",    std::make_shared<profiling_period_event>(end_event, CL_PROFILING_COMMAND_SUBMIT, CL_PROFILING_COMMAND_START) },
                { "executing",   std::make_shared<profiling_period_event>(end_event, CL_PROFILING_COMMAND_START,  CL_PROFILING_COMMAND_END) }
            } });
        }
        else {
            const_cast<cl::CommandQueue&>(context()->queue()).enqueueMarkerWithWaitList(&events, &end_event);
        }
        return new cldnn::event_impl(end_event);
    }
};
}}