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
#include "ocl_toolkit.h"
#include "ocl_base_event.h"
#include "ocl_user_event.h"

#include <cassert>

namespace cldnn { namespace gpu {

cl_device_type convert_configuration_device_type(configuration::device_types device_type)
{
    cl_device_type device_types[] = {
            CL_DEVICE_TYPE_DEFAULT,
            CL_DEVICE_TYPE_CPU,
            CL_DEVICE_TYPE_GPU,
            CL_DEVICE_TYPE_ACCELERATOR };
    return device_types[device_type];
}

cl::Device get_gpu_device(const configuration& config)
{
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    cl::Device default_device;
    for (auto& p : platforms)
    {
        std::vector<cl::Device> devices;
        p.getDevices(CL_DEVICE_TYPE_ALL, &devices);
        for (auto& d : devices)
        {
            if (d.getInfo<CL_DEVICE_TYPE>() == convert_configuration_device_type(config.device_type))
            {
                auto vendor_id = d.getInfo<CL_DEVICE_VENDOR_ID>();
                //set Intel GPU device default
                if (vendor_id == config.device_vendor)
                    return d;
            }
        }
    }
    throw std::runtime_error("No OpenCL GPU device found.");
}

gpu_toolkit::gpu_toolkit(const configuration& config) 
    : _configuration(config)
    , _device(get_gpu_device(config))
    , _context(_device)
    , _command_queue(_context,
                     _device,
                     (config.enable_profiling
                        ? cl::QueueProperties::Profiling
                        : cl::QueueProperties::None) | 
                     (config.host_out_of_order
                        ? cl::QueueProperties::OutOfOrder
                        : cl::QueueProperties::None))
    , _engine_info(*this)
    , _kernels_cache(*this)
    {
        _device.getInfo(CL_DEVICE_EXTENSIONS, &extensions);
    }

event_impl::ptr gpu_toolkit::enqueue_kernel(cl::Kernel const& kern, cl::NDRange const& global, cl::NDRange const& local, std::vector<event_impl::ptr> const & deps)
{
    std::vector<cl::Event> dep_events;
    auto dep_events_ptr = &dep_events;
    if (!_configuration.host_out_of_order)
    {
        for (auto& dep : deps)
            dep_events.push_back(dynamic_cast<base_event*>(dep.get())->get());
    }
    else
    {
        dep_events_ptr = nullptr;
        sync_events(deps);
    }

    cl::Event ret_ev;
    _command_queue.enqueueNDRangeKernel(kern, cl::NullRange, global, local, dep_events_ptr, &ret_ev);
    return{ new base_event(ret_ev, ++_queue_counter), false };
}

event_impl::ptr gpu_toolkit::enqueue_marker(std::vector<event_impl::ptr> const& deps)
{
    if (deps.empty())
        return{ new user_event(cl::UserEvent(_context), true), false };

    if (!_configuration.host_out_of_order)
    {
        cl::Event ret_ev;
        if (!enabled_single_kernel())
        {
            std::vector<cl::Event> dep_events;
            for (auto& dep : deps)
                dep_events.push_back(dynamic_cast<base_event*>(dep.get())->get());

            _command_queue.enqueueMarkerWithWaitList(&dep_events, &ret_ev);
        }
        else
            _command_queue.enqueueMarkerWithWaitList(nullptr, &ret_ev);

        return{ new base_event(ret_ev), false };
    }
    else
    {
        sync_events(deps);
        assert(_last_barrier_ev() != nullptr);
        return{ new base_event(_last_barrier_ev), false };
    }
}

void gpu_toolkit::wait_for_events(std::vector<event_impl::ptr> const & events)
{
    std::vector<cl::Event> clevents;
    for (auto& ev : events)
        clevents.push_back(dynamic_cast<base_event*>(ev.get())->get());

    cl::WaitForEvents(clevents);
}

void gpu_toolkit::sync_events(std::vector<event_impl::ptr> const & deps)
{
    if (!_configuration.host_out_of_order)
        return;

    bool needs_barrier = false;
    for (auto& dep : deps)
    {
        auto* ocl_ev = dynamic_cast<base_event*>(dep.get());
        if (ocl_ev->get_queue_stamp() > _last_barrier)
            needs_barrier = true;
    }

    if (needs_barrier)
    {
        _command_queue.enqueueBarrierWithWaitList(nullptr, &_last_barrier_ev);
        _last_barrier = _queue_counter;
    }
}

}}
