﻿/*
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

// we want exceptions
#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120
#include <cl2_wrapper.h>
#include <memory>
#include <chrono>
#include "api/CPP/profiling.hpp"
#include "kernels_cache.h"
#include "engine_info.h"
#include "event_impl.h"

namespace cldnn { namespace gpu {

struct configuration
{
    enum device_types { default_device = 0, cpu, gpu, accelerator };

    configuration();

    bool enable_profiling;
    bool meaningful_kernels_names;
    bool dump_custom_program;
    device_types device_type;
    uint32_t device_vendor;
    std::string compiler_options;
    std::string single_kernel_name;
    bool host_out_of_order;
};

class gpu_toolkit;

class context_holder
{
protected:
    context_holder(std::shared_ptr<gpu_toolkit> context) : _context(context) {}
    virtual ~context_holder() = default;

    const std::shared_ptr<gpu_toolkit>& context() const { return _context; }

    std::shared_ptr<gpu_toolkit> _context;

};

struct profiling_period_event : instrumentation::profiling_period
{
    profiling_period_event(const cl::Event& event, cl_profiling_info start, cl_profiling_info end)
        : _event(event)
        , _start(start)
        , _end(end)
    {}

    std::chrono::nanoseconds value() const override
    {
        cl_ulong start_nanoseconds;
        _event.getProfilingInfo(_start, &start_nanoseconds);
        cl_ulong end_nanoseconds;
        _event.getProfilingInfo(_end, &end_nanoseconds);
        return std::chrono::nanoseconds(static_cast<long long>(end_nanoseconds - start_nanoseconds));
    }

private:
    cl::Event _event;
    cl_profiling_info _start;
    cl_profiling_info _end;
};

class gpu_toolkit
{
    std::string extensions;
    friend class context_holder;

public:
    gpu_toolkit(const configuration& configuration = configuration());

    const configuration& get_configuration() const { return _configuration; }
    const cl::Device& device() const { return _device; }
    const cl::Context& context() const { return _context; }
    const cl::CommandQueue& queue() const { return _command_queue; }

    kernels_cache& get_kernels_cache() { return _kernels_cache; }

    engine_info_internal get_engine_info() const { return _engine_info; }
    inline bool extension_supported(const std::string ext) { return extensions.find(ext) != std::string::npos; }

    gpu_toolkit(const gpu_toolkit& other) = delete;
    gpu_toolkit(gpu_toolkit&& other) = delete;
    gpu_toolkit& operator=(const gpu_toolkit& other) = delete;
    gpu_toolkit& operator=(gpu_toolkit&& other) = delete;
    std::string single_kernel_name() const { return _configuration.single_kernel_name; }
    bool enabled_single_kernel() const { return single_kernel_name() == "" ? false : true; }

    event_impl::ptr enqueue_kernel(cl::Kernel const& kern, cl::NDRange const& global, cl::NDRange const& local, std::vector<event_impl::ptr> const& deps);
    event_impl::ptr enqueue_marker(std::vector<event_impl::ptr> const& deps);
    void wait_for_events(std::vector<event_impl::ptr> const& events);

private:
    configuration _configuration;
    cl::Device _device;
    cl::Context _context;
    cl::CommandQueue _command_queue;
    engine_info_internal _engine_info;
    kernels_cache _kernels_cache;

    uint64_t _queue_counter = 0;
    uint64_t _last_barrier = 0;
    cl::Event _last_barrier_ev;

    //returns whether a barrier has been added
    void sync_events(std::vector<event_impl::ptr> const& deps);
};

}}
