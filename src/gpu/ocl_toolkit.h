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

// we want exceptions
#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120
#include <cl2_wrapper.h>
#include <memory>
#include <chrono>
#include "api/instrumentation.h"
#include "kernels_cache.h"

namespace neural { namespace gpu {
class gpu_toolkit;

class context_holder {
    std::shared_ptr<gpu_toolkit> _context;
protected:
    context_holder();
    virtual ~context_holder() = default;
    const std::shared_ptr<gpu_toolkit>& context() const { return _context; }
};

struct profiling_period_event : instrumentation::profiling_period {
    profiling_period_event(const cl::Event& event, cl_profiling_info start, cl_profiling_info end )
        : _event(event)
        , _start(start)
        , _end(end)
        {}

    std::chrono::nanoseconds value() const override {
        size_t start_nanoseconds;
        _event.getProfilingInfo(_start, &start_nanoseconds);
        size_t end_nanoseconds;
        _event.getProfilingInfo(_end, &end_nanoseconds);
        return std::chrono::nanoseconds(static_cast<long long>(end_nanoseconds - start_nanoseconds));
    }

private:
    cl::Event _event;
    cl_profiling_info _start;
    cl_profiling_info _end;
};

class gpu_toolkit {
    cl::Device _device;
    cl::Context _context;
    cl::CommandQueue _command_queue;
    kernels_cache _kernels_cache;
//    cl::Program _program;
    std::vector<instrumentation::profiling_info> _profiling_info;

    gpu_toolkit();

    static std::shared_ptr<gpu_toolkit>get();
    friend class context_holder;
public:

    cl::Device& device() { return _device; }
    cl::Context& context() { return _context; }
    cl::CommandQueue& queue() { return _command_queue; }
//    cl::Program& program() { return _program; }
    kernels_cache& get_kernels_cache() { return _kernels_cache; }
    void report_profiling(const instrumentation::profiling_info& info) { _profiling_info.push_back(info); }
    const std::vector<instrumentation::profiling_info>& get_profiling_info() const { return _profiling_info; }

    gpu_toolkit(const gpu_toolkit& other) = delete;
    gpu_toolkit(gpu_toolkit&& other) = delete;
    gpu_toolkit& operator=(const gpu_toolkit& other) = delete;
    gpu_toolkit& operator=(gpu_toolkit&& other) = delete;
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////
inline context_holder::context_holder() : _context(gpu_toolkit::get()) {}

}}
