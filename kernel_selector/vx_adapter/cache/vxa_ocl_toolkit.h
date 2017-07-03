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

namespace clDNN { namespace gpu {

class GPUToolkit;

class context_holder {
protected:
    std::shared_ptr<GPUToolkit> _context;
    context_holder();
    context_holder(std::shared_ptr<GPUToolkit> context) : _context(context) {}
    virtual ~context_holder() = default;
    const std::shared_ptr<GPUToolkit>& context() const { return _context; }
};

class GPUToolkit 
{
    cl::Device _device;
    cl::Context _context;
    cl::CommandQueue _command_queue;
    std::string _extensions;
public:
    static std::shared_ptr<GPUToolkit>get();
    friend class context_holder;

    GPUToolkit();

    const cl::Device& device() const { return _device; }
    const cl::Context& context() const { return _context; }
    const cl::CommandQueue& queue() const { return _command_queue; }
    inline bool extension_supported(const std::string ext) { return _extensions.find(ext) != std::string::npos; }

    GPUToolkit(const GPUToolkit& other) = delete;
    GPUToolkit(GPUToolkit&& other) = delete;
    GPUToolkit& operator=(const GPUToolkit& other) = delete;
    GPUToolkit& operator=(GPUToolkit&& other) = delete;
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////
inline context_holder::context_holder() : _context(GPUToolkit::get()) {}

struct context_device
{
    cl::Context context;
    cl::Device device;
};

}}
