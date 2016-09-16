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

namespace neural { namespace gpu {

configuration::configuration()
    : enable_profiling(false)
    , device_type(gpu)
    , device_vendor(0x8086)
    , compiler_options("")
{}

configuration& configuration::get() {
    static configuration instance;
    return instance;
}

cl_device_type convert_configuration_device_type(configuration::device_types device_type) {
    cl_device_type device_types[] = {
            CL_DEVICE_TYPE_DEFAULT,
            CL_DEVICE_TYPE_CPU,
            CL_DEVICE_TYPE_GPU,
            CL_DEVICE_TYPE_ACCELERATOR };
    return device_types[device_type];
}

cl::Device get_gpu_device() {
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    cl::Device default_device;
    for (auto& p : platforms) {
        std::vector<cl::Device> devices;
        p.getDevices(CL_DEVICE_TYPE_ALL, &devices);
        for (auto& d : devices) {
            if (d.getInfo<CL_DEVICE_TYPE>() == convert_configuration_device_type(configuration::get().device_type)) {
                auto vendor_id = d.getInfo<CL_DEVICE_VENDOR_ID>();
                //set Intel GPU device default
                if (vendor_id == configuration::get().device_vendor) {
                    return d;
                }
            }
        }
    }
    throw std::runtime_error("No OpenCL GPU device found.");
}

gpu_toolkit::gpu_toolkit() 
    : _device(get_gpu_device())
    , _context(_device)
    , _command_queue(_context,
                     _device,
                     configuration::get().enable_profiling
                        ? cl::QueueProperties::Profiling
                        : cl::QueueProperties::None)
    {}

std::shared_ptr<gpu_toolkit> gpu_toolkit::get() {
    static std::recursive_mutex mutex;
    static std::weak_ptr<gpu_toolkit> toolkit;
    std::lock_guard<std::recursive_mutex> create_lock{ mutex };
    if(toolkit.expired()) {
        std::shared_ptr<gpu_toolkit> result{ new gpu_toolkit(), [&](gpu_toolkit* ptr) {
            std::lock_guard<std::recursive_mutex> delete_lock{ mutex };
            delete ptr;
        } };
        toolkit = result;
        return result;
    }

    return std::shared_ptr<gpu_toolkit>(toolkit);
}

}}
