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
#include "memory.h"
#include <numeric>

namespace neural { namespace gpu {

cl::Device get_gpu_device() {
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    cl::Device default_device;
    for (auto& p : platforms) {
        std::vector<cl::Device> devices;
        p.getDevices(CL_DEVICE_TYPE_ALL, &devices);
        for (auto& d : devices) {
            if (d.getInfo<CL_DEVICE_TYPE>() == CL_DEVICE_TYPE_GPU) {
                auto vendor_id = d.getInfo<CL_DEVICE_VENDOR_ID>();
                //set Intel GPU device default
                if (vendor_id == 0x8086) {
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
    , _command_queue(_context, _device) {}

/*
void gpu_toolkit::initialize() {
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    cl::Device default_device;
    for (auto& p : platforms) {
        std::vector<cl::Device> devices;
        p.getDevices(CL_DEVICE_TYPE_ALL, &devices);
        for(auto& d: devices) {
            if (d.getInfo<CL_DEVICE_TYPE>() == CL_DEVICE_TYPE_GPU) {
                auto vendor_id = d.getInfo<CL_DEVICE_VENDOR_ID>();
                //set Intel GPU device default
                if (vendor_id == 0x8086) {
                    _platform = p;
                    _device = d;
                    _context = cl::Context(_device);
                    _command_queue = cl::CommandQueue(_context);
                    auto default_platform = cl::Platform::setDefault(p);
                    if (default_platform != p) throw std::runtime_error("Error setting default platform.");

                    default_device = cl::Device::setDefault(d);
                    if (default_device != d) throw std::runtime_error("Error setting default device.");
                    break;
                }
            }
        }
        if(default_device() !=  nullptr) break;
    }

    if (default_device() == nullptr) {
        throw std::runtime_error("No OpenCL GPU device found.");
    }
}
*/

mapped_buffer<neural_memory>* gpu_toolkit::new_memory_buffer(neural::memory::arguments arg) {
    std::unique_ptr<mapped_buffer<neural_memory>> memory{ new mapped_buffer<neural_memory>(get(), arg) };
    auto result = memory.get();
    result->data()->initialize(arg);
    _mapped_memory.push(memory->data()->pointer(), std::move(memory));
    return result;
}

mapped_buffer<neural_memory>* gpu_toolkit::map_memory_buffer(const cl::Buffer& buf, neural::memory::arguments arg) {
    std::unique_ptr<mapped_buffer<neural_memory>> memory{ new mapped_buffer<neural_memory>(get(), buf, arg) };
    auto result = memory.get();
    result->data()->initialize(arg);
    _mapped_memory.push(memory->data()->pointer(), std::move(memory));
    return result;
}

std::unique_ptr<mapped_buffer<neural_memory>> gpu_toolkit::unmap_buffer(void* pointer) {
    return _mapped_memory.pop(pointer);
}

void* gpu_toolkit::allocate_memory_gpu(neural::memory::arguments arg) {
    auto mapped_mem = get()->new_memory_buffer(arg);
    return mapped_mem->data()->pointer();
}

void gpu_toolkit::deallocate_memory_gpu(void* pointer, neural::memory::arguments) {
    get()->unmap_buffer(pointer);
}

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
    else {
        return std::shared_ptr<gpu_toolkit>(toolkit);
    }
}

}}
