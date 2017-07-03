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
#include "vxa_ocl_toolkit.h"

namespace clDNN {
    namespace gpu
    {
        cl::Device GetGPUDevice() 
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
                    if (d.getInfo<CL_DEVICE_TYPE>() == CL_DEVICE_TYPE_GPU) 
                    {
                        auto vendor_id = d.getInfo<CL_DEVICE_VENDOR_ID>();
                        //set Intel GPU device default
                        constexpr cl_uint intel_vendor = 0x8086;
                        if (vendor_id == intel_vendor) 
                        {
                            return d;
                        }
                    }
                }
            }
            throw std::runtime_error("No OpenCL GPU device found.");
        }

        GPUToolkit::GPUToolkit()
            : _device(GetGPUDevice())
            , _context(_device)
            , _command_queue(_context, _device, cl::QueueProperties::Profiling)
        {
            _device.getInfo(CL_DEVICE_EXTENSIONS, &_extensions);
        }

        std::shared_ptr<GPUToolkit> GPUToolkit::get()
        {
            static std::recursive_mutex mutex;
            static std::weak_ptr<GPUToolkit> toolkit;
            std::lock_guard<std::recursive_mutex> create_lock{ mutex };
            if (toolkit.expired()) {
                std::shared_ptr<GPUToolkit> result{ new GPUToolkit(), [&](GPUToolkit* ptr) {
                    std::lock_guard<std::recursive_mutex> delete_lock{ mutex };
                    delete ptr;
                } };
                toolkit = result;
                return result;
            }

            return std::shared_ptr<GPUToolkit>(toolkit);
        }

    }
}
