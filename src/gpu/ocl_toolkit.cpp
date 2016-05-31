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

namespace neural {

std::once_flag ocl_toolkit::ocl_initialized;
void ocl_toolkit::initialize_opencl() {
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    cl::Platform plat;
    for (auto& p : platforms) {
        std::string platver = p.getInfo<CL_PLATFORM_VERSION>();
        if (platver.find("OpenCL 2.") != std::string::npos) {
            plat = p;
        }
    }

    if (plat() == nullptr) {
        throw std::runtime_error("No OpenCL 2.0 platform found.");
    }

    cl::Platform newP = cl::Platform::setDefault(plat);
    if (newP != plat) {
        throw std::runtime_error("Error setting default platform.");
    }
}

ocl_toolkit& ocl_toolkit::get() {
    static ocl_toolkit toolkit;
    return toolkit;
}
}
