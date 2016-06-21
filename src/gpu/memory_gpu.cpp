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
#include "memory.h"
#include "ocl_toolkit.h"
#include "kernels_cache.h"

namespace neural { namespace gpu {

namespace {
    struct attach_gpu_allocator {
        attach_gpu_allocator() {
            allocators_map::instance().insert({ engine::gpu,{ gpu_toolkit::allocate_memory_gpu, gpu_toolkit::deallocate_memory_gpu } });
        }
        ~attach_gpu_allocator() {}
    };

#ifdef __GNUC__
    __attribute__((visibility("default"))) //todo meybe dll_sym?
#elif _MSC_VER
#   pragma section(".nn_init$m", read, write)
#endif
    attach_gpu_allocator attach_impl;
}

}}
