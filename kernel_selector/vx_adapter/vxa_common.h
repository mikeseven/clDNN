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

#pragma once

#include <cstddef>
#include <memory>
#include <mutex>
#include "api/vx_cldnn_adapter.h"
#include "api/vx_cldnn_adapter_types.h"
#include "cache/program_cache.h"
#include "cache/cache_types.h"

namespace clDNN
{
    using kernel_cache = KernelSelector::gpu::cache::program_cache;
    using binary_data = KernelSelector::gpu::cache::binary_data;

    bool IsSupported(const Params& params);

    inline std::shared_ptr<kernel_cache> GetKernelCache()
    {
        static std::recursive_mutex mutex;
        
        // We don't want to clear the cache even if no one use binary manager at some points
        static std::shared_ptr<kernel_cache> primitive_handle;
        std::lock_guard<std::recursive_mutex> create_lock{ mutex };
        
        if (primitive_handle == nullptr) 
        {
            primitive_handle = std::make_shared<kernel_cache>();
        }

        return primitive_handle;
    }
}