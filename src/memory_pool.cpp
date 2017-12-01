/*
// Copyright (c) 2017 Intel Corporation
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

#include "memory_pool.h"
#include "engine_impl.h"
#include "memory_impl.h"
#include "gpu/memory_gpu.h"

namespace cldnn
{
    memory_impl::ptr memory_pool::alloc_memory(layout layout)
    {
        auto context = _engine->get_context();
        if (layout.bytes_count() > context->get_engine_info().max_alloc_mem_size)
        {
            throw error("exceeded max size of memory object allocation", CLDNN_ALLOC_SIZE_EXCEEDED);
        }

        _global_memory_used += layout.bytes_count();
        if (_global_memory_used > context->get_engine_info().max_global_mem_size)
        {
            throw error("exceeded global device memory", CLDNN_GLOBAL_SIZE_EXCEEDED);
        }

        try {
            if (layout.format.is_image_weights_fyx_b())
                return{ new gpu::gpu_image2d(_engine, layout), false };
            else
                return{ new gpu::gpu_buffer(_engine, layout), false };
        }
        catch (const cl::Error& clErr)
        {
            switch (clErr.err())
            {
            case CL_MEM_OBJECT_ALLOCATION_FAILURE:
            case CL_OUT_OF_RESOURCES:
            case CL_OUT_OF_HOST_MEMORY:
            case CL_INVALID_BUFFER_SIZE:
                throw error("out of GPU resources", CLDNN_OUT_OF_RESOURCES);
            default:
                throw error("GPU buffer allocation failed", CLDNN_ERROR);
            }
        }
    }

    memory_impl::ptr memory_pool::get_memory(layout layout, primitive_id , std::vector<primitive_id> , bool )
    {

        //if (reusable)
        //{
        //   // return _engine->allocate_memory(layout);
        //}

        return alloc_memory(layout);
    }

    void memory_pool::clear_pool()
    {
        _non_padded_pool.clear();
    }
}

