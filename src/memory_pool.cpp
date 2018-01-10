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

#include <algorithm> 
#include <fstream>

#include "memory_pool.h"
#include "engine_impl.h"
#include "memory_impl.h"
#include "program_impl.h"

#include "gpu/memory_gpu.h"

namespace cldnn
{
    memory_record::memory_record(std::set<primitive_id> users, refcounted_obj_ptr<memory_impl>& memory) :
        _users(users),
        _memory(memory)
    {}

    memory_impl::ptr memory_pool::alloc_memory(const layout& layout)
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
            if (layout.format.is_image_2d())
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

    bool memory_pool::has_conflict(std::set<primitive_id>& a, std::set<primitive_id>& b)
    {
        std::vector<primitive_id> intersection;
        set_intersection(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(intersection));
        return !intersection.empty();
    }

    memory_impl::ptr memory_pool::get_from_non_padded_pool(const layout& layout, const primitive_id& id, std::set<primitive_id>& restrictions)
    {
        auto it = _non_padded_pool.find(layout.bytes_count());
        // can't find exact size? try to find bigger one
        if (it == _non_padded_pool.end())
            it = _non_padded_pool.upper_bound(layout.bytes_count());
        while (it != _non_padded_pool.end())
        {
            if (!has_conflict(it->second._users, restrictions))
            {
                it->second._users.insert(id);
                auto ret_mem = _engine->reinterpret_buffer(*it->second._memory, layout);
                return ret_mem;
            }
            else
                it++;
        }
        // didn't find anything for you? create new resource
        auto mem = alloc_memory(layout);
        {
            _non_padded_pool.insert(std::pair<uint64_t, cldnn::memory_record>(layout.bytes_count(), memory_record({ id }, mem)));
            // we don't want to store any resources with no parents so memory pool has to store weak pointer of _engine. 
            _engine->release();
        }
        return mem;
    }

    memory_impl::ptr memory_pool::get_memory(const layout& layout)
    {
        return alloc_memory(layout);
    }

    memory_impl::ptr memory_pool::get_memory(const layout& layout, const primitive_id& id, std::set<primitive_id>& restrictions, bool reusable)
    {

        if (reusable)
        {
            if (!layout.format.is_image() && layout.data_padding == padding{ { 0,0,0,0 }, 0 }) // non-padded buffers
            {
                 return get_from_non_padded_pool(layout, id, restrictions);
            }
            else if (!layout.format.is_image()) // padded buffers
            {
                // not yet
                return alloc_memory(layout);
            }
            else  // images
            {
                // not yet
                return alloc_memory(layout);
            }
        }

        return alloc_memory(layout);
    }

    void memory_pool::clear_pool()
    {
        _non_padded_pool.clear();
    }

    memory_pool::memory_pool(engine_impl& engine)
        : _engine(&engine),
        _global_memory_used(0)
    {
        _engine->release(); // since engine is refcount object and there is circular dependency until context will be moved to memory pool we need 
                            // to detach engine while destroying memory pool
    }

    void memory_pool::dump_memory_pool(const program_impl& program , std::string path, std::string dep)
    {
        using namespace std;
        ofstream log(path);

        log << "\nNon-padded pool:" <<endl;
        log << "Size\tUsers:" << endl;
        for (auto record : _non_padded_pool)
        {
            log << record.first;
            for (auto usr : record.second._users)
                log << ", " << usr;
            log << endl;
        }
        log << dep;
        log.close();

        color_graph(program);
    }

    void memory_pool::color_graph(const program_impl& program)
    {
        uint32_t color = 0;
        auto last_key = _non_padded_pool.begin()->first;
        for (auto mem : _non_padded_pool)
        {
            if (last_key != mem.first)
            {
                color++;
                last_key -= mem.first;
            }
            for (auto usr : mem.second._users)
                program.get_node(usr).set_reused_memory_color(color);
        }
    }
}

