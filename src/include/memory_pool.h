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
#pragma once
#include "api/CPP/layout.hpp"
#include "api/CPP/primitive.hpp"
#include "api_impl.h"
#include "refcounted_obj.h"

#include <vector>
#include <map>

namespace cldnn
{

struct memory_impl;
struct engine_impl;

struct memory_record
{
    std::vector<primitive_id> _users; // list of primitives that already use this memory object
    //refcounted_obj_ptr<memory_impl> _memory;
};
    
    // this class implements memory manager that handles 4 memory pools
    // - non padded buffers - 
    //     1 user requests for buffer with no padding. 
    //     2 Check if buffer with requested size exist
    //     3   * yes: check if any of current users exist on request conflict list if no - return this memory, otherwise goto 4
    //         * no: goto 4
    //     4 take next (allocations are sorted in increasing order) allocation. if there is no more allocations, create new allocation otherwise go t
    // - padded buffers - not implemented yet
    // - images 2d - not implemented yet
    // - images 2d arrays - not implemented yet
    // - immutable - if user request for non reusable resource don't use pool, return 

class memory_pool
{
    memory_pool();
    refcounted_obj_ptr<memory_impl> alloc_memory(layout layout);

    std::multimap<uint64_t, memory_record> _non_padded_pool;
    refcounted_obj_ptr<engine_impl> _engine;
    uint64_t _global_memory_used;

public:
    memory_pool(refcounted_obj_ptr<engine_impl> engine)
        : _engine(engine),
        _global_memory_used(0)
    {}

    refcounted_obj_ptr<memory_impl> get_memory(layout layout, primitive_id id, std::vector<primitive_id> conflict, bool reusable = true); // get from pool or create memory allocation
    void clear_pool();

};

}
