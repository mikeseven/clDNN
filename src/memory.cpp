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
#include "primitive_arg.h"
#include "primitive_type.h"
#include "memory_impl.h"
#include "engine_impl.h"
#include <memory>


namespace cldnn
{

const layout& memory::get_memory_layout(cldnn_memory_t memory)
{
    return memory->get_layout();
}

bool memory::is_memory_allocated_by(cldnn_memory_t memory, cldnn_engine_t engine)
{
    return memory->is_allocated_by(engine);
}

memory_impl* memory::allocate_buffer(cldnn_engine_t engine, layout layout, status_t* status)
{
    try
    {
        if (status)
            *status = CLDNN_SUCCESS;
        return engine->allocate_buffer(layout);
    }
    catch (...)
    {
        if (status)
            *status = CLDNN_ERROR;
        return nullptr;
    }
}

memory_impl* memory::attach_buffer(layout layout, void* pointer, size_t size, status_t* status)
{
    try
    {
        if (status)
            *status = CLDNN_SUCCESS;
        if (layout.data_size() > size) std::invalid_argument("buffer size does not match layout size");
        return new simple_attached_memory(layout, pointer);
    }
    catch (...)
    {
        if (status)
            *status = CLDNN_ERROR;
        return nullptr;
    }
}

void memory::retain_memory(cldnn_memory_t memory)
{
    memory->add_ref();
}

void memory::release_memory(cldnn_memory_t memory)
{
    memory->release();
}

void* memory::lock_buffer(cldnn_memory_t memory, status_t* status)
{
    try
    {
        if (status)
            *status = CLDNN_SUCCESS;
        return memory->lock();
    }
    catch (...)
    {
        if (status)
            *status = CLDNN_ERROR;
        return nullptr;
    }
}

status_t memory::unlock_buffer(cldnn_memory_t memory)
{
    try
    {
        memory->unlock();
        return CLDNN_SUCCESS;
    }
    catch (...)
    {
        return CLDNN_ERROR;
    }
}
}
