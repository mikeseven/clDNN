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
#include "primitive_type.h"
#include "primitive_arg.h"
#include "memory_impl.h"
#include "engine_impl.h"
#include <memory>


namespace cldnn
{
memory::memory(const memory& other): _data(other._data)
{
    _data->add_ref();
}

memory& memory::operator=(const memory& other)
{
    if (this == &other || this->_data == other._data )
        return *this;
    _data->release();
    _data = other._data;
    _data->add_ref();
    return *this;
}

memory::~memory()
{
    _data->release();
}

const layout& memory::get_layout() const noexcept
{
    return _data->get_layout();
}

bool memory::is_allocated_by(const engine& engine) const noexcept
{
    return _data->is_allocated_by(engine.get());
}

memory_impl* memory::allocate_buffer(engine engine, layout layout, status_t* status) noexcept
{
    try
    {
        if (status)
            *status = CLDNN_SUCCESS;
        return engine.get()->allocate_buffer(layout);
    }
    catch (...)
    {
        if (status)
            *status = CLDNN_ERROR;
        return nullptr;
    }
}

memory_impl* memory::attach_buffer(layout layout, void* pointer, size_t size, status_t* status) noexcept
{
    try
    {
        if (status)
            *status = CLDNN_SUCCESS;
        assert(layout.data_size() == size);
        return new simple_attached_memory(layout, pointer);
    }
    catch (...)
    {
        if (status)
            *status = CLDNN_ERROR;
        return nullptr;
    }
}

void* memory::lock_buffer(status_t* status) const noexcept
{
    try
    {
        if (status)
            *status = CLDNN_SUCCESS;
        return _data->lock();
    }
    catch (...)
    {
        if (status)
            *status = CLDNN_ERROR;
        return nullptr;
    }
}

status_t memory::unlock_buffer() const noexcept
{
    try
    {
        _data->unlock();
        return CLDNN_SUCCESS;
    }
    catch (...)
    {
        return CLDNN_ERROR;
    }
}
}
