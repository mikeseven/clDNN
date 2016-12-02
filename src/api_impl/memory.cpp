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
#include <memory>


namespace cldnn
{
struct simple_alloc_memory : memory_impl
{
    simple_alloc_memory(size_t size): _data(size)
    {}

    void* lock() override { return _data.data(); }
    void unlock() override {};
    size_t size() const override { return _data.size(); }
private:
    std::vector<uint8_t> _data;
};

struct simple_attached_memory : memory_impl
{
    simple_attached_memory(void* pointer, size_t size)
    : _pointer(pointer), _size(size)
    {}

    void* lock() override { return _pointer; }
    void unlock() override {}
    size_t size() const override { return _size; }
private:
    void* _pointer;
    size_t _size;
};

memory::memory(const memory& other): _layout(other._layout)
                                   , _data(other._data)
{
    _data->add_ref();
}

memory& memory::operator=(const memory& other)
{
    if (this == &other)
        return *this;
    _data->release();
    _layout = other._layout;
    _data = other._data;
    _data->add_ref();
    return *this;
}

memory::~memory()
{
    _data->release();
}

memory_impl* memory::allocate_buffer(size_t size, status_t* status) noexcept
{
    try
    {
        if (status)
            *status = CLDNN_SUCCESS;
        return new simple_alloc_memory(size);
    }
    catch (...)
    {
        if (status)
            *status = CLDNN_ERROR;
        return nullptr;
    }
}

memory_impl* memory::attach_buffer(void* pointer, size_t size, status_t* status) noexcept
{
    try
    {
        if (status)
            *status = CLDNN_SUCCESS;
        return new simple_attached_memory(pointer, size);
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
