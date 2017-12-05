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
#include "memory_gpu.h"
#include "engine_impl.h"

namespace cldnn { namespace gpu {

gpu_buffer::gpu_buffer(const refcounted_obj_ptr<engine_impl>& engine, const layout& layout)
    : memory_impl(engine, layout)
    , _context(engine->get_context())
    , _lock_count(0)
    , _buffer(_context->context(), CL_MEM_READ_WRITE, size())
    , _mapped_ptr(nullptr)
{
    void* ptr = gpu_buffer::lock();
    memset(ptr, 0, size());
    gpu_buffer::unlock();
}

gpu_buffer::gpu_buffer(const refcounted_obj_ptr<engine_impl>& engine, const layout& new_layout, const cl::Buffer& buffer)
    : memory_impl(engine, new_layout)
    , _context(engine->get_context())
    , _lock_count(0)
    , _buffer(buffer)
    , _mapped_ptr(nullptr)
{

}

void* gpu_buffer::lock() {
    std::lock_guard<std::mutex> locker(_mutex);
    if (0 == _lock_count) {
        _mapped_ptr = _context->queue().enqueueMapBuffer(_buffer, CL_TRUE, CL_MAP_WRITE, 0, size());
    }
    _lock_count++;
    return _mapped_ptr;
}

void gpu_buffer::unlock() {
    std::lock_guard<std::mutex> locker(_mutex);
    _lock_count--;
    if (0 == _lock_count) {
        _context->queue().enqueueUnmapMemObject(_buffer, _mapped_ptr);
        _mapped_ptr = nullptr;
    }
}

gpu_image2d::gpu_image2d(const refcounted_obj_ptr<engine_impl>& engine, const layout& layout)
    : memory_impl(engine, layout)
    , _context(engine->get_context())
    , _lock_count(0)
    , _buffer(_context->context(), CL_MEM_READ_WRITE, cl::ImageFormat(layout.format.image_channel_count() == 4 ? CL_RGBA : CL_R,
        layout.data_type == data_types::f16 ? CL_HALF_FLOAT : CL_FLOAT),
        static_cast<size_t>(layout.format.image_channel_count() == 4 ? (layout.size.spatial[0] * layout.size.feature[0] * layout.size.spatial[1] + 3) / 4 : layout.size.batch[0]),
        static_cast<size_t>(layout.size.spatial[0] * layout.size.feature[0] * layout.size.spatial[1]), 0)
    , _mapped_ptr(nullptr)
{
    switch (layout.format)
    {
    case format::image_weights_2d_c1_b_fyx:
        _width = layout.size.batch[0];
        _height = layout.size.spatial[0] * layout.size.feature[0] * layout.size.spatial[1];
        break;
    case format::image_weights_2d_c4_fyx_b:
        _width = layout.size.batch[0];
        _height = layout.size.spatial[0] * layout.size.feature[0] * layout.size.spatial[1];
        break;
    default:
        throw error("unsupported image type!");
    }

    cl_channel_order order = layout.format.image_channel_count() == 4 ? CL_RGBA : CL_R;
    cl_channel_type type = layout.data_type == data_types::f16 ? CL_HALF_FLOAT : CL_FLOAT;
    cl::ImageFormat imageFormat(order, type);
    _buffer = cl::Image2D(_context->context(), CL_MEM_READ_WRITE, imageFormat, _width, _height, 0);
    void* ptr = gpu_image2d::lock();
    for(uint64_t y = 0; y < static_cast<uint64_t>(layout.size.spatial[0] * layout.size.feature[0] * layout.size.spatial[1]); y++)
        memset(ptr, 0, static_cast<size_t>(y*_row_pitch));
    gpu_image2d::unlock();
}

gpu_image2d::gpu_image2d(const refcounted_obj_ptr<engine_impl>& engine, const layout& new_layout, const cl::Image2D& buffer)
    : memory_impl(engine, new_layout)
    , _context(engine->get_context())
    , _lock_count(0)
    , _buffer(buffer)
    , _mapped_ptr(nullptr)
{

}

void* gpu_image2d::lock() {
    std::lock_guard<std::mutex> locker(_mutex);
    if (0 == _lock_count) {
        _mapped_ptr = _context->queue().enqueueMapImage(_buffer, CL_TRUE, CL_MAP_WRITE, { 0, 0, 0 }, { _width, _height, 1 }, &_row_pitch, &_slice_pitch);
    }
    _lock_count++;
    return _mapped_ptr;
}

void gpu_image2d::unlock() {
    std::lock_guard<std::mutex> locker(_mutex);
    _lock_count--;
    if (0 == _lock_count) {
        _context->queue().enqueueUnmapMemObject(_buffer, _mapped_ptr);
        _mapped_ptr = nullptr;
    }
}

}}
