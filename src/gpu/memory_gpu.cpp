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
#include "memory.h"

namespace neural { namespace gpu {
    gpu_buffer::gpu_buffer(memory::arguments arg) : _argument(arg),
        _ref_count(0),
        _data_size(memory::size_of(arg)),
        _buffer(context()->context(), CL_MEM_READ_WRITE, _data_size),
        _mapped_ptr(nullptr) {}

    void* gpu_buffer::lock() {
        std::lock_guard<std::mutex> locker(_mutex);
        if (0 == _ref_count) {
            _mapped_ptr = context()->queue().enqueueMapBuffer(_buffer, CL_TRUE, CL_MAP_WRITE, 0, _data_size);
        }
        _ref_count++;
        return _mapped_ptr;
    }

    void gpu_buffer::release() {
        std::lock_guard<std::mutex> locker(_mutex);
        _ref_count--;
        if (0 == _ref_count) {
            context()->queue().enqueueUnmapMemObject(_buffer, _mapped_ptr);
            _mapped_ptr = nullptr;
        }
    }

    void gpu_buffer::reset(void* ptr) {
        auto me = lock();
        ::memcpy(me, ptr, _data_size);
        release();
    }

}}
