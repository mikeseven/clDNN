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
#pragma once
#include "kernel.h"
#include "memory_gpu.h"

namespace neural { namespace gpu {

    vector_arg::vector_arg(const neural::vector<uint32_t>& arg) : _vec(arg) {
        _clBuffer = cl::Buffer(context()->context(), CL_MEM_READ_ONLY, neural_vector::size_of_vector(arg));
        cl::Event end_event;
        auto mapped_vec = reinterpret_cast<neural_vector*>(context()->queue().enqueueMapBuffer(_clBuffer, true, CL_MAP_WRITE, 0, neural_vector::size_of_vector(_vec), 0, &end_event));
        end_event.wait();
        mapped_vec->initialize(_vec);
        context()->queue().enqueueUnmapMemObject(_clBuffer, mapped_vec, 0, &end_event);
        end_event.wait();
    }

    vector_arg::~vector_arg() {}

    memory_arg::memory_arg(const neural::memory& mem, bool copy_input, bool copy_output) : _mem(mem), _copy_input(copy_input), _copy_output(copy_output) {
        if (is_own()) {
            _clBuffer = context()->unmap_buffer(mem.pointer)->buffer();
        }
        else {
            mapped_buffer<neural_memory> buffer(context(), _mem.argument);
            if (_copy_input) {
                auto src = reinterpret_cast<char*>(_mem.pointer);
                auto dst = reinterpret_cast<char*>(buffer.data()->pointer());
                auto data_size = buffer.data()->data_size();
                std::copy(arr_begin(src, data_size), arr_end(src, data_size), arr_begin(dst, data_size));
            }
            _clBuffer = buffer.buffer();
        }
    }

    memory_arg::~memory_arg() {
        if (is_own()) {
            //TODO remove const_cast: check if .pointer field of gpu owned_memory can be kept unchanged.
            const_cast<neural::memory&>(_mem).pointer = context()->map_memory_buffer(_clBuffer, _mem.argument)->data()->pointer();
        }
        else if (_copy_output) {
            mapped_buffer<neural_memory> buffer(context(), _clBuffer, _mem.argument);
            auto src = reinterpret_cast<char*>(buffer.data()->pointer());
            auto dst = reinterpret_cast<char*>(_mem.pointer);
            auto data_size = buffer.data()->data_size();
            std::copy(arr_begin(src, data_size), arr_end(src, data_size), arr_begin(dst, data_size));
        }
    }

} }