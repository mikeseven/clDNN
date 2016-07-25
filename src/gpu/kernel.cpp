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
#include <iterator>
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
            _gpu_buffer = std::static_pointer_cast<gpu_buffer>(_mem.get_buffer());
        }
        else {
            _gpu_buffer = std::make_shared<gpu_buffer>(_mem.argument);
            if (_copy_input) {
                auto src = mem.pointer<char>();
                memory::ptr<char> dst(_gpu_buffer);
                std::copy(std::begin(src), std::end(src), std::begin(dst));
            }
        }
    }

    memory_arg::~memory_arg() {
        if (_copy_output && !is_own()) {
            memory::ptr<char> src(_gpu_buffer);
            auto dst = _mem.pointer<char>();
            std::copy(std::begin(src),std::end(src), std::begin(dst));
        }
    }

} }