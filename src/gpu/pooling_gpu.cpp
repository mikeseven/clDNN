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

#include "api/neural.h"
#include "multidimensional_counter.h"
#include "implementation_map.h"
#include "kernel.h"
#include "cache/primitive_db.h"

const std::string kernelName_max = "Pooling_GPU_max";
const std::string kernelName_max_offset = "Pooling_GPU_max_offset";
const std::string kernelName_average = "Pooling_GPU_average";
const std::string kernelName_average_offset = "Pooling_GPU_average_offset";

namespace neural {
struct pooling_gpu : is_an_implementation {
    pooling &outer;
    gpu::kernel _kernel;

    pooling_gpu(pooling &arg) : is_an_implementation(neural::type_id<pooling_gpu>())
        , outer(arg) 
        , _kernel(select_kernel_name(), get_jit_constants())
    {}

    // Check if we need boundary checking in kernel
    bool needs_boundary_check() const
    {
        auto& input_mem = outer.input_memory(0);
        auto& input_offset = outer.argument.input_offset;
        
        if (input_offset.spatial[0] || input_offset.spatial[1])
            return true;

        auto& kernel_size = outer.argument.size;
        auto& stride = outer.argument.stride;

        // if modulo is not 0 that means it does not divide by stride, so we would go out of boundary
        auto mod_x = (input_mem.argument.size.spatial[0] - (2 * input_offset.spatial[0]) - kernel_size.spatial[0]) % stride.spatial[0];
        auto mod_y = (input_mem.argument.size.spatial[1] - (2 * input_offset.spatial[1]) - kernel_size.spatial[1]) % stride.spatial[1];

        return mod_x || mod_y;
    }

    const std::string& select_kernel_name() const {
        bool needs_boundary = needs_boundary_check();
        switch (outer.argument.mode)
        {
        case pooling::mode::max:
            if (needs_boundary)
            {
                return kernelName_max_offset;
            }
            else
            {
                return kernelName_max;
            }
        case pooling::mode::average:
            if (needs_boundary)
            {
                return kernelName_average_offset;
            }
            else
            {
                return kernelName_average;
            }
        default:
            throw std::runtime_error("Unknown pooling mode.");
        }
    }

    gpu::jit_constants get_jit_constants() const {
        gpu::jit_constants mem_consts{
            gpu::make_jit_constant("INPUT", outer.input_memory(0).argument.size),
            gpu::make_jit_constant("OUTPUT", outer.output_memory(0).argument.size),
            gpu::make_jit_constant("WINDOW", outer.argument.size),
            gpu::make_jit_constant("STRIDE", outer.argument.stride),
            gpu::make_jit_constant("INPUT_OFFSET", outer.argument.input_offset)
        };
        return mem_consts;
    }

    static void implementation(const void *ptr) {
        auto me = static_cast<const pooling_gpu*>(ptr);
        auto& outer = me->outer;

        // input
        auto& input_mem = outer.input_memory(0);

        // output
        auto& output_mem = outer.output_memory(0);

        size_t gws0 = output_mem.argument.size.batch[0] * output_mem.argument.size.feature[0];
        size_t lws0 = std::min(gws0, static_cast<size_t>(32));
        while (gws0%lws0)
        {
            lws0 /= 2;
        }
        me->_kernel.run<gpu::input_mem, gpu::output_mem>
            ({ { gws0, output_mem.argument.size.spatial[0], output_mem.argument.size.spatial[1]}, { lws0, 1, 1 } }, input_mem, output_mem);
    }

    static is_an_implementation *create(pooling &arg) {
        auto& input_arg = arg.input_memory(0).argument;
        auto& input_offset = arg.argument.input_offset;

        auto& input_buffer_size = input_arg.size;
        auto& output_arg = arg.output_memory(0).argument;
        auto& output_buffer_size = output_arg.size;
        auto& output_size = arg.argument.output_size;
        auto& output_offset = arg.argument.output_offset;
        auto& stride = arg.argument.stride;
        auto& window = arg.argument.size;
        auto& padding = arg.argument.padding;

        if (padding::zero != padding)                                      throw std::logic_error("Pooling supports only zero padding.");
        if (input_arg.format != memory::format::yxfb_f32)                  throw std::logic_error("Pooling reference uses yxfb_f32 format."); //todo, only this format?
        if (input_buffer_size.raw.size() != output_buffer_size.raw.size()) throw std::invalid_argument("Pooling input/output number of dimension does not match.");
        if (stride.raw.size() != output_buffer_size.raw.size())            throw std::invalid_argument("Pooling stride/output number of dimension does not match.");
        if (window.raw.size() != output_buffer_size.raw.size())            throw std::invalid_argument("Pooling window_size/output number of dimension does not match.");
        if (input_arg.format != output_arg.format)                         throw std::invalid_argument("Pooling input/output data format does not match.");
        
        // general formula: output size = (input size - window size) / step + 1
        for (size_t i = 0; i < input_offset.raw.size(); ++i) {
            if (output_buffer_size.raw[i] < output_size.raw[i] + output_offset.raw[i])
                throw std::runtime_error("Pooling output buffer size is to small.");
        }

        return new pooling_gpu(arg);
    }

    task_group work() override { return{ { task{ implementation, this } }, schedule::single }; };

};



namespace
{

    struct attach
    {
        attach()
        {
            auto key_fw = std::make_tuple(engine::gpu, memory::format::yxfb_f32, memory::format::yxfb_f32);
            auto val_fw = pooling_gpu::create;

            implementation_map<pooling>::add(key_fw, val_fw);
        }

        ~attach()
        {
        }
    };

#ifdef __GNUC__
    __attribute__((visibility("default")))
#elif _MSC_VER
#   pragma section(".nn_init$m", read, write)
#endif
    attach attach_impl;

}
}
