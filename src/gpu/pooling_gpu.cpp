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
const std::string kernelName_average = "Pooling_GPU_average";

namespace neural {
struct pooling_gpu : is_an_implementation {
    pooling &outer;
    gpu::kernel _kernel;

    pooling_gpu(pooling &arg) : is_an_implementation(neural::type_id<pooling_gpu>())
        , outer(arg) 
        , _kernel(select_kernel_name(), get_jit_constants())
    {}

    const std::string& select_kernel_name() const {
        switch (outer.argument.mode)
        {
        case pooling::mode::max:
            return kernelName_max;
        case pooling::mode::average:
            return kernelName_average;
        default:
            throw std::runtime_error("Unknown pooling mode.");
        }
    }

    gpu::jit_constants get_jit_constants() const {
        return {
            gpu::make_jit_constant("INPUT", outer.input_memory(0).argument.size),
            gpu::make_jit_constant("OUTPUT", outer.output_memory(0).argument.size),
            gpu::make_jit_constant("WINDOW", outer.argument.size),
            gpu::make_jit_constant("STRIDE", outer.argument.stride)
        };
    }

    static void implementation(const void *ptr) {
        auto me = static_cast<const pooling_gpu*>(ptr);
        auto& outer = me->outer;

        // input
        auto& input_mem = outer.input_memory(0);

        // output
        auto& output_mem = outer.output_memory(0);

        size_t gws0 = output_mem.argument.size.batch[0] * output_mem.argument.size.feature[0];

        me->_kernel.run<gpu::input_mem, gpu::output_mem>
            ({ { gws0, output_mem.argument.size.spatial[0], output_mem.argument.size.spatial[1]}, { std::min(gws0, static_cast<size_t>(32)), 1, 1 } }, input_mem, output_mem);
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
			// cache implementation phase #1 that is a initial switch for using primitive database instead of string kernels
			// at later steps primitive database will be created only once per loading library but as for now it would require 
			// large refactor, so it will be done in smaller incremental steps. The same goes for picking first implementation
			// from the returned list.
			gpu::manager::primitive_db database;
            gpu::kernel_templates::add(kernelName_max, database.get(kernelName_max).at(0));
            gpu::kernel_templates::add(kernelName_average, database.get(kernelName_average).at(0));

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
