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
#include "implementation_map.h"
#include "gpu/kernel.h"

const std::string kernelName = "depth_concatenate_gpu";
const std::string kernelCode = R"__krnl(
KERNEL (depth_concatenate_gpu)(__global float* input, __global float* output, uint depth_offset)
{
    uint global_id = get_global_id(0);

    uint input_offset = global_id * INPUT_FEATURE_NUM * OUTPUT_BATCH_NUM;
    uint output_offset = OUTPUT_BATCH_NUM * (depth_offset + global_id * OUTPUT_FEATURE_NUM);
    for(uint f = 0; f < INPUT_FEATURE_NUM * OUTPUT_BATCH_NUM; f++)
    {
        output[output_offset++] = input[input_offset++];
    }
}
)__krnl";

namespace neural {

    struct depth_concatenate_gpu : is_an_implementation {
        depth_concatenate &outer;
        std::vector<gpu::kernel> _kernel;

        depth_concatenate_gpu(depth_concatenate &arg) : is_an_implementation(neural::type_id<depth_concatenate_gpu>())
            , outer(arg)
        {
            for (size_t i = 0; i < outer.argument.input.size(); i++)
            {
                _kernel.emplace_back(select_kernel_name(), get_jit_constants(i));
            }
        }

        const std::string& select_kernel_name() const {
            return kernelName;
        }

        gpu::jit_constants get_jit_constants(size_t input_id) {
            return gpu::jit_constants{
                gpu::make_jit_constant("INPUT", outer.input_memory(input_id).argument.size),
                gpu::make_jit_constant("OUTPUT", outer.output_memory(0).argument.size)
            };
        }

        static void implementation(const void *ptr) {
            auto me = static_cast<const depth_concatenate_gpu*>(ptr);
            auto& outer = me->outer;

            size_t input_count = outer.argument.input.size();

            auto& input_mem = outer.input_memory(0);
            auto& output_mem = outer.output_memory(0);

            size_t gws0 = input_mem.argument.size.spatial[0] * input_mem.argument.size.spatial[1];
            size_t lws0 = 32;
            while (gws0 % lws0)
            {
                lws0--;
            }

            uint32_t depth_offset = 0;
            for (size_t i = 0; i < input_count; i++)
            {
                uint32_t input_depth_count = outer.input_memory(i).argument.size.feature[0];
                me->_kernel[i].run<gpu::input_mem, gpu::output_mem, cl_uint>
                    ({ { gws0 },{ lws0 } }, outer.input_memory(i), output_mem, depth_offset);
                depth_offset += input_depth_count;
            }

        }

        static is_an_implementation *create(depth_concatenate &arg) { return new depth_concatenate_gpu(arg); };
        task_group work() override { return{ { task{ implementation, this } }, schedule::single }; };

    };

    depth_concatenate::arguments::arguments(std::vector<primitive_at> in, primitive out)
    : output({out})
    , input({in})
    {}

    depth_concatenate::arguments::arguments(neural::memory::format::type out_fmt, std::vector<primitive_at> in)
        : input({in})
    {
        uint32_t out_depth_count = 0;
        for (auto i : input)
        {
            out_depth_count += get_memory_primitive(i.primitive()).argument.size.feature[0];
        }
        auto output_size = get_memory_primitive(input[0].primitive()).argument.size;
        output_size.feature[0] = out_depth_count;
        output = { memory::allocate({ out_fmt, output_size }) };
    }

primitive depth_concatenate::create(depth_concatenate::arguments arg) {
    auto& input_arg  = get_memory_primitive(arg.input[0].primitive()).argument;
    auto& output_arg = arg.output[0].as<const memory&>().argument;

    auto format = input_arg.format;

    uint32_t depth_count = 0;
    auto input_size = input_arg.size;
    for (auto i : arg.input)
    {
        auto& input_mem = get_memory_primitive(i.primitive());
        if (input_mem.argument.format != format) throw std::runtime_error("Every input must have the same format!");
        if (input_mem.argument.size.batch[0] != input_size.batch[0]) throw std::runtime_error("Every input must have the same number of batches!");
        if (input_mem.argument.size.spatial[0] != input_size.spatial[0]) throw std::runtime_error("Every input must have the same size in X dimension!");
        if (input_size.spatial.size() > 1)
            if (input_mem.argument.size.spatial[1] != input_size.spatial[1]) throw std::runtime_error("Every input must have the same size in Y dimension!");
        depth_count += input_mem.argument.size.feature[0];
    }

    if (output_arg.format != format) throw std::runtime_error("Input and output must have the same format!");
    if (depth_count != output_arg.size.feature[0]) throw std::runtime_error("Output depth count mismatch sum of input depths!");
    if (output_arg.size.batch[0] != input_size.batch[0]) throw std::runtime_error("Output batch size must match input batch size!");
    if (output_arg.size.spatial[0] != input_size.spatial[0]) throw std::runtime_error("Output X size must match input X size!");
    if (input_size.spatial.size() > 1)
        if (output_arg.size.spatial[1] != input_size.spatial[1]) throw std::runtime_error("Output Y size must match input Y size!");

    return is_a_primitive::create<depth_concatenate>(arg);
}


namespace {
    struct attach {
        attach() {
            gpu::kernel_templates::add(kernelName, kernelCode);

            auto key_fw = std::make_tuple(engine::gpu, memory::format::yxfb_f32, memory::format::yxfb_f32);
            auto val_fw = depth_concatenate_gpu::create;

            implementation_map<depth_concatenate>::add(key_fw, val_fw);
        }
        ~attach() {}
    };
}

#ifdef __GNUC__
__attribute__((visibility("default"))) //todo meybe dll_sym?
#elif _MSC_VER
#   pragma section(".nn_init$m", read, write)
#endif
attach attach_impl;

} // namespace neural