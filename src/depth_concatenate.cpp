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
#include "gpu/cache/primitive_db.h"
#include "gpu/kernel.h"
#include "implementation_map.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>


namespace neural
{
// Kernel names.
static const std::string kernel_name = "depth_concatenate_gpu";

// GPU engine information helpers.
namespace
{
struct gpu_info_helper : gpu::context_holder
{
    gpu::engine_info get_engine_info() const
    {
        return context()->get_engine_info();
    }
};
}

struct depth_concatenate_gpu : is_an_implementation
{
    struct kernel_data
    {
        size_t input_idx;
        size_t gws0;
        size_t lws0;
        std::string kernel_name;
        bool fp16_unit_used;
        bool fp16_supported;
    };

    const depth_concatenate& _outer;
    std::vector<std::pair<gpu::kernel, kernel_data>> _kernels_with_data;


    depth_concatenate_gpu(const depth_concatenate& outer)
        : is_an_implementation(neural::type_id<depth_concatenate_gpu>()),
        _outer(outer)
    {
        gpu_info_helper gpu_info;
        auto engine_info = gpu_info.get_engine_info();

        auto inputs_count = _outer.argument.input.size();

        _kernels_with_data.reserve(inputs_count);
        for (size_t input_idx = 0; input_idx < inputs_count; ++input_idx)
        {
            auto data = set_kernel_data(input_idx, _outer, engine_info);
            gpu::kernel kernel(data.kernel_name, get_jit_constants(_outer, data));

            _kernels_with_data.emplace_back(std::move(kernel), std::move(data));
        }
    }

    static kernel_data set_kernel_data(size_t input_idx, const depth_concatenate& outer, const gpu::engine_info& info)
    {
        const auto& input_mem = outer.input_memory(input_idx);  // current input

        kernel_data kd;

        kd.input_idx = input_idx;
        kd.fp16_unit_used = memory::traits(input_mem.argument.format).type->name == type_id<half_t>()->name;
        kd.fp16_supported = info.supports_fp16 != 0;

        // Determine global work sizes.
        kd.gws0 = input_mem.argument.size.spatial[0] * input_mem.argument.size.spatial[1];
        // Find largest positive local work size that is divider for global work size.
        kd.lws0 = std::min(std::max(kd.gws0, static_cast<size_t>(1)), static_cast<size_t>(32));
        while (kd.gws0 % kd.lws0 != 0)
        {
            --kd.lws0;
        }

        // Select kernel name.
        kd.kernel_name = kernel_name;

        return kd;
    }

    static gpu::jit_constants get_jit_constants(const depth_concatenate& outer, const kernel_data& data)
    {
        if (!data.fp16_supported && data.fp16_unit_used)
            throw std::invalid_argument("GPU device does not support half precision floating-point formats (cl_khr_fp16 extension)");

        return gpu::jit_constants {
            gpu::make_jit_constant("INPUT",          outer.input_memory(data.input_idx).argument.size),
            gpu::make_jit_constant("OUTPUT",         outer.output_memory(0).argument.size),
            gpu::make_jit_constant("FP16_SUPPORTED", static_cast<int>(data.fp16_supported)),
            gpu::make_jit_constant("FP16_UNIT_USED", static_cast<int>(data.fp16_unit_used)),
            gpu::make_jit_constant("UNIT_TYPE",      data.fp16_unit_used ? "half" : "float")
        };
    }

    static void implementation(const void *ptr)
    {
        auto me = static_cast<const depth_concatenate_gpu*>(ptr);
        const auto& outer = me->_outer;

        size_t inputs_count = outer.argument.input.size();

        const auto& output_mem = outer.output_memory(0);  // output

        uint32_t depth_offset = 0;
        for (size_t input_idx = 0; input_idx < inputs_count; ++input_idx)
        {
            const auto& kd = me->_kernels_with_data[input_idx].second;

            const auto& input_mem = outer.input_memory(input_idx);  // current input

            uint32_t input_depth_count = input_mem.argument.size.feature[0];
            me->_kernels_with_data[input_idx].first.run<gpu::input_mem, gpu::output_mem, cl_uint>
                ({{kd.gws0}, {kd.lws0}}, input_mem, output_mem, depth_offset);
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
            auto val_fw = depth_concatenate_gpu::create;

            auto key_fw = std::make_tuple(engine::gpu, memory::format::yxfb_f32, memory::format::yxfb_f32);
            implementation_map<depth_concatenate>::add(key_fw, val_fw);
            key_fw = std::make_tuple(engine::gpu, memory::format::yxfb_f16, memory::format::yxfb_f16);
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