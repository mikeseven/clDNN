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

#include "depth_concatenate_arg.h"
#include "network_impl.h"
#include "primitive_type_base.h"

#include "neural_impl.h"
#include "gpu/kernel.h"
#include "implementation_map.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace cldnn
{
primitive_type_id depth_concatenate::type_id()
{
    static primitive_type_base<depth_concatenate, depth_concatenate_arg> instance;
    return &instance;
}

layout depth_concatenate_arg::calc_output_layout(network_impl& network, std::shared_ptr<const depth_concatenate> desc)
{
    auto& input_ids = desc->input();
    auto result = network.get_primitive(input_ids.at(0))->output_memory().get_layout();
    for(size_t i = 1; i < input_ids.size(); i++)
    {
        result.size.feature[0] += network.get_primitive(input_ids[i])->output_memory().get_layout().size.feature[0];
    }
    return result;
}

depth_concatenate_arg::depth_concatenate_arg(network_impl& network, std::shared_ptr<const depth_concatenate> desc)
    :primitive_arg_base(network, desc, calc_output_layout(network, desc))
{
    auto input_arg = input_memory(0).argument();
    auto output_arg = output_memory().argument();

    auto format = input_arg.format;

    tensor::value_type depth_count = 0;
    auto input_size = input_arg.size;
    for (auto i : _inputs)
    {
        auto& input_mem = i->output_memory();
        if (input_mem.argument().format != format) throw std::runtime_error("Every input must have the same format!");
        if (input_mem.argument().size.batch[0] != input_size.batch[0]) throw std::runtime_error("Every input must have the same number of batches!");
        if (input_mem.argument().size.spatial[0] != input_size.spatial[0]) throw std::runtime_error("Every input must have the same size in X dimension!");
        if (input_size.spatial.size() > 1)
            if (input_mem.argument().size.spatial[1] != input_size.spatial[1]) throw std::runtime_error("Every input must have the same size in Y dimension!");
        depth_count += input_mem.argument().size.feature[0];
    }

    if (output_arg.format != format) throw std::runtime_error("Input and output must have the same format!");
    if (depth_count != output_arg.size.feature[0]) throw std::runtime_error("Output depth count mismatch sum of input depths!");
    if (output_arg.size.batch[0] != input_size.batch[0]) throw std::runtime_error("Output batch size must match input batch size!");
    if (output_arg.size.spatial[0] != input_size.spatial[0]) throw std::runtime_error("Output X size must match input X size!");
    if (input_size.spatial.size() > 1)
        if (output_arg.size.spatial[1] != input_size.spatial[1]) throw std::runtime_error("Output Y size must match input Y size!");
}
}

namespace neural
{
// Kernel names.
static const std::string kernel_name = "depth_concatenate_gpu";

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
        : _outer(outer)
    {
        auto context = outer.get_network().get_engine()->get_context();
        auto engine_info = context->get_engine_info();

        auto inputs_count = _outer.argument.input.size();

        _kernels_with_data.reserve(inputs_count);
        for (size_t input_idx = 0; input_idx < inputs_count; ++input_idx)
        {
            auto data = set_kernel_data(input_idx, _outer, engine_info);
            gpu::kernel kernel(context, data.kernel_name, get_jit_constants(_outer, data));

            _kernels_with_data.emplace_back(std::move(kernel), std::move(data));
        }
    }

    static kernel_data set_kernel_data(size_t input_idx, const depth_concatenate& outer, const gpu::engine_info_internal& info)
    {
        const auto& input_mem = outer.input_memory(input_idx);  // current input

        kernel_data kd;

        kd.input_idx = input_idx;
        kd.fp16_unit_used = input_mem.get_layout().data_type == cldnn::data_types::f16;
        kd.fp16_supported = info.supports_fp16 != 0;

        // Determine global work sizes.
        kd.gws0 = input_mem.argument().size.spatial[0] * input_mem.argument().size.spatial[1];
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
            gpu::make_jit_constant("INPUT",          outer.input_memory(data.input_idx).argument().size),
            gpu::make_jit_constant("OUTPUT",         outer.output_memory().argument().size),
            gpu::make_jit_constant("FP16_SUPPORTED", static_cast<int>(data.fp16_supported)),
            gpu::make_jit_constant("FP16_UNIT_USED", static_cast<int>(data.fp16_unit_used)),
            gpu::make_jit_constant("UNIT_TYPE",      data.fp16_unit_used ? "half" : "float")
        };
    }

    cldnn::refcounted_obj_ptr<cldnn::event_impl> execute(const std::vector<cldnn::refcounted_obj_ptr<cldnn::event_impl>>& events) override
    {

        size_t inputs_count = _outer.argument.input.size();

        const auto& output_mem = _outer.output_memory();  // output

        uint32_t depth_offset = 0;
        auto tmp_events = events;
        for (size_t input_idx = 0; input_idx < inputs_count; ++input_idx)
        {
            const auto& kd = _kernels_with_data[input_idx].second;

            const auto& input_mem = _outer.input_memory(input_idx);  // current input

            uint32_t input_depth_count = input_mem.argument().size.feature[0];
            auto event = _kernels_with_data[input_idx].first.run<gpu::input_mem, gpu::output_mem, cl_uint>
                ({{kd.gws0}, {kd.lws0}}, tmp_events, input_mem, output_mem, depth_offset);
            depth_offset += input_depth_count;
            tmp_events.clear();
            tmp_events.push_back(event);
        }
        return tmp_events.at(0);
    }

    static is_an_implementation *create(depth_concatenate &arg) { return new depth_concatenate_gpu(arg); };
};

namespace {
    struct attach {
        attach() {
            auto val_fw = depth_concatenate_gpu::create;

            auto key_fw = std::make_tuple(cldnn::engine_types::ocl, memory::format::yxfb_f32);
            implementation_map<depth_concatenate>::add(key_fw, val_fw);
            key_fw = std::make_tuple(cldnn::engine_types::ocl, memory::format::yxfb_f16);
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