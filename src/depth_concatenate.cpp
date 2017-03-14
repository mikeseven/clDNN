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
#include "gpu/kd_selector.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <functional>

namespace cldnn
{
primitive_type_id depth_concatenate_type_id()
{
    static primitive_type_base<depth_concatenate, depth_concatenate_arg> instance;
    return &instance;
}

layout depth_concatenate_arg::calc_output_layout(const topology_map& topology_map, std::shared_ptr<const depth_concatenate> desc)
{
    auto& input_ids = desc->input();
    auto input0_desc = topology_map.at(input_ids.at(0))->primitive_desc;
    auto input_layout = input0_desc->type()->calc_output_layout(topology_map, input0_desc);
    auto result_sizes = input_layout.size.sizes();
    auto input_format = input_layout.size.format;

    // get indicies of feature coordinates and initialize particular result coordinate to 0
    auto& format_order = input_format.order();
    assert(result_sizes.size() == format_order.size());
    if (input_layout.size.feature.size() != 1) throw std::domain_error("depth_concatenate supports only one feature dimension");

    auto feature_index = format_order.find_first_of(format_traits::feature_chars());
    assert(feature_index != std::string::npos);

    // calculate sum of features from all inputs
    result_sizes[feature_index] = 0;
    for(auto id : input_ids)
    {
        auto input_desc = topology_map.at(id)->primitive_desc;
        auto input_sizes = input_desc->type()->calc_output_layout(topology_map, input_desc).size.sizes();
        result_sizes[feature_index] += input_sizes[feature_index];
    }
    return layout{input_layout.data_type, {input_format, result_sizes}};
}

depth_concatenate_arg::depth_concatenate_arg(network_impl& network, std::shared_ptr<const depth_concatenate> desc)
    :primitive_arg_base(network, desc, calc_output_layout(network.get_topology()->get_primitives(), desc))
{
    auto input_format = input_memory(0).argument().format;
    auto output_format = output_memory().argument().format;

    tensor::value_type depth_count = 0;
    auto input_size = _inputs.at(0)->non_padded_output_layout().size;
    auto output_size = non_padded_output_layout().size;
    for (const auto& i : _inputs)
    {
        auto& input_mem = i->output_memory();
        auto input_mem_size = i->non_padded_output_layout().size;
        if (input_mem.argument().format != input_format) throw std::runtime_error("Every input must have the same format!");
        if (input_mem_size.batch[0] != input_size.batch[0]) throw std::runtime_error("Every input must have the same number of batches!");
        if (input_mem_size.spatial[0] != input_size.spatial[0]) throw std::runtime_error("Every input must have the same size in X dimension!");
        if (input_size.spatial.size() > 1)
            if (input_mem_size.spatial[1] != input_size.spatial[1]) throw std::runtime_error("Every input must have the same size in Y dimension!");
        depth_count += input_mem.argument().size.feature[0];
    }

    if (output_format != input_format) throw std::runtime_error("Input and output must have the same format!");
    if (depth_count != output_size.feature[0]) throw std::runtime_error("Output depth count mismatch sum of input depths!");
    if (output_size.batch[0] != input_size.batch[0]) throw std::runtime_error("Output batch size must match input batch size!");
    if (output_size.spatial[0] != input_size.spatial[0]) throw std::runtime_error("Output X size must match input X size!");
    if (input_size.spatial.size() > 1)
        if (output_size.spatial[1] != input_size.spatial[1]) throw std::runtime_error("Output Y size must match input Y size!");
}
}

namespace neural
{
// Kernel names.
static const std::string kernel_name_yxfb = "depth_concatenate_gpu_yxfb";
static const std::string kernel_name_bfyx = "depth_concatenate_gpu_bfyx";
static const std::string kernel_name_bfyx_no_padding = "depth_concatenate_gpu_bfyx_no_padding";

template <>
struct kd_default_value_selector<neural::gpu::engine_info_internal::architectures>
{
    static constexpr neural::gpu::engine_info_internal::architectures value = neural::gpu::engine_info_internal::architectures::GEN_UNKNOWN;
};

template <>
struct kd_default_value_selector<gpu::engine_info_internal::configurations>
{
    static constexpr gpu::engine_info_internal::configurations value = gpu::engine_info_internal::configurations::GT_UNKNOWN;
};

struct depth_concatenate_gpu : is_an_implementation
{
    struct kernel_data
    {
        size_t gws0;
        size_t gws1;
        size_t lws0;
        size_t lws1;
        std::string kernel_name;
        bool fp16_unit_used;
    };

    const depth_concatenate& _outer;
    gpu::engine_info_internal _engine_info;

    std::vector<std::pair<gpu::kernel, kernel_data>> _kernels_with_data;

    typedef kd_selector_t<kernel_data, std::pair<int, const depth_concatenate &>, neural::memory::format::type, kd_optional_selector_t, int, neural::gpu::engine_info_internal::architectures, gpu::engine_info_internal::configurations> ks_type;
    static ks_type ks;

    depth_concatenate_gpu(const depth_concatenate& outer)
        : _outer(outer)
        , _engine_info(outer.get_network().get_engine()->get_context()->get_engine_info())
    {
        auto context = outer.get_network().get_engine()->get_context();

        const int inputs_count = static_cast<int>(_outer.argument.input().size());

        _kernels_with_data.reserve(inputs_count);
        for (auto input_idx = 0; input_idx < inputs_count; ++input_idx)
        {
            auto data = ks.get_kernel(std::make_pair(input_idx, std::cref(_outer)), _outer.input_memory(input_idx).argument().format, outer.input_memory(input_idx).argument().size.batch[0], _engine_info.architecture, _engine_info.configuration);//set_kernel_data(/*input_idx,*/ _outer/*, engine_info*/);
            gpu::kernel kernel(context, data.kernel_name, get_jit_constants(input_idx, data), _outer.id());

            _kernels_with_data.emplace_back(std::move(kernel), std::move(data));
        }
    }

    static kernel_data set_kernel_data(int input_idx, const depth_concatenate& outer)
    {
        const auto& input_mem = outer.input_memory(input_idx);  // current input

        kernel_data kd;

        kd.fp16_unit_used = input_mem.get_layout().data_type == cldnn::data_types::f16;

        // Determine global work sizes.
        kd.gws0 = input_mem.argument().size.spatial[0] * input_mem.argument().size.spatial[1];
        kd.gws1 = input_mem.argument().size.batch[0];
        // Find largest positive local work size that is divider for global work size.
        kd.lws0 = std::min(std::max(kd.gws0, static_cast<size_t>(1)), static_cast<size_t>(32));
        while (kd.gws0 % kd.lws0 != 0)
        {
            --kd.lws0;
        }

        kd.lws1 = 1;

        return kd;
    }

    gpu::jit_constants get_jit_constants(const int input_idx, const kernel_data& data)
    {
        auto fp16_supported = _engine_info.supports_fp16;
        if (!fp16_supported && data.fp16_unit_used)
            throw std::invalid_argument("GPU device does not support half precision floating-point formats (cl_khr_fp16 extension)");

        return gpu::jit_constants{
            gpu::make_jit_constant("INPUT",          _outer.input().at(input_idx)->non_padded_output_layout().size),
            gpu::make_jit_constant("OUTPUT",         _outer.non_padded_output_layout().size),
            gpu::make_jit_constant("INPUT_ELEMENTS_COUNT", _outer.input_memory(input_idx).count() / _outer.input_memory(input_idx).get_layout().size.batch[0]),
            gpu::make_jit_constant("FP16_SUPPORTED", static_cast<int>(fp16_supported)),
            gpu::make_jit_constant("FP16_UNIT_USED", static_cast<int>(data.fp16_unit_used)),
            gpu::make_jit_constant("UNIT_TYPE",      data.fp16_unit_used ? "half" : "float"),
            gpu::make_jit_constant("INPUT_PADDING",  _outer.input().at(input_idx)->desc()->output_padding()),
            gpu::make_jit_constant("OUTPUT_PADDING", _outer.argument.output_padding())
        };
    }

    cldnn::refcounted_obj_ptr<cldnn::event_impl> execute(const std::vector<cldnn::refcounted_obj_ptr<cldnn::event_impl>>& events) override
    {

        size_t inputs_count = _outer.argument.input().size();

        const auto& output_mem = _outer.output_memory();  // output

        uint32_t depth_offset = 0;
        auto tmp_events = events;
        for (size_t input_idx = 0; input_idx < inputs_count; ++input_idx)
        {
            const auto& kd = _kernels_with_data[input_idx].second;

            const auto& input_mem = _outer.input_memory(input_idx);  // current input

            uint32_t input_depth_count = input_mem.argument().size.feature[0];
            auto event = _kernels_with_data[input_idx].first.run<gpu::input_mem, gpu::output_mem, cl_uint>
                ({{kd.gws0, kd.gws1}, {kd.lws0, kd.lws1}}, tmp_events, input_mem, output_mem, depth_offset);
            depth_offset += input_depth_count;
            tmp_events.clear();
            tmp_events.push_back(event);
        }
        return tmp_events.at(0);
    }

    static is_an_implementation *create(depth_concatenate &arg) { return new depth_concatenate_gpu(arg); };
};

depth_concatenate_gpu::kernel_data default_yxfb(const std::pair<int, const depth_concatenate&>& arg)
{
    depth_concatenate_gpu::kernel_data kd = depth_concatenate_gpu::set_kernel_data(arg.first, arg.second);
    kd.gws1 = 1;
    kd.kernel_name = kernel_name_yxfb;
    return kd;
}

depth_concatenate_gpu::kernel_data default_bfyx(const std::pair<int, const depth_concatenate&>& arg)
{
    auto idx = arg.first;
    auto input_mem = arg.second.input_memory(idx);
    auto input_size = input_mem.argument().size;
    depth_concatenate_gpu::kernel_data kd = depth_concatenate_gpu::set_kernel_data(idx, arg.second);
    
    auto input_padding = arg.second.input().at(idx)->desc()->output_padding();
    auto output_padding = arg.second.argument.output_padding();
    // TODO: add support for f16 into this no padding kernel
    if (!input_padding && !output_padding && input_mem.get_layout().data_type == cldnn::data_types::f32)
    {
        kd.gws0 = input_size.batch[0];
        kd.gws1 = cldnn::align_to(input_mem.count() / input_mem.get_layout().size.batch[0] / 8, 16);

        kd.lws0 = 1;
        kd.lws1 = 16;

        kd.kernel_name = kernel_name_bfyx_no_padding;
    }
    else
    {
        kd.gws0 = input_size.batch[0];
        kd.gws1 = cldnn::align_to(input_size.feature[0] * input_size.spatial[1], 32);

        kd.lws0 = 1;
        kd.lws1 = 32;
        kd.kernel_name = kernel_name_bfyx;

    }
    return kd;
}

depth_concatenate_gpu::ks_type depth_concatenate_gpu::ks = {
    { std::make_tuple(memory::format::yxfb_f32, 0, gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), default_yxfb },
    { std::make_tuple(memory::format::yxfb_f16, 0, gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), default_yxfb },
    { std::make_tuple(memory::format::bfyx_f32, 0, gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), default_bfyx },
    { std::make_tuple(memory::format::bfyx_f16, 0, gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), default_bfyx }

};

namespace {
    struct attach {
        attach() {
            implementation_map<depth_concatenate>::add({
                { std::make_tuple(cldnn::engine_types::ocl, memory::format::yxfb_f32), depth_concatenate_gpu::create },
                { std::make_tuple(cldnn::engine_types::ocl, memory::format::yxfb_f16), depth_concatenate_gpu::create },
                { std::make_tuple(cldnn::engine_types::ocl, memory::format::bfyx_f32), depth_concatenate_gpu::create },
                { std::make_tuple(cldnn::engine_types::ocl, memory::format::bfyx_f16), depth_concatenate_gpu::create }
            });
        }
        ~attach() {}
    };
}
attach attach_impl;
} // namespace neural