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

#include "depth_concatenate_inst.h"
#include "kernel.h"
#include "kd_selector.h"
#include "implementation_map.h"

#include <initializer_list>

using namespace cldnn;

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

struct depth_concatenate_gpu : typed_primitive_impl<depth_concatenate>
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

    const depth_concatenate_node& outer;
    gpu::engine_info_internal _engine_info;

    std::vector<std::pair<gpu::kernel, kernel_data>> _kernels_with_data;

    typedef kd_selector_t<kernel_data, std::pair<int, const depth_concatenate_node &>, data_types, format::type, kd_optional_selector_t, int, neural::gpu::engine_info_internal::architectures, gpu::engine_info_internal::configurations> ks_type;
    static ks_type ks;

    depth_concatenate_gpu(const depth_concatenate_node& outer)
        : outer(outer)
        , _engine_info(outer.get_program().get_engine()->get_context()->get_engine_info())
    {
        auto context = outer.get_program().get_engine()->get_context();

        const int inputs_count = static_cast<int>(outer.inputs_count());

        _kernels_with_data.reserve(inputs_count);
        for (auto input_idx = 0; input_idx < inputs_count; ++input_idx)
        {
            auto input_layout = outer.input(input_idx).get_output_layout();
            auto data = ks.get_kernel(std::make_pair(input_idx, std::cref(outer)), input_layout.data_type, input_layout.size.format, input_layout.size.batch[0], _engine_info.architecture, _engine_info.configuration);//set_kernel_data(/*input_idx,*/ _outer/*, engine_info*/);
            gpu::kernel kernel(context, data.kernel_name, get_jit_constants(input_idx, data), outer.id());

            _kernels_with_data.emplace_back(std::move(kernel), std::move(data));
        }
    }

    static kernel_data set_kernel_data(int input_idx, const depth_concatenate_node& outer)
    {
        const auto& input_layout = outer.input(input_idx).get_output_layout();  // current input
        const auto& input_buffer_size = input_layout.get_buffer_size();

        kernel_data kd;

        kd.fp16_unit_used = input_layout.data_type == cldnn::data_types::f16;

        // Determine global work sizes.
        kd.gws0 = input_buffer_size.spatial[0] * input_buffer_size.spatial[1];
        kd.gws1 = input_buffer_size.batch[0];
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

        auto input_layout = outer.input(input_idx).get_output_layout();
        auto input_padding = outer.input(input_idx).get_output_layout().data_padding;
        auto output_padding = outer.get_output_layout().data_padding;
        auto input_buffer_size = outer.input(input_idx).get_output_layout().get_buffer_size();

        return gpu::jit_constants{
            gpu::make_jit_constant("INPUT",          input_layout.size),
            gpu::make_jit_constant("OUTPUT",         outer.get_output_layout().size),
            gpu::make_jit_constant("INPUT_ELEMENTS_COUNT", input_buffer_size.count() / input_buffer_size.batch[0]),
            gpu::make_jit_constant("FP16_SUPPORTED", static_cast<int>(fp16_supported)),
            gpu::make_jit_constant("FP16_UNIT_USED", static_cast<int>(data.fp16_unit_used)),
            gpu::make_jit_constant("UNIT_TYPE",      data.fp16_unit_used ? "half" : "float"),
            gpu::make_jit_constant("INPUT_PADDING",  input_padding),
            gpu::make_jit_constant("OUTPUT_PADDING", output_padding)
        };
    }

    event_impl::ptr execute_impl(const std::vector<event_impl::ptr>& events, depth_concatenate_inst& instance) override
    {
        size_t inputs_count = outer.inputs_count();

        const auto& output_mem = instance.output_memory();  // output

        uint32_t depth_offset = 0;
        auto tmp_events = events;
        for (size_t input_idx = 0; input_idx < inputs_count; ++input_idx)
        {
            const auto& kd = _kernels_with_data[input_idx].second;

            const auto& input_mem = instance.input_memory(input_idx);  // current input

            uint32_t input_depth_count = input_mem.get_layout().size.feature[0];
            auto event = _kernels_with_data[input_idx].first.run<gpu::input_mem, gpu::output_mem, cl_uint>
                ({ { kd.gws0, kd.gws1 },{ kd.lws0, kd.lws1 } }, tmp_events, input_mem, output_mem, depth_offset);
            depth_offset += input_depth_count;
            tmp_events.clear();
            tmp_events.push_back(event);
        }
        return tmp_events.at(0);
    }

    static primitive_impl* create(const depth_concatenate_node& arg) { return new depth_concatenate_gpu(arg); };
};

depth_concatenate_gpu::kernel_data default_yxfb(const std::pair<int, const depth_concatenate_node&>& arg)
{
    depth_concatenate_gpu::kernel_data kd = depth_concatenate_gpu::set_kernel_data(arg.first, arg.second);
    kd.gws1 = 1;
    kd.kernel_name = kernel_name_yxfb;
    return kd;
}

depth_concatenate_gpu::kernel_data default_bfyx(const std::pair<int, const depth_concatenate_node&>& arg)
{
    auto idx = arg.first;
    auto input_layout = arg.second.input(idx).get_output_layout();
    auto input_buffer_size = input_layout.get_buffer_size();

    depth_concatenate_gpu::kernel_data kd = depth_concatenate_gpu::set_kernel_data(idx, arg.second);

    auto input_padding = arg.second.input(idx).get_output_layout().data_padding;
    auto output_padding = arg.second.get_output_layout().data_padding;

    // TODO: add support for f16 into this no padding kernel
    if (!input_padding && !output_padding && input_layout.data_type == cldnn::data_types::f32)
    {
        kd.gws0 = input_buffer_size.batch[0];
        kd.gws1 = align_to(input_layout.count() / input_buffer_size.batch[0] / 8, 16);

        kd.lws0 = 1;
        kd.lws1 = 16;

        kd.kernel_name = kernel_name_bfyx_no_padding;
    }
    else
    {
        kd.gws0 = input_buffer_size.batch[0];
        kd.gws1 = align_to(input_buffer_size.feature[0] * input_buffer_size.spatial[1], 32);

        kd.lws0 = 1;
        kd.lws1 = 32;
        kd.kernel_name = kernel_name_bfyx;

    }
    return kd;
}

depth_concatenate_gpu::ks_type depth_concatenate_gpu::ks = {
    { std::make_tuple(data_types::f32, format::yxfb, 0, gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), default_yxfb },
    { std::make_tuple(data_types::f16, format::yxfb, 0, gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), default_yxfb },
    { std::make_tuple(data_types::f32, format::bfyx, 0, gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), default_bfyx },
    { std::make_tuple(data_types::f16, format::bfyx, 0, gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), default_bfyx }
};

namespace {
    struct attach {
        attach() {
            implementation_map<depth_concatenate>::add({
                { std::make_tuple(cldnn::engine_types::ocl, data_types::f32, format::yxfb), depth_concatenate_gpu::create },
                { std::make_tuple(cldnn::engine_types::ocl, data_types::f16, format::yxfb), depth_concatenate_gpu::create },
                { std::make_tuple(cldnn::engine_types::ocl, data_types::f32, format::bfyx), depth_concatenate_gpu::create },
                { std::make_tuple(cldnn::engine_types::ocl, data_types::f16, format::bfyx), depth_concatenate_gpu::create }
            });
        }
        ~attach() {}
    };
}

attach attach_impl;

} // namespace neural
