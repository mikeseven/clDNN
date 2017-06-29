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

#include "pooling_inst.h"
#include "kernel.h"
#include "kd_selector.h"
#include "implementation_map.h"

#include <algorithm>

using namespace cldnn;

namespace neural
{
// Kernel names.
static const std::string kernel_name_max            = "pooling_gpu_max";
static const std::string kernel_name_max_offset     = "pooling_gpu_max_offset";
static const std::string kernel_name_average        = "pooling_gpu_average";
static const std::string kernel_name_average_offset = "pooling_gpu_average_offset";
static const std::string kernel_name_bfyx_max       = "pooling_gpu_bfyx_max";
static const std::string kernel_name_bfyx_max_offset = "pooling_gpu_bfyx_max_offset";
static const std::string kernel_name_bfyx_average   = "pooling_gpu_bfyx_average";
static const std::string kernel_name_bfyx_average_offset = "pooling_gpu_bfyx_average_offset";
static const std::string kernel_name_bfyx_average_opt    = "pooling_gpu_bfyx_average_opt";

template <>
struct kd_default_value_selector<neural::gpu::engine_info_internal::architectures>
{
    static constexpr neural::gpu::engine_info_internal::architectures value = neural::gpu::engine_info_internal::architectures::GEN_UNKNOWN;
};

template <>
struct kd_default_value_selector<neural::gpu::engine_info_internal::configurations>
{
    static constexpr neural::gpu::engine_info_internal::configurations value = neural::gpu::engine_info_internal::configurations::GT_UNKNOWN;
};

struct pooling_gpu : typed_primitive_impl<pooling>
{
    const pooling_node& outer;
    gpu::engine_info_internal _engine_info;

    struct kernel_data 
    {
        size_t gws0, gws1, gws2; ///< Local work sizes (3D).
        size_t lws0, lws1, lws2; ///< Global work sizes (3D).
        std::string kernel_name;
        bool fp16_unit_used;
        size_t tile_height, tile_width;
    } _kernel_data;
    gpu::kernel _kernel;

    static kd_selector_t<kernel_data, pooling_node, data_types, format::type, kd_optional_selector_t, int, neural::gpu::engine_info_internal::architectures, neural::gpu::engine_info_internal::configurations> ks;

    pooling_gpu(const pooling_node& arg)
        : outer(arg),
        _engine_info(outer.get_program().get_engine()->get_context()->get_engine_info()),
        _kernel_data(ks.get_kernel(
            outer,
            outer.input().get_output_layout().data_type,
            outer.input().get_output_layout().format,
            outer.input().get_output_layout().size.batch[0],
            _engine_info.architecture,
            _engine_info.configuration)),
        _kernel(outer.get_program().get_engine()->get_context(), _kernel_data.kernel_name, get_jit_constants(outer, _kernel_data), outer.id())
    {}

    static kernel_data set_default(const pooling_node& arg)
    {
        auto input_layout = arg.input().get_output_layout();  // input
        auto const& output_buffer_size = arg.get_output_layout().get_buffer_size();

        kernel_data kd;

        kd.fp16_unit_used = input_layout.data_type == cldnn::data_types::f16;

        // Determine global work sizes.
        kd.gws0 = output_buffer_size.batch[0] * output_buffer_size.feature[0];
        kd.gws1 = output_buffer_size.spatial[0];
        kd.gws2 = output_buffer_size.spatial[1];

        // Find largest positive local work size that is divider for global work size.
        kd.lws0 = std::min(std::max(kd.gws0, static_cast<size_t>(1)), static_cast<size_t>(32));
        while (kd.gws0 % kd.lws0 != 0)
        {
            --kd.lws0;
        }
        kd.lws1 = 1;
        kd.lws2 = 1;

        // Select kernel name.
        auto needs_boundary = needs_boundary_check(arg);
        switch (arg.get_primitive()->mode)
        {
        case cldnn::pooling_mode::max:
            kd.kernel_name = needs_boundary ? kernel_name_max_offset : kernel_name_max;
            break;
        case cldnn::pooling_mode::average:
        case cldnn::pooling_mode::average_no_padding:
            kd.kernel_name = needs_boundary ? kernel_name_average_offset : kernel_name_average;
            break;

        default:
            throw std::runtime_error("Unknown pooling mode.");
        }

        kd.tile_height = 0;
        kd.tile_width = 0;

        return kd;
    }

    // Checks if we need boundary checking in kernel.
    static bool needs_boundary_check(const pooling_node& outer)
    {
        auto input_size = outer.input().get_output_layout().size;
        auto input_offset = outer.get_primitive()->input_offset;
        
        if (input_offset.spatial[0] || input_offset.spatial[1])
            return true;

        auto& kernel_size = outer.get_primitive()->size;

        if ((kernel_size.spatial[0] > input_size.spatial[0]) || (kernel_size.spatial[1] > input_size.spatial[1]))
        {
            return true;
        }
        
        auto& stride = outer.get_primitive()->stride;

        // If modulo is not 0 that means it is not dividable by stride, so we would go out of boundary.
        auto mod_x = (input_size.spatial[0] - (2 * input_offset.spatial[0]) - kernel_size.spatial[0]) % stride.spatial[0];
        auto mod_y = (input_size.spatial[1] - (2 * input_offset.spatial[1]) - kernel_size.spatial[1]) % stride.spatial[1];

        return mod_x || mod_y;
    }

    static gpu::jit_constants get_jit_constants(const pooling_node& outer, const kernel_data& data)
    {
        auto engine_info = outer.get_program().get_engine()->get_context()->get_engine_info();

        if (!engine_info.supports_fp16 && data.fp16_unit_used)
            throw std::invalid_argument("GPU device does not support half precision floating-point formats (cl_khr_fp16 extension)");

        auto input_layout = outer.input().get_output_layout();
        auto output_layout = outer.get_output_layout();
        auto input_padding = outer.input().get_output_layout().data_padding;
        auto output_padding = outer.get_output_layout().data_padding;
        auto input_size = input_layout.size;

        gpu::jit_constants mem_consts{
            gpu::make_jit_constant("INPUT",             input_size),
            gpu::make_jit_constant("OUTPUT",            output_layout.size),
            gpu::make_jit_constant("WINDOW",            outer.get_primitive()->size),
            gpu::make_jit_constant("STRIDE",            outer.get_primitive()->stride),
            gpu::make_jit_constant("INPUT_OFFSET",      outer.get_primitive()->input_offset),
            gpu::make_jit_constant("FP16_SUPPORTED",    static_cast<int>(engine_info.supports_fp16)),
            gpu::make_jit_constant("FP16_UNIT_USED",    static_cast<int>(data.fp16_unit_used)),
            gpu::make_jit_constant("UNIT_TYPE",         data.fp16_unit_used ? "half" : "float"),
            gpu::make_jit_constant("UNIT_INIT_VAL_MAX", data.fp16_unit_used ? "-HALF_MAX" : "-FLT_MAX"),
            gpu::make_jit_constant("UNIT_INIT_VAL_AVG", data.fp16_unit_used ? "0.0h" : "0.0f"),
            gpu::make_jit_constant("INPUT_PADDING",     input_padding),
            gpu::make_jit_constant("OUTPUT_PADDING",    output_padding),
            gpu::make_jit_constant("DYNAMIC_AVERAGE",   outer.get_primitive()->mode == pooling_mode::average_no_padding ? 1 : 0)
        };

        if (data.kernel_name == kernel_name_bfyx_average_opt)
        {
            mem_consts.add_constant(gpu::make_jit_constant("SUB_GROUP_SIZE", data.lws0));
            mem_consts.add_constant(gpu::make_jit_constant("TILE_HEIGHT", data.tile_height));
            mem_consts.add_constant(gpu::make_jit_constant("TILE_WIDTH", data.tile_width));
            mem_consts.add_constant(gpu::make_jit_constant("ONE_OVER_POOL_SIZE", 1.f / (outer.get_primitive()->size.spatial[0] * outer.get_primitive()->size.spatial[1])));
        }

        return mem_consts;
    }

    event_impl::ptr execute_impl(const std::vector<event_impl::ptr>& events, pooling_inst& instance) override
    {
        const auto& kd    = _kernel_data;

        return _kernel.run<gpu::input_mem, gpu::output_mem>
          ({{ kd.gws0, kd.gws1, kd.gws2 }, { kd.lws0, kd.lws1, kd.lws2 }},
              events,
              instance.input_memory(),
              instance.output_memory());
    }

    static primitive_impl* create(const pooling_node& arg)
    {
        auto const& input_buffer_size = arg.input().get_output_layout().get_buffer_size();
        auto const& input_dimensions = input_buffer_size.batch.size() + input_buffer_size.feature.size() + input_buffer_size.spatial.size();
        auto const& output_buffer_size = arg.get_output_layout().get_buffer_size();
        auto const& output_dimensions = output_buffer_size.batch.size() + output_buffer_size.feature.size() + output_buffer_size.spatial.size();
        auto const& input_format = arg.input().get_output_layout().format;
        auto const& output_format = arg.get_output_layout().format;
        auto& stride = arg.get_primitive()->stride;
        auto const& stride_dimensions = stride.batch.size() + stride.feature.size() + stride.spatial.size();
        auto& window = arg.get_primitive()->size;
        auto const& window_dimensions = window.batch.size() + window.feature.size() + window.spatial.size();

        if (input_dimensions != output_dimensions)
            throw std::invalid_argument("Pooling input/output number of dimension does not match.");

        if (stride_dimensions != output_dimensions)
            throw std::invalid_argument("Pooling stride/output number of dimension does not match.");

        if (window_dimensions != output_dimensions)
            throw std::invalid_argument("Pooling window_size/output number of dimension does not match.");

        if (input_format != output_format)
            throw std::invalid_argument("Pooling input/output data format does not match.");
        
        return new pooling_gpu(arg);
    }
};

pooling_gpu::kernel_data defauly_yxfb(const pooling_node& arg)
{
    pooling_gpu::kernel_data kd = pooling_gpu::set_default(arg);
    return kd;
}

pooling_gpu::kernel_data defauly_bfyx(const pooling_node& arg)
{
    pooling_gpu::kernel_data kd = pooling_gpu::set_default(arg);

    auto const& prim = arg.get_primitive();
    auto const& input_layout = arg.input().get_output_layout();
    auto const& output_layout = arg.get_output_layout();

    // Optimized sub-groups kernel for specific use-case:
    // AVG pooling with 3x3 kernel, 1x1 stride, -1x-1 input offset, no input/output padding, input size = output size, batch=1, FP32.
    if ((prim->mode == cldnn::pooling_mode::average) &&
        !kd.fp16_unit_used &&
        (prim->size.spatial[0] == 3) &&
        (prim->size.spatial[1] == 3) &&
        (prim->stride.spatial[0] == 1) &&
        (prim->stride.spatial[1] == 1) &&
        (prim->input_offset.spatial[0] == -1) &&
        (prim->input_offset.spatial[1] == -1) &&
        !input_layout.data_padding &&
        !output_layout.data_padding &&
        (input_layout.size == output_layout.size) &&
        (input_layout.size.batch[0] == 1))
    {
        kd.kernel_name = kernel_name_bfyx_average_opt;
    
        const int simd_size = 16;     
        kd.tile_height = 7;
        kd.tile_width = simd_size - 2;
        
        const int num_tiles_x = static_cast<int>(std::ceil(static_cast<float>(input_layout.size.spatial[0]) / static_cast<float>(kd.tile_width)));
        const int num_tiles_y = static_cast<int>(std::ceil(static_cast<float>(input_layout.size.spatial[1]) / static_cast<float>(kd.tile_height)));
        
        kd.gws0 = num_tiles_x * simd_size;
        kd.gws1 = num_tiles_y;
        kd.gws2 = input_layout.size.feature[0];
        kd.lws0 = simd_size;
        kd.lws1 = 1;
        kd.lws2 = 1;
    
        return kd;
    }

    // Select kernel name.
    auto needs_boundary = pooling_gpu::needs_boundary_check(arg);
    //if (kd.gws0 > 256)
    {
        while (kd.gws0 % kd.lws0 != 0)
        {
            --kd.lws0;
        }
    }

    if (needs_boundary)
    {
        kd.kernel_name = cldnn::pooling_mode::max == arg.get_primitive()->mode ? kernel_name_bfyx_max_offset : kernel_name_bfyx_average_offset;
    }
    else
    {
        kd.kernel_name = cldnn::pooling_mode::max == arg.get_primitive()->mode ? kernel_name_bfyx_max : kernel_name_bfyx_average;
    }

    auto const& output_size = output_layout.size;
    // Determine global work sizes.
    kd.gws2 = output_size.batch[0] * output_size.feature[0];
    kd.gws0 = cldnn::align_to(output_size.spatial[0], 32);
    kd.gws1 = output_size.spatial[1];

    // Find largest positive local work size that is divider for global work size.
    kd.lws0 = 32;
    kd.lws1 = 1;
    kd.lws2 = 1;

    return kd;
}

kd_selector_t<pooling_gpu::kernel_data, pooling_node, data_types, format::type, kd_optional_selector_t, int, neural::gpu::engine_info_internal::architectures, neural::gpu::engine_info_internal::configurations> pooling_gpu::ks = {
    { std::make_tuple(data_types::f32, format::yxfb, 0, gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), defauly_yxfb },
    { std::make_tuple(data_types::f32, format::bfyx, 0, gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), defauly_bfyx },
    { std::make_tuple(data_types::f16, format::yxfb, 0, gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), defauly_yxfb },
    { std::make_tuple(data_types::f16, format::bfyx, 0, gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), defauly_bfyx },
};

namespace
{

    struct attach
    {
        attach()
        {
            implementation_map<pooling>::add(std::make_tuple(cldnn::engine_types::ocl, data_types::f32, format::yxfb), pooling_gpu::create);
            implementation_map<pooling>::add(std::make_tuple(cldnn::engine_types::ocl, data_types::f16, format::yxfb), pooling_gpu::create);
            implementation_map<pooling>::add(std::make_tuple(cldnn::engine_types::ocl, data_types::f32, format::bfyx), pooling_gpu::create);
            implementation_map<pooling>::add(std::make_tuple(cldnn::engine_types::ocl, data_types::f16, format::bfyx), pooling_gpu::create);
        }
        ~attach() {}
    };
    attach attach_impl;
}
}
