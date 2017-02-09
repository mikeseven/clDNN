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

#include "neural_impl.h"
#include "engine_impl.h"
#include "network_impl.h"
#include "implementation_map.h"
#include "kernel.h"
#include "kd_selector.h"

#include <algorithm>
#include <stdexcept>
#include <string>


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

struct pooling_gpu : is_an_implementation {
    const pooling& _outer;
    gpu::engine_info_internal _engine_info;

    struct kernel_data 
    {
        size_t gws0, gws1, gws2; ///< Local work sizes (3D).
        size_t lws0, lws1, lws2; ///< Global work sizes (3D).
        std::string kernel_name;
        bool fp16_unit_used;
    } _kernel_data;
    gpu::kernel _kernel;

    static kd_selector_t<kernel_data, pooling, neural::memory::format::type, kd_optional_selector_t, int, neural::gpu::engine_info_internal::architectures, neural::gpu::engine_info_internal::configurations> ks;

    pooling_gpu(const pooling& outer)
        : _outer(outer),
        _engine_info(outer.get_network().get_engine()->get_context()->get_engine_info()),
        _kernel_data(ks.get_kernel(outer, outer.input_memory(0).argument().format, outer.input_memory(0).argument().size.batch[0], _engine_info.architecture, _engine_info.configuration)),
        _kernel(_outer.get_network().get_engine()->get_context(), _kernel_data.kernel_name, get_jit_constants(_outer, _kernel_data), _outer.id())
    {}

    static kernel_data set_default(const pooling& arg)
    {
        const auto& input_mem = arg.input_memory(0);  // input
        const auto& output_mem = arg.output_memory(); // output

        kernel_data kd;

        kd.fp16_unit_used = input_mem.get_layout().data_type == cldnn::data_types::f16;

        // Determine global work sizes.
        kd.gws0 = output_mem.argument().size.batch[0] * output_mem.argument().size.feature[0];
        kd.gws1 = output_mem.argument().size.spatial[0];
        kd.gws2 = output_mem.argument().size.spatial[1];

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
        switch (arg.argument.mode)
        {
        case cldnn::pooling_mode::max:
            kd.kernel_name = needs_boundary ? kernel_name_max_offset : kernel_name_max;
            break;
        case cldnn::pooling_mode::average:
            kd.kernel_name = needs_boundary ? kernel_name_average_offset : kernel_name_average;
            break;

        default:
            throw std::runtime_error("Unknown pooling mode.");
        }

        return kd;
    }

    // Checks if we need boundary checking in kernel.
    static bool needs_boundary_check(const pooling& outer)
    {
        auto& input_mem = outer.input_memory(0);
        auto input_offset = outer.desc()->input_offset().transform(input_mem.get_layout().size.format, 0);
        
        if (input_offset.spatial[0] || input_offset.spatial[1])
            return true;

        auto& kernel_size = outer.argument.size;
        auto& stride = outer.argument.stride;

        // If modulo is not 0 that means it is not dividable by stride, so we would go out of boundary.
        auto mod_x = (input_mem.argument().size.spatial[0] - (2 * input_offset.spatial[0]) - kernel_size.spatial[0]) % stride.spatial[0];
        auto mod_y = (input_mem.argument().size.spatial[1] - (2 * input_offset.spatial[1]) - kernel_size.spatial[1]) % stride.spatial[1];

        return mod_x || mod_y;
    }

    static gpu::jit_constants get_jit_constants(const pooling& outer, const kernel_data& data)
    {
        auto engine_info = outer.get_network().get_engine()->get_context()->get_engine_info();

        if (!engine_info.supports_fp16 && data.fp16_unit_used)
            throw std::invalid_argument("GPU device does not support half precision floating-point formats (cl_khr_fp16 extension)");

        auto input_size = outer.input().at(0)->non_padded_output_layout().size;
        gpu::jit_constants mem_consts{
            gpu::make_jit_constant("INPUT",             input_size),
            gpu::make_jit_constant("OUTPUT",            outer.non_padded_output_layout().size),
            gpu::make_jit_constant("WINDOW",            outer.argument.size),
            gpu::make_jit_constant("STRIDE",            outer.argument.stride),
            gpu::make_jit_constant("INPUT_OFFSET",      outer.desc()->input_offset()),
            gpu::make_jit_constant("FP16_SUPPORTED",    static_cast<int>(engine_info.supports_fp16)),
            gpu::make_jit_constant("FP16_UNIT_USED",    static_cast<int>(data.fp16_unit_used)),
            gpu::make_jit_constant("UNIT_TYPE",         data.fp16_unit_used ? "half" : "float"),
            gpu::make_jit_constant("UNIT_INIT_VAL_MAX", data.fp16_unit_used ? "-HALF_MAX" : "-FLT_MAX"),
            gpu::make_jit_constant("UNIT_INIT_VAL_AVG", data.fp16_unit_used ? "0.0h" : "0.0f"),
            gpu::make_jit_constant("OUTPUT_PADDING",    outer.argument.output_padding().size())
        };
        return mem_consts;
    }

    cldnn::refcounted_obj_ptr<cldnn::event_impl> execute(const std::vector<cldnn::refcounted_obj_ptr<cldnn::event_impl>>& events) override
    {
        const auto& outer = _outer;
        const auto& kd    = _kernel_data;

        const auto& input_mem  = outer.input_memory(0);  // input
        const auto& output_mem = outer.output_memory(); // output

        return _kernel.run<gpu::input_mem, gpu::output_mem>
          ({{kd.gws0, kd.gws1, kd.gws2}, {kd.lws0, kd.lws1, kd.lws2}}, events, input_mem, output_mem);
    }

    static is_an_implementation *create(pooling &arg) {
        auto input_arg = arg.input_memory(0).argument();
        auto output_arg = arg.output_memory().argument();

        auto& input_buffer_size = input_arg.size;
        auto& output_buffer_size = output_arg.size;
        auto& stride = arg.argument.stride;
        auto& window = arg.argument.size;
        auto padding = arg.desc()->padding_type();

        if (cldnn::padding::types::zero != padding)                                      throw std::logic_error("Pooling supports only zero padding.");
        if (input_buffer_size.raw.size() != output_buffer_size.raw.size()) throw std::invalid_argument("Pooling input/output number of dimension does not match.");
        if (stride.raw.size() != output_buffer_size.raw.size())            throw std::invalid_argument("Pooling stride/output number of dimension does not match.");
        if (window.raw.size() != output_buffer_size.raw.size())            throw std::invalid_argument("Pooling window_size/output number of dimension does not match.");
        if (input_arg.format != output_arg.format)                         throw std::invalid_argument("Pooling input/output data format does not match.");
        
        return new pooling_gpu(arg);
    }
};

pooling_gpu::kernel_data defauly_yxfb(const pooling& arg)
{
    pooling_gpu::kernel_data kd = pooling_gpu::set_default(arg);
    return kd;
}

pooling_gpu::kernel_data defauly_bfyx(const pooling& arg)
{
    pooling_gpu::kernel_data kd = pooling_gpu::set_default(arg);

    // Select kernel name.
    auto needs_boundary = pooling_gpu::needs_boundary_check(arg);
    if (needs_boundary && arg.argument.mode != cldnn::pooling_mode::max)
        throw std::runtime_error("Not implemented boundary in pooling bfyx!");
    //if (kd.gws0 > 256)
    {
        while (kd.gws0 % kd.lws0 != 0)
        {
            --kd.lws0;
        }
    }

    if (needs_boundary)
        kd.kernel_name = kernel_name_bfyx_max_offset;
    else if (arg.argument.mode == cldnn::pooling_mode::average)
        kd.kernel_name = kernel_name_bfyx_average;
    else
        kd.kernel_name = kernel_name_bfyx_max;

    auto output_size = arg.non_padded_output_layout().size;
    // Determine global work sizes.
    kd.gws2 = output_size.batch[0] * output_size.feature[0];
    kd.gws0 = output_size.spatial[0];
    kd.gws1 = output_size.spatial[1];

    // Find largest positive local work size that is divider for global work size.
    kd.lws0 = kd.gws0;
    kd.lws1 = 1;
    kd.lws2 = 1;

    return kd;
}

kd_selector_t<pooling_gpu::kernel_data, pooling, neural::memory::format::type, kd_optional_selector_t, int, neural::gpu::engine_info_internal::architectures, neural::gpu::engine_info_internal::configurations> pooling_gpu::ks = {
    { std::make_tuple(memory::format::yxfb_f32, 0, gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), defauly_yxfb },
    { std::make_tuple(memory::format::bfyx_f32, 0, gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), defauly_bfyx },
    { std::make_tuple(memory::format::yxfb_f16, 0, gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), defauly_yxfb },
    { std::make_tuple(memory::format::bfyx_f16, 0, gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), defauly_bfyx },
};

namespace
{

    struct attach
    {
        attach()
        {
            implementation_map<pooling>::add(std::make_tuple(cldnn::engine_types::ocl, memory::format::yxfb_f32), pooling_gpu::create);
            implementation_map<pooling>::add(std::make_tuple(cldnn::engine_types::ocl, memory::format::yxfb_f16), pooling_gpu::create);
            implementation_map<pooling>::add(std::make_tuple(cldnn::engine_types::ocl, memory::format::bfyx_f32), pooling_gpu::create);
            implementation_map<pooling>::add(std::make_tuple(cldnn::engine_types::ocl, memory::format::bfyx_f16), pooling_gpu::create);
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
