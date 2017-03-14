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
#include <cmath>
#include <stdexcept>
#include <string>
#include "api/primitives/data.hpp"

namespace neural
{
// Kernel names.
static const std::string kernel_name = "lrn_gpu";
static const std::string kernel_name_b8 = "lrn_gpu_b8";
static const std::string kernel_name_bfyx = "lrn_gpu_bfyx";
static const std::string kernel_name_within_channel_bfyx = "lrn_gpu_within_channel_bfyx";
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


struct lrn_gpu : is_an_implementation
{
    const normalization::response& _outer;
    gpu::engine_info_internal _engine_info;

    struct kernel_data
    {
        size_t gws0, gws1, gws2;
        size_t lws0, lws1, lws2;
        std::string kernel_name;
        bool fp16_unit_used;
    } _kernel_data;
    gpu::kernel _kernel;

    static kd_selector_t<kernel_data, normalization::response, neural::memory::format::type, kd_optional_selector_t, int, neural::gpu::engine_info_internal::architectures, neural::gpu::engine_info_internal::configurations> ks;

    lrn_gpu(const normalization::response& outer)
        : _outer(outer),
        _engine_info(outer.get_network().get_engine()->get_context()->get_engine_info()),
        _kernel_data(ks.get_kernel(outer, outer.input_memory(0).argument().format, outer.input_memory(0).argument().size.batch[0], _engine_info.architecture, _engine_info.configuration)),
        _kernel(_outer.get_network().get_engine()->get_context(), _kernel_data.kernel_name, get_jit_constants(_outer, _kernel_data), _outer.id())
    {}

    static kernel_data set_default(const normalization::response& arg)
    {
        const auto& input_mem = arg.input_memory(0);  // input

        kernel_data kd;

        kd.fp16_unit_used = input_mem.get_layout().data_type == cldnn::data_types::f16;

        // Determine global work sizes.
        kd.gws0 = input_mem.argument().size.batch[0] * input_mem.argument().size.feature[0];   // B, F
        kd.gws1 = input_mem.argument().size.spatial[0] * input_mem.argument().size.spatial[1]; // X, Y
        kd.gws2 = 1;
        // Find largest positive local work size that is divider for global work size.
        kd.lws0 = std::min(std::max(kd.gws0, static_cast<size_t>(1)), static_cast<size_t>(32));
        while (kd.gws0 % kd.lws0 != 0)
        {
            --kd.lws0;
        }
        kd.lws1 = 1;
        kd.lws2 = 1;

        if (arg.argument.norm_region == cldnn_lrn_norm_region_across_channel)
        {
            // TODO: add half case: b16 (b*f dividable by 128).
            if (!kd.fp16_unit_used &&                        // halfs are not used
                input_mem.argument().size.batch[0] % 8 == 0 && // batch_num is multiple of 8
                kd.gws0 % 64 == 0)                           // batch_num * feature_num is multiple of 64
            {
                kd.gws0 /= 8;
                kd.lws0 = 8; // gws0 is dividable by 64, so after correction it will be dividable by 8.

                kd.kernel_name = kernel_name_b8;
            }
            else
            {
                kd.kernel_name = kernel_name;
            }
        }
        else if (arg.argument.norm_region == cldnn_lrn_norm_region_within_channel)
        {
            kd.kernel_name = kernel_name_within_channel_bfyx;
        }
        else
        {
            throw std::runtime_error("Invalid norm region");
        }

        // Checking for supported paddings.
        if (arg.desc()->input_padding().filling_value() != 0.0f)
            throw std::runtime_error("Unknown padding mode in lrn");

        return kd;
    }

    static gpu::jit_constants get_jit_constants(const normalization::response& outer, const kernel_data& data)
    {
        auto engine_info = outer.get_network().get_engine()->get_context()->get_engine_info();

        if (!engine_info.supports_fp16 && data.fp16_unit_used)
            throw std::invalid_argument("GPU device does not support half precision floating-point formats (cl_khr_fp16 extension)");

        int size = outer.argument.size;
        int pad = (size - 1) / 2;

        //alpha_div_by_size is used for norm. region: within channels, alpha is used for norm. region: across channel
        auto alpha = outer.argument.alpha;
        auto alpha_div_by_size = outer.argument.alpha / outer.argument.size;
        auto alpha_sign = std::signbit(alpha) ? -1.0f : 1.0f;
        // When used FP16 the value cannot be scaled afterwards by alpha (it must be scaled before computing sum of squares).
        auto alpha_abs_sqrt = std::sqrt(std::abs(alpha));
        auto alpha_div_by_size_abs_sqrt = std::sqrt(std::abs(alpha_div_by_size));

        auto input_padding = outer.argument.input_padding();
        if (input_padding)
        {
            throw std::runtime_error("input padding not implemented in LRN yet!");
        }

        auto input_size = outer.input().at(0)->non_padded_output_layout().size;

        int count = input_size.sizes()[0] * input_size.sizes()[1] * input_size.sizes()[2] * input_size.sizes()[3];

        gpu::jit_constants mem_consts {
            gpu::make_jit_constant("INPUT",                         input_size),
            gpu::make_jit_constant("COUNT",                         count),
            gpu::make_jit_constant("OUTPUT",                        outer.non_padded_output_layout().size),
            gpu::make_jit_constant("P_SIZE",                        size),
            gpu::make_jit_constant("PAD",                           pad),
            gpu::make_jit_constant("ALPHA",                         data.fp16_unit_used ? alpha_sign : alpha),
            gpu::make_jit_constant("ALPHA_DIV_BY_SIZE",             data.fp16_unit_used ? alpha_sign : alpha_div_by_size),
            gpu::make_jit_constant("ALPHA_VAL_FACTOR",              data.fp16_unit_used ? alpha_abs_sqrt : 1.0f),
            gpu::make_jit_constant("ALPHA_VAL_FACTOR_DIV_BY_SIZE",  data.fp16_unit_used ? alpha_div_by_size_abs_sqrt : 1.0f),
            gpu::make_jit_constant("BETA",                          outer.argument.beta),
            gpu::make_jit_constant("K",                             outer.argument.k),
            gpu::make_jit_constant("HELP_INPUT_OFFSET",             outer.desc()->input_offset().feature[0] - static_cast<int32_t>(size / 2)),
            gpu::make_jit_constant("FP16_SUPPORTED",                static_cast<int>(engine_info.supports_fp16)),
            gpu::make_jit_constant("FP16_UNIT_USED",                static_cast<int>(data.fp16_unit_used)),
            gpu::make_jit_constant("UNIT_TYPE",                     data.fp16_unit_used ? "half" : "float"),
            gpu::make_jit_constant("UNIT_VAL_ZERO",                 data.fp16_unit_used ? "0.0h" : "0.0f"),
            gpu::make_jit_constant("INPUT_PADDING",                 outer.argument.input_padding()),
            gpu::make_jit_constant("OUTPUT_PADDING",                outer.argument.output_padding())
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


    static is_an_implementation *create(normalization::response &arg) { return new lrn_gpu(arg); }

};

lrn_gpu::kernel_data default_yxfb(const normalization::response& arg)
{
    if (arg.argument.norm_region == cldnn_lrn_norm_region_within_channel)
    {
        throw std::runtime_error("LRN within channel is not implemented for YXFB format");
    }

    lrn_gpu::kernel_data kd = lrn_gpu::set_default(arg);
    return kd;
}


lrn_gpu::kernel_data default_bfyx_within_channel(const normalization::response& arg)
{
    lrn_gpu::kernel_data kd = lrn_gpu::set_default(arg);

    kd.kernel_name = kernel_name_within_channel_bfyx;
   
    kd.gws0 = 128 * 128;
    kd.gws1 = 1;
    kd.gws2 = 1;

    kd.lws0 = 128;
    kd.lws1 = 1;
    kd.lws2 = 1;
    return kd;
}

lrn_gpu::kernel_data default_bfyx_across_channel(const normalization::response& arg)
{
    auto& input_mem = arg.input_memory(0);
    lrn_gpu::kernel_data kd = lrn_gpu::set_default(arg);

    kd.kernel_name = kernel_name_bfyx;
   
    kd.gws0 = cldnn::align_to(input_mem.argument().size.spatial[0],32);
    kd.gws1 = input_mem.argument().size.spatial[1];
    kd.gws2 = input_mem.argument().size.feature[0] * input_mem.argument().size.batch[0];

    kd.lws0 = 32;
    kd.lws1 = 1;
    kd.lws2 = 1;
    return kd;
}


lrn_gpu::kernel_data default_bfyx(const normalization::response& arg)
{
    switch (arg.argument.norm_region)
    {
        case cldnn_lrn_norm_region_across_channel: 
            return default_bfyx_across_channel(arg);
        case cldnn_lrn_norm_region_within_channel:
            return default_bfyx_within_channel(arg);
        default:
            throw std::runtime_error("Invalid norm region");
    }
}

kd_selector_t<lrn_gpu::kernel_data, normalization::response, neural::memory::format::type, kd_optional_selector_t, int, neural::gpu::engine_info_internal::architectures, neural::gpu::engine_info_internal::configurations> lrn_gpu::ks = {
    { std::make_tuple(memory::format::yxfb_f32, 0, gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), default_yxfb },
    { std::make_tuple(memory::format::yxfb_f16, 0, gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), default_yxfb },
    { std::make_tuple(memory::format::bfyx_f32, 0, gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), default_bfyx },
    { std::make_tuple(memory::format::bfyx_f16, 0, gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), default_bfyx },
};


namespace {
    struct attach {
        attach() {
            implementation_map<normalization::response>::add(std::make_tuple(cldnn::engine_types::ocl, memory::format::yxfb_f32), lrn_gpu::create);
            implementation_map<normalization::response>::add(std::make_tuple(cldnn::engine_types::ocl, memory::format::yxfb_f16), lrn_gpu::create);
            implementation_map<normalization::response>::add(std::make_tuple(cldnn::engine_types::ocl, memory::format::bfyx_f32), lrn_gpu::create);
            implementation_map<normalization::response>::add(std::make_tuple(cldnn::engine_types::ocl, memory::format::bfyx_f16), lrn_gpu::create);
        }
        ~attach() {}
    };
    attach attach_impl;
}
}
