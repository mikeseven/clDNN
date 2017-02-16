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
#include "network_impl.h"
#include "implementation_map.h"
#include "kernel.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace neural
{
// Kernel names.
static const std::string kernel_name = "eltwise_gpu";

struct eltwise_gpu : is_an_implementation
{
    const eltwise& _outer;
    struct kernel_data
    {
        size_t gws0;
        size_t lws0;
        std::string kernel_name;
        bool fp16_unit_used;
        bool bfyx_mean_format_used;
    } _kernel_data;
    gpu::kernel _kernel;

    eltwise_gpu(const eltwise& outer)
        : _outer(outer),
        _kernel_data(set_kernel_data(_outer)),
        _kernel(_outer.get_network().get_engine()->get_context(), _kernel_data.kernel_name, get_jit_constants(_outer, _kernel_data), _outer.id())
    {}

    static kernel_data set_kernel_data(const eltwise& outer)
    {
        const auto& output_mem = outer.output_memory(); // output

        if (outer.input().at(0)->desc()->output_padding() ||
            outer.input().at(1)->desc()->output_padding() ||
            outer.desc()->input_padding())
        {
            throw std::runtime_error("Input padding for eltwise not yet supported");
        }

        kernel_data kd;

        kd.fp16_unit_used = output_mem.get_layout().data_type == cldnn::data_types::f16;

        // Determine global work sizes.
        kd.gws0 = output_mem.count();
        // Find largest positive local work size that is divider for global work size.
        kd.lws0 = std::min(std::max(kd.gws0, static_cast<size_t>(1)), static_cast<size_t>(32));
        while (kd.gws0 % kd.lws0 != 0)
        {
            --kd.lws0;
        }

        kd.kernel_name = kernel_name;

        return kd;
    }

    static gpu::jit_constants get_jit_constants(const eltwise& outer, const kernel_data& data)
    {
        auto engine_info = outer.get_network().get_engine()->get_context()->get_engine_info();

        if (!engine_info.supports_fp16 && data.fp16_unit_used)
            throw std::invalid_argument("GPU device does not support half precision floating-point formats (cl_khr_fp16 extension)");

        return{
            gpu::make_jit_constant("INPUT",                 outer.input_memory(0).argument().size),
            gpu::make_jit_constant("OUTPUT",                outer.non_padded_output_layout().size),
            gpu::make_jit_constant("INPUT2" ,               outer.input_memory(1).argument().size),
            gpu::make_jit_constant("OUTPUT_PADDING",        outer.argument.output_padding().size()),
            gpu::make_jit_constant("FP16_SUPPORTED",        static_cast<int>(engine_info.supports_fp16)),
            gpu::make_jit_constant("FP16_UNIT_USED",        static_cast<int>(data.fp16_unit_used)),
            gpu::make_jit_constant("UNIT_TYPE",             data.fp16_unit_used ? "half" : "float"),
            gpu::make_jit_constant("SUM_MODE_USED",         outer.argument.mode == cldnn::eltwise_mode::sum ? 1 : 0),
            gpu::make_jit_constant("MAX_MODE_USED",         outer.argument.mode == cldnn::eltwise_mode::max ? 1 : 0),
            gpu::make_jit_constant("SUB_MODE_USED",         outer.argument.mode == cldnn::eltwise_mode::sub ? 1 : 0),
            gpu::make_jit_constant("PROD_MODE_USED",        outer.argument.mode == cldnn::eltwise_mode::prod ? 1 : 0),
            gpu::make_jit_constant("RELU",                  static_cast<int>(outer.argument.with_activation)),
            gpu::make_jit_constant("NEGATIVE_SLOPE",        outer.argument.activation_negative_slope),
        };
    }

    cldnn::refcounted_obj_ptr<cldnn::event_impl> execute(const std::vector<cldnn::refcounted_obj_ptr<cldnn::event_impl>>& events) override
    {
        const auto& kd    = _kernel_data;

        const auto& input_mem  = _outer.input_memory(0);  // input
        const auto& input2_mem   = _outer.input_memory(1);  // input2
        const auto& output_mem = _outer.output_memory(); // output

        // input2_mem memory in bfyx or yxfb.
        return _kernel.run<gpu::input_mem, gpu::output_mem, gpu::input_mem>({kd.gws0, kd.lws0}, events, input_mem, output_mem, input2_mem);
    }

    static is_an_implementation *create(const eltwise& outer) { return new eltwise_gpu(outer); }

};

namespace {
    struct attach {
        attach() {
            implementation_map<eltwise>::add({
                { std::make_tuple(cldnn::engine_types::ocl, memory::format::yxfb_f32), eltwise_gpu::create },
                { std::make_tuple(cldnn::engine_types::ocl, memory::format::yxfb_f16), eltwise_gpu::create },
                { std::make_tuple(cldnn::engine_types::ocl, memory::format::bfyx_f32), eltwise_gpu::create },
                { std::make_tuple(cldnn::engine_types::ocl, memory::format::bfyx_f16), eltwise_gpu::create }
            });
        }
        ~attach() {}
    };
    attach attach_impl;
}
}
