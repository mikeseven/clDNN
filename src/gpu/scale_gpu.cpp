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

#include "scale_inst.h"
#include "kernel.h"
#include "kd_selector.h"
#include "network_impl.h"
#include "implementation_map.h"

#include <algorithm>
#include <stdexcept>
#include <string>

using namespace cldnn;


namespace neural
{
// Kernel names.
static const std::string kernel_name = "scale_gpu";
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

struct scale_gpu : primitive_impl
{
    const scale_inst& _outer;
    gpu::engine_info_internal _engine_info;

    struct kernel_data
    {
        size_t gws0, gws1, gws2;
        size_t lws0, lws1, lws2;
        std::string kernel_name;
        bool fp16_unit_used;
        bool fp16_supported;
        bool input_bfyx_used; ///< Indicates that bfyx format of input is used.
        bool scale_bfyx_used; ///< Indicates that bfyx format of scale is used.
    } _kernel_data;
    gpu::kernel _kernel;

    static kd_selector_t<kernel_data, scale_inst, data_types, format::type, kd_optional_selector_t, neural::gpu::engine_info_internal::architectures, neural::gpu::engine_info_internal::configurations> ks;

    scale_gpu(const scale_inst& outer) :
        _outer(outer),
        _engine_info(outer.get_network().get_engine()->get_context()->get_engine_info()),
        _kernel_data(ks.get_kernel(outer, outer.input_memory().get_layout().data_type, outer.input_memory().get_layout().size.format, _engine_info.architecture, _engine_info.configuration)),
        _kernel(_outer.get_network().get_engine()->get_context(), _kernel_data.kernel_name, get_jit_constants(_outer, _kernel_data))
    {}

    static kernel_data set_kernel_data(const scale_inst& outer)
    {
        auto engine_info = outer.get_network().get_engine()->get_context()->get_engine_info();

        const auto& input_mem = outer.input_memory();  // input

        kernel_data kd;

        kd.fp16_unit_used = input_mem.get_layout().data_type == cldnn::data_types::f16;
        kd.fp16_supported = engine_info.supports_fp16 != 0;
        kd.scale_bfyx_used = false;
        kd.input_bfyx_used = false;
        // Determine global work sizes.
        kd.gws0 = input_mem.get_layout().size.batch[0];   // B
        kd.gws1 = input_mem.get_layout().size.feature[0]; // F
        kd.gws2 = input_mem.get_layout().size.spatial[0] *
            (input_mem.get_layout().size.spatial.size() > 1 ? input_mem.get_layout().size.spatial[1] : 1); // X, Y
        // Find largest positive local work size that is divider for global work size.
        kd.lws2 = std::min(std::max(kd.gws2, static_cast<size_t>(1)), static_cast<size_t>(32));
        while (kd.gws2 % kd.lws2 != 0)
        {
            --kd.lws2;
        }
        kd.lws1 = 1;
        kd.lws0 = 1;

        return kd;
    }

    static gpu::jit_constants get_jit_constants(const scale_inst& outer, const kernel_data& data)
    {
        if (!data.fp16_supported && data.fp16_unit_used)
            throw std::invalid_argument("GPU device does not support half precision floating-point formats (cl_khr_fp16 extension)");

            auto input_padding = outer.input().at(0)->desc()->output_padding;
            auto output_padding = outer.argument.output_padding;

        gpu::jit_constants mem_consts{
            gpu::make_jit_constant("INPUT",                 outer.input().at(0)->non_padded_output_layout().size),
            gpu::make_jit_constant("OUTPUT",                outer.non_padded_output_layout().size),
            gpu::make_jit_constant("SCALE",                 outer.scale_memory().get_layout().size),
            gpu::make_jit_constant("FP16_UNIT_USED",        static_cast<int>(data.fp16_unit_used)),
            gpu::make_jit_constant("UNIT_TYPE",             data.fp16_unit_used ? "half" : "float"),
            gpu::make_jit_constant("BIAS_TERM",             static_cast<int>(outer.bias_term())),
            gpu::make_jit_constant("SCALE_BFYX_USED",       static_cast<int>(data.scale_bfyx_used)),
            gpu::make_jit_constant("INPUT_BFYX_USED",       static_cast<int>(data.input_bfyx_used)),
            gpu::make_jit_constant("INPUT_PADDING",         input_padding),
            gpu::make_jit_constant("OUTPUT_PADDING",        output_padding)
        };

        return mem_consts;
    }

    cldnn::refcounted_obj_ptr<cldnn::event_impl> execute(const std::vector<cldnn::refcounted_obj_ptr<cldnn::event_impl>>& events) override
    {
        const auto& outer = _outer;
        const auto& kd = _kernel_data;

        const auto& input_mem = outer.input_memory();  // input
        const auto& scale_mem = outer.scale_memory();  // scale_input
        const auto& output_mem = outer.output_memory(); // output

        if (outer.bias_term())
        {
            const auto& bias_mem = outer.bias_memory();  // bias
            return _kernel.run<gpu::input_mem, gpu::output_mem, gpu::input_mem, gpu::input_mem>({ { kd.gws0, kd.gws1, kd.gws2 },{ kd.lws0, kd.lws1, kd.lws2 } }, events, input_mem, output_mem, scale_mem, bias_mem);
        }
        else
            return _kernel.run<gpu::input_mem, gpu::output_mem, gpu::input_mem>({ {kd.gws0, kd.gws1, kd.gws2 }, {kd.lws0, kd.lws1, kd.lws2 } }, events, input_mem, output_mem, scale_mem);
    }

    static primitive_impl* create(scale_inst &arg) { return new scale_gpu(arg); };
};

scale_gpu::kernel_data set_default(const scale_inst& arg)
{
    scale_gpu::kernel_data kd = scale_gpu::set_kernel_data(arg);
    kd.scale_bfyx_used = (arg.scale_memory().get_layout().size.format == cldnn::format::bfyx) ? true : false;
    kd.input_bfyx_used = (arg.input_memory().get_layout().size.format == cldnn::format::bfyx) ? true : false;
    kd.kernel_name = kernel_name;

    return kd;
}

kd_selector_t<scale_gpu::kernel_data, scale_inst, data_types, format::type, kd_optional_selector_t, neural::gpu::engine_info_internal::architectures, neural::gpu::engine_info_internal::configurations> scale_gpu::ks = {
    { std::make_tuple(data_types::f32, format::yxfb, gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), set_default },
    { std::make_tuple(data_types::f16, format::yxfb, gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), set_default },
    { std::make_tuple(data_types::f32, format::bfyx, gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), set_default },
    { std::make_tuple(data_types::f16, format::bfyx, gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), set_default },
    { std::make_tuple(data_types::f32, format::xb, gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), set_default },
    { std::make_tuple(data_types::f16, format::xb, gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), set_default },
    { std::make_tuple(data_types::f32, format::bx, gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), set_default },
    { std::make_tuple(data_types::f16, format::bx, gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), set_default },
};

namespace {
    struct attach {
        attach() {
            auto val_fw = scale_gpu::create;

            implementation_map<scale_inst>::add(std::make_tuple(cldnn::engine_types::ocl, data_types::f32, format::yxfb), val_fw);
            implementation_map<scale_inst>::add(std::make_tuple(cldnn::engine_types::ocl, data_types::f16, format::yxfb), val_fw);
            implementation_map<scale_inst>::add(std::make_tuple(cldnn::engine_types::ocl, data_types::f32, format::bfyx), val_fw);
            implementation_map<scale_inst>::add(std::make_tuple(cldnn::engine_types::ocl, data_types::f16, format::bfyx), val_fw);
            implementation_map<scale_inst>::add(std::make_tuple(cldnn::engine_types::ocl, data_types::f32, format::xb), val_fw);
            implementation_map<scale_inst>::add(std::make_tuple(cldnn::engine_types::ocl, data_types::f16, format::xb), val_fw);
            implementation_map<scale_inst>::add(std::make_tuple(cldnn::engine_types::ocl, data_types::f32, format::bx), val_fw);
            implementation_map<scale_inst>::add(std::make_tuple(cldnn::engine_types::ocl, data_types::f16, format::bx), val_fw);
        }
        ~attach() {}
    };
    attach attach_impl;
}
} // namespace neural
