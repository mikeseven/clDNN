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
#include "crop_arg.h"
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
    static const std::string kernel_name = "crop_gpu";
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

    struct crop_gpu : is_an_implementation
    {
        const cldnn::crop_arg& _outer;
        gpu::engine_info_internal _engine_info;

        struct kernel_data
        {
            size_t gws0, gws1, gws2;
            size_t lws0, lws1, lws2;
            std::string kernel_name;
            bool fp16_unit_used;
            bool fp16_supported;
            bool crop_bfyx_used; ///< Indicates that bfyx format of crop is used.
        } _kernel_data;
        gpu::kernel _kernel;

        static kd_selector_t<kernel_data, cldnn::crop_arg, kd_optional_selector_t, neural::gpu::engine_info_internal::architectures, neural::gpu::engine_info_internal::configurations> ks;

        crop_gpu(const cldnn::crop_arg& outer) :
            _outer(outer),
            _engine_info(outer.get_network().get_engine()->get_context()->get_engine_info()),
            _kernel_data(ks.get_kernel(outer, _engine_info.architecture, _engine_info.configuration)),
            _kernel(_outer.get_network().get_engine()->get_context(), _kernel_data.kernel_name, get_jit_constants(_outer, _kernel_data))
        {}

        static kernel_data set_kernel_data(const cldnn::crop_arg& outer)
        {
            auto engine_info = outer.get_network().get_engine()->get_context()->get_engine_info();

            const auto& output_mem = outer.output_memory();  // input

            kernel_data kd;

            kd.fp16_unit_used = output_mem.get_layout().data_type == cldnn::data_types::f16;
            kd.fp16_supported = engine_info.supports_fp16 != 0;
            kd.crop_bfyx_used = false;

            // Determine global work sizes.
            kd.gws0 = output_mem.argument().size.batch[0];   // B
            kd.gws1 = output_mem.argument().size.feature[0]; // F
            kd.gws2 = output_mem.argument().size.spatial[0] * output_mem.argument().size.spatial[1]; // X, Y
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

        static gpu::jit_constants get_jit_constants(const cldnn::crop_arg& outer, const kernel_data& data)
        {
            if (!data.fp16_supported && data.fp16_unit_used)
                throw std::invalid_argument("GPU device does not support half precision floating-point formats (cl_khr_fp16 extension)");

            gpu::jit_constants mem_consts{
                gpu::make_jit_constant("INPUT",                outer.input_memory(0).argument().size),
                gpu::make_jit_constant("OUTPUT",               outer.output_memory().argument().size),
                gpu::make_jit_constant("OFFSETS",              outer.argument.offsets),
                gpu::make_jit_constant("FP16_UNIT_USED",       static_cast<int>(data.fp16_unit_used)),
                gpu::make_jit_constant("UNIT_TYPE",            data.fp16_unit_used ? "half" : "float"),
                gpu::make_jit_constant("CROP_BFYX_USED",       static_cast<int>(data.crop_bfyx_used)),
            };

            return mem_consts;
        }

        cldnn::refcounted_obj_ptr<cldnn::event_impl> execute(const std::vector<cldnn::refcounted_obj_ptr<cldnn::event_impl>>& events) override
        {
            const auto& outer = _outer;
            const auto& kd = _kernel_data;

            const auto& input_mem = outer.input_memory(0);  // input
            const auto& output_mem = outer.output_memory(); // output

            return _kernel.run<gpu::input_mem, gpu::output_mem>({ {kd.gws0, kd.gws1, kd.gws2 }, {kd.lws0, kd.lws1, kd.lws2 } }, events, input_mem, output_mem);
        }

        static is_an_implementation *create(cldnn::crop_arg &arg) { return new crop_gpu(arg); };
    };

    crop_gpu::kernel_data set_default(const cldnn::crop_arg& arg)
    {
        crop_gpu::kernel_data kd = crop_gpu::set_kernel_data(arg);
        kd.crop_bfyx_used = (memory::to_tensor_format(arg.input_memory(0).argument().format) == cldnn::format::bfyx) ? true : false;
        kd.kernel_name = kernel_name;

        return kd;
    }

    kd_selector_t<crop_gpu::kernel_data, cldnn::crop_arg, kd_optional_selector_t, neural::gpu::engine_info_internal::architectures, neural::gpu::engine_info_internal::configurations> crop_gpu::ks = {
        { std::make_tuple(gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), set_default },
    };

    namespace {
        struct attach {
            attach() {
                auto val_fw = crop_gpu::create;

                implementation_map<cldnn::crop_arg>::add(std::make_tuple(cldnn::engine_types::ocl, memory::format::yxfb_f32), val_fw);
                implementation_map<cldnn::crop_arg>::add(std::make_tuple(cldnn::engine_types::ocl, memory::format::yxfb_f16), val_fw);
                implementation_map<cldnn::crop_arg>::add(std::make_tuple(cldnn::engine_types::ocl, memory::format::bfyx_f32), val_fw);
                implementation_map<cldnn::crop_arg>::add(std::make_tuple(cldnn::engine_types::ocl, memory::format::bfyx_f16), val_fw);
            }
            ~attach() {}
        };

        attach attach_impl;

    }
} // namespace neural
