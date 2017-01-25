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
    static const std::string kernel_name = "batch_norm_gpu";
    static const std::string kernel_name_global_stats = "batch_norm_use_global_stats_gpu";
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

    namespace normalization
    {
        struct batch_norm_gpu : is_an_implementation
        {
            const batch_norm& _outer;
            gpu::engine_info_internal _engine_info;

            struct kernel_data
            {
                size_t gws0, gws1, gws2;
                size_t lws0, lws1, lws2;
                std::string kernel_name;
                bool fp16_unit_used;
                bool fp16_supported;
                bool bfyx_mean_format_used; ///< Indicates that old bfyx format of mean is used.
                bool bfyx_variance_format_used; ///< Indicates that old bfyx format of variance is used.
            } _kernel_data;
            gpu::kernel _kernel;

            static kd_selector_t<kernel_data, batch_norm, neural::memory::format::type, neural::memory::format::type, neural::memory::format::type, bool, kd_optional_selector_t, neural::gpu::engine_info_internal::architectures, neural::gpu::engine_info_internal::configurations> ks;

            batch_norm_gpu(const batch_norm& outer):
                _outer(outer),
                _engine_info(outer.get_network().get_engine()->get_context()->get_engine_info()),
                _kernel_data(ks.get_kernel(outer, outer.input_memory(0).argument().format, outer.mean_memory().argument().format, outer.variance_memory().argument().format, outer.use_global_stats(), _engine_info.architecture, _engine_info.configuration)),
                _kernel(_outer.get_network().get_engine()->get_context(), _outer.id(), _kernel_data.kernel_name, get_jit_constants(_outer, _kernel_data))
            {}

            static kernel_data set_kernel_data(const batch_norm& outer)
            {
                auto engine_info = outer.get_network().get_engine()->get_context()->get_engine_info();

                const auto& input_mem = outer.input_memory(0);  // input

                kernel_data kd;

                kd.fp16_unit_used = input_mem.get_layout().data_type == cldnn::data_types::f16;
                kd.fp16_supported = engine_info.supports_fp16 != 0;
                kd.bfyx_mean_format_used = false;
                kd.bfyx_variance_format_used = false;

                // Determine global work sizes.
                kd.gws0 = input_mem.argument().size.batch[0];   // B
                kd.gws1 = input_mem.argument().size.feature[0]; // F
                kd.gws2 = input_mem.argument().size.spatial[0] * input_mem.argument().size.spatial[1]; // X, Y
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

            static gpu::jit_constants get_jit_constants(const batch_norm& outer, const kernel_data& data)
            {
                if (!data.fp16_supported && data.fp16_unit_used)
                    throw std::invalid_argument("GPU device does not support half precision floating-point formats (cl_khr_fp16 extension)");

                gpu::jit_constants mem_consts {
                    gpu::make_jit_constant("INPUT",                 outer.input_memory(0).argument().size),
                    gpu::make_jit_constant("EPSILON",               data.fp16_unit_used ? 0.0f : outer.argument.epsilon),
                    gpu::make_jit_constant("FP16_UNIT_USED",        static_cast<int>(data.fp16_unit_used)),
                    gpu::make_jit_constant("UNIT_TYPE",             data.fp16_unit_used ? "half" : "float"),
                    gpu::make_jit_constant("UNIT_VAL_ZERO",         data.fp16_unit_used ? "0.0h" : "0.0f"),
                };

                if (outer.argument.use_global_stats)
                {
                    mem_consts.add_constant(gpu::make_jit_constant("MEAN", outer.input_memory(1).argument().size));
                    mem_consts.add_constant(gpu::make_jit_constant("VARIANCE", outer.input_memory(2).argument().size));
                    mem_consts.add_constant(gpu::make_jit_constant("BFYX_MEAN_FORMAT_USED", static_cast<int>(data.bfyx_mean_format_used)));
                    mem_consts.add_constant(gpu::make_jit_constant("BFYX_VARIANCE_FORMAT_USED", static_cast<int>(data.bfyx_variance_format_used)));
                }

                return mem_consts;
            }

            cldnn::refcounted_obj_ptr<cldnn::event_impl> execute(const std::vector<cldnn::refcounted_obj_ptr<cldnn::event_impl>>& events) override
            {
                const auto& outer = _outer;
                const auto& kd = _kernel_data;

                const auto& input_mem = outer.input_memory(0);  // input
                const auto& mean_mem = outer.input_memory(1);  // mean
                const auto& variance_mem = outer.input_memory(2);  // variance
                const auto& output_mem = outer.output_memory(); // output

                return _kernel.run<gpu::input_mem, gpu::output_mem, gpu::input_mem, gpu::input_mem>({{kd.gws0, kd.gws1, kd.gws2 }, {kd.lws0, kd.lws1, kd.lws2 }}, events, input_mem, output_mem, mean_mem, variance_mem);
            }

            static is_an_implementation *create(batch_norm &arg) { return new batch_norm_gpu(arg); };
        };

        batch_norm_gpu::kernel_data set_default_use_global_stats(const normalization::batch_norm& arg)
        {
            batch_norm_gpu::kernel_data kd = batch_norm_gpu::set_kernel_data(arg);
            kd.kernel_name = kernel_name_global_stats;

            return kd;
        }

        batch_norm_gpu::kernel_data set_default(const normalization::batch_norm& arg)
        {
            batch_norm_gpu::kernel_data kd = batch_norm_gpu::set_kernel_data(arg);
            kd.kernel_name = kernel_name;

            return kd;
        }

        batch_norm_gpu::kernel_data set_mean_bfyx(const normalization::batch_norm& arg)
        {
            batch_norm_gpu::kernel_data kd = set_default_use_global_stats(arg);
            kd.bfyx_mean_format_used = true;

            return kd;
        }

        batch_norm_gpu::kernel_data set_variance_bfyx(const normalization::batch_norm& arg)
        {
            batch_norm_gpu::kernel_data kd = set_default_use_global_stats(arg);
            kd.bfyx_variance_format_used = true;

            return kd;
        }

        batch_norm_gpu::kernel_data set_mean_variance_bfyx(const normalization::batch_norm& arg)
        {
            batch_norm_gpu::kernel_data kd = set_default_use_global_stats(arg);
            kd.bfyx_mean_format_used = true;
            kd.bfyx_variance_format_used = true;

            return kd;
        }

        kd_selector_t<batch_norm_gpu::kernel_data, normalization::batch_norm, neural::memory::format::type, neural::memory::format::type, neural::memory::format::type, bool, kd_optional_selector_t, neural::gpu::engine_info_internal::architectures, neural::gpu::engine_info_internal::configurations> batch_norm_gpu::ks = {
            { std::make_tuple(memory::format::yxfb_f32, memory::format::yxfb_f32, memory::format::yxfb_f32, false, gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), set_default },
            { std::make_tuple(memory::format::yxfb_f32, memory::format::yxfb_f32, memory::format::yxfb_f32, true, gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), set_default_use_global_stats },
            { std::make_tuple(memory::format::yxfb_f32, memory::format::bfyx_f32, memory::format::yxfb_f32, true, gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), set_mean_bfyx },
            { std::make_tuple(memory::format::yxfb_f32, memory::format::yxfb_f32, memory::format::bfyx_f32, true, gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), set_variance_bfyx },
            { std::make_tuple(memory::format::yxfb_f32, memory::format::bfyx_f32, memory::format::bfyx_f32, true, gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), set_mean_variance_bfyx },
            { std::make_tuple(memory::format::yxfb_f16, memory::format::yxfb_f16, memory::format::yxfb_f16, false, gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), set_default },
            { std::make_tuple(memory::format::yxfb_f16, memory::format::yxfb_f16, memory::format::yxfb_f16, true, gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), set_default_use_global_stats },
            { std::make_tuple(memory::format::yxfb_f16, memory::format::bfyx_f16, memory::format::yxfb_f16, true, gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), set_mean_bfyx },
            { std::make_tuple(memory::format::yxfb_f16, memory::format::yxfb_f16, memory::format::bfyx_f16, true, gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), set_variance_bfyx },
            { std::make_tuple(memory::format::yxfb_f16, memory::format::bfyx_f16, memory::format::bfyx_f16, true, gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), set_mean_variance_bfyx },
        };

        namespace {
            struct attach {
                attach() {
                    auto val_fw = batch_norm_gpu::create;

                    implementation_map<batch_norm>::add(std::make_tuple(cldnn::engine_types::ocl, memory::format::yxfb_f32), val_fw);
                    implementation_map<batch_norm>::add(std::make_tuple(cldnn::engine_types::ocl, memory::format::yxfb_f16), val_fw);
                }
                ~attach() {}
            };

#ifdef __GNUC__
            __attribute__((visibility("default"))) //todo meybe dll_sym?
#elif _MSC_VER
#   pragma section(".nn_init$m", read, write)
#endif
            attach attach_impl;

        }
    } // namespace normalization
} // namespace neural
