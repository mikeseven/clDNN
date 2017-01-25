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
static const std::string kernel_name = "mean_subtract_gpu";

struct mean_subtract_gpu : is_an_implementation
{
    const mean_substract& _outer;
    struct kernel_data
    {
        size_t gws0;
        size_t lws0;
        std::string kernel_name;
        bool fp16_unit_used;
        bool bfyx_mean_format_used; ///< Indicates that old bfyx format of mean is used.
    } _kernel_data;
    gpu::kernel _kernel;

    mean_subtract_gpu(const mean_substract& outer)
        : _outer(outer),
        _kernel_data(set_kernel_data(_outer)),
        _kernel(_outer.get_network().get_engine()->get_context(), _outer.id(), _kernel_data.kernel_name, get_jit_constants(_outer, _kernel_data))
    {}

    static kernel_data set_kernel_data(const mean_substract& outer)
    {
        const auto& input_mem  = outer.input_memory(0);  // input
        const auto& mean_mem   = outer.input_memory(1);  // mean
        const auto& output_mem = outer.output_memory(); // output

        kernel_data kd;

        kd.fp16_unit_used = input_mem.get_layout().data_type == cldnn::data_types::f16;
        kd.bfyx_mean_format_used = false;

        // Determine global work sizes.
        kd.gws0 = output_mem.count();
        // Find largest positive local work size that is divider for global work size.
        kd.lws0 = std::min(std::max(kd.gws0, static_cast<size_t>(1)), static_cast<size_t>(32));
        while (kd.gws0 % kd.lws0 != 0)
        {
            --kd.lws0;
        }

        // Checking for supported mean formats.
        if (kd.fp16_unit_used)
        {
            switch (mean_mem.argument().format)
            {
            case memory::format::yxfb_f16:
                break;
            case memory::format::bfyx_f16:
                kd.bfyx_mean_format_used = true;
                break;

            default:
                throw std::runtime_error("mean_subtract mean isn't yxfb_f16 or bfyx_f16 format");
            }
        }
        else {
            switch (mean_mem.argument().format)
            {
            case memory::format::yxfb_f32:
                break;
            case memory::format::bfyx_f32:
                kd.bfyx_mean_format_used = true;
                break;

            default:
                throw std::runtime_error("mean_subtract mean isn't yxfb_f32 or bfyx_f32 format");
            }
        }

        kd.kernel_name = kernel_name;

        return kd;
    }

    static gpu::jit_constants get_jit_constants(const mean_substract& outer, const kernel_data& data)
    {
        auto engine_info = outer.get_network().get_engine()->get_context()->get_engine_info();

        if (!engine_info.supports_fp16 && data.fp16_unit_used)
            throw std::invalid_argument("GPU device does not support half precision floating-point formats (cl_khr_fp16 extension)");

        return {
            gpu::make_jit_constant("INPUT",                 outer.input_memory(0).argument().size),
            gpu::make_jit_constant("MEAN" ,                 outer.input_memory(1).argument().size),
            gpu::make_jit_constant("FP16_SUPPORTED",        static_cast<int>(engine_info.supports_fp16)),
            gpu::make_jit_constant("FP16_UNIT_USED",        static_cast<int>(data.fp16_unit_used)),
            gpu::make_jit_constant("UNIT_TYPE",             data.fp16_unit_used ? "half" : "float"),
            gpu::make_jit_constant("BFYX_MEAN_FORMAT_USED", static_cast<int>(data.bfyx_mean_format_used))
        };
    }

    cldnn::refcounted_obj_ptr<cldnn::event_impl> execute(const std::vector<cldnn::refcounted_obj_ptr<cldnn::event_impl>>& events) override
    {
        const auto& kd    = _kernel_data;

        const auto& input_mem  = _outer.input_memory(0);  // input
        const auto& mean_mem   = _outer.input_memory(1);  // mean
        const auto& output_mem = _outer.output_memory(); // output

        // mean_mem memory in bfyx or yxfb.
        return _kernel.run<gpu::input_mem, gpu::output_mem, gpu::input_mem>({kd.gws0, kd.lws0}, events, input_mem, output_mem, mean_mem);
    }

    static is_an_implementation *create(const mean_substract& outer) { return new mean_subtract_gpu(outer); }

};

namespace {
    struct attach {
        attach() {
            auto val_fw = mean_subtract_gpu::create;

            auto key_fw = std::make_tuple(cldnn::engine_types::ocl, memory::format::yxfb_f32);
            implementation_map<mean_substract>::add(key_fw, val_fw);
            key_fw = std::make_tuple(cldnn::engine_types::ocl, memory::format::yxfb_f16);
            implementation_map<mean_substract>::add(key_fw, val_fw);
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
}
