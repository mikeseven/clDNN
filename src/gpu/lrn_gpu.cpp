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

#include "api/neural.h"
#include "cache/primitive_db.h"
#include "implementation_map.h"
#include "kernel.h"

#include <algorithm>
#include <stdexcept>
#include <string>


namespace neural
{
// Kernel names.
static const std::string kernel_name = "lrn_gpu";
static const std::string kernel_name_b8 = "lrn_gpu_b8";

// GPU engine information helpers.
namespace
{
struct gpu_info_helper : gpu::context_holder
{
    gpu::engine_info get_engine_info() const
    {
        return context()->get_engine_info();
    }
};
}

struct lrn_gpu : is_an_implementation
{
    const normalization::response& _outer;
    struct kernel_data
    {
        size_t gws0, gws1;
        size_t lws0, lws1;
        std::string kernel_name;
        bool fp16_unit_used;
    } _kernel_data;
    gpu::kernel _kernel;

    lrn_gpu(const normalization::response& outer)
        : is_an_implementation(neural::type_id<lrn_gpu>()),
        _outer(outer),
        _kernel_data(set_kernel_data(_outer)),
        _kernel(_kernel_data.kernel_name, get_jit_constants(_outer, _kernel_data))
    {}

    static kernel_data set_kernel_data(const normalization::response& outer)
    {
        const auto& input_mem  = outer.input_memory(0);  // input

        kernel_data kd;

        kd.fp16_unit_used = memory::traits(input_mem.argument.format).type->name == type_id<half_t>()->name;

        // Determine global work sizes.
        kd.gws0 = input_mem.argument.size.batch[0] * input_mem.argument.size.feature[0];   // B, F
        kd.gws1 = input_mem.argument.size.spatial[0] * input_mem.argument.size.spatial[1]; // X, Y
        // Find largest positive local work size that is divider for global work size.
        kd.lws0 = std::min(std::max(kd.gws0, static_cast<size_t>(1)), static_cast<size_t>(32));
        while (kd.gws0 % kd.lws0 != 0)
        {
            --kd.lws0;
        }
        kd.lws1 = 1;

        // TODO: add half case: b16 (b*f dividable by 128).
        if (!kd.fp16_unit_used &&                        // halfs are not used
            input_mem.argument.size.batch[0] % 8 == 0 && // batch_num is multiple of 8
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

        // Checking for supported paddings.
        switch (outer.argument.padding)
        {
        case padding::zero:
            break;

        default:
            throw std::runtime_error("Unknown padding mode in lrn");
        }

        return kd;
    }

    static gpu::jit_constants get_jit_constants(const normalization::response& outer, const kernel_data& data)
    {
        gpu_info_helper gpu_info;
        auto engine_info = gpu_info.get_engine_info();

        if (!engine_info.supports_fp16 && data.fp16_unit_used)
            throw std::invalid_argument("GPU device does not support half precision floating-point formats (cl_khr_fp16 extension)");

        auto size = outer.argument.size;

        gpu::jit_constants mem_consts {
            gpu::make_jit_constant("INPUT",             outer.input_memory(0).argument.size),
            gpu::make_jit_constant("P_SIZE",            size),
            gpu::make_jit_constant("ALPHA",             outer.argument.alpha),
            gpu::make_jit_constant("BETA",              outer.argument.beta),
            gpu::make_jit_constant("K",                 outer.argument.k),
            gpu::make_jit_constant("HELP_INPUT_OFFSET", outer.argument.input_offset.feature[0] - static_cast<uint32_t>(size / 2)),
            gpu::make_jit_constant("FP16_SUPPORTED",    static_cast<int>(engine_info.supports_fp16)),
            gpu::make_jit_constant("FP16_UNIT_USED",    static_cast<int>(data.fp16_unit_used)),
            gpu::make_jit_constant("UNIT_TYPE",         data.fp16_unit_used ? "half" : "float"),
            gpu::make_jit_constant("UNIT_VAL_ZERO",     data.fp16_unit_used ? "0.0h" : "0.0f")
        };

        return mem_consts;
    }

    static void implementation(const void *ptr)
    {
        auto me = static_cast<const lrn_gpu*>(ptr);
        const auto& outer = me->_outer;
        const auto& kd    = me->_kernel_data;

        const auto& input_mem  = outer.input_memory(0);  // input
        const auto& output_mem = outer.output_memory(0); // output

        me->_kernel.run<gpu::input_mem, gpu::output_mem>
            ({{kd.gws0, kd.gws1}, {kd.lws0, kd.lws1}}, input_mem, output_mem);
    }


    static is_an_implementation *create(normalization::response &arg) { return new lrn_gpu(arg); }

    task_group work() override { return{ { task{ implementation, this } }, schedule::single }; }

};

    namespace {
        struct attach {
            attach() {
                auto val_fw = lrn_gpu::create;

                auto key = std::make_tuple(engine::gpu, memory::format::yxfb_f32, memory::format::yxfb_f32);
                implementation_map<normalization::response>::add(key, val_fw);
                key = std::make_tuple(engine::gpu, memory::format::yxfb_f16, memory::format::yxfb_f16);
                implementation_map<normalization::response>::add(key, val_fw);
            }
            ~attach() {}
        };

#ifdef __GNUC__
        __attribute__((visibility("default"))) //todo maybe dll_sym?
#elif _MSC_VER
#   pragma section(".nn_init$m", read, write)
#endif
        attach attach_impl;

    }

}