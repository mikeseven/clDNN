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

///////////////////////////////////////////////////////////////////////////////////////////////////

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
static const std::string kernel_name_xb_xb = "fully_connected_gpu_xb_xb";
static const std::string kernel_name_xb_bx = "fully_connected_gpu_xb_bx";
static const std::string kernel_name_xb_bx_b8 = "fully_connected_gpu_xb_bx_b8";
static const std::string kernel_name_xb_xb_b8_x8 = "fully_connected_gpu_xb_xb_b8_x8";
static const std::string kernel_name_xb_xb_b16 = "fully_connected_gpu_xb_xb_b16";
static const std::string kernel_name_xb_xb_b8_x8_vload = "fully_connected_gpu_xb_xb_b8_x8_vload";
static const std::string kernel_name_yxfn = "fully_connected_gpu_yxfn";

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

struct fully_connected_gpu : is_an_implementation
{
    const fully_connected& _outer;
    struct kernel_data 
    {
        size_t gws0, gws1;
        size_t lws0, lws1;
        std::string kernel_name;
        bool fp16_unit_used;
    } _kernel_data;
    gpu::kernel _kernel;
       
    fully_connected_gpu(const fully_connected& outer)
        : is_an_implementation(neural::type_id<fully_connected_gpu>()),
        _outer(outer),
        _kernel_data(set_kernel_data(_outer)),
        _kernel(_kernel_data.kernel_name, get_jit_constants(_outer, _kernel_data))
    {}

    static kernel_data set_kernel_data(const fully_connected& outer)
    {
        const auto& input_mem  = outer.input_memory(0);   // input
        const auto& weight_mem = outer.input_memory(1);   // weights
        const auto& output_mem = outer.output_memory(0);  // output

        kernel_data kd;

        kd.fp16_unit_used = memory::traits(input_mem.argument.format).type->name == type_id<half_t>()->name;

        // Determine global work sizes.
        kd.gws0 = output_mem.count();
        kd.gws1 = 1;

        // Find largest positive local work size that is divider for global work size.
        kd.lws0 = std::min(std::max(kd.gws0, static_cast<size_t>(1)), static_cast<size_t>(32));
        while (kd.gws0 % kd.lws0 != 0)
        {
            --kd.lws0;
        }
        kd.lws1 = 1;

        bool batch_multiple_of_8 = input_mem.argument.size.batch[0] % 8 == 0;

        switch (input_mem.argument.format)
        {
        case memory::format::yxfb_f32:
        case memory::format::xb_f32:
        case memory::format::x_f32:
        {
            switch (weight_mem.argument.format)
            {
            case memory::format::byxf_f32:
            case memory::format::bx_f32:
            {
                if (input_mem.argument.size.batch[0] == 8)
                {
                    kd.gws0 = output_mem.argument.size.batch[0];
                    kd.gws1 = output_mem.argument.size.spatial[0];
                    kd.lws0 = 8;
                    kd.lws1 = 1;
                    kd.kernel_name = kernel_name_xb_bx_b8;
                }
                else
                {
                    kd.kernel_name = kernel_name_xb_bx;
                }
                break;
            }
            case memory::format::yxfb_f32:
            case memory::format::xb_f32:
            {
                if (batch_multiple_of_8 &&
                    (output_mem.count() / output_mem.argument.size.batch[0]) % 8 == 0)
                {
                    size_t groups_per_batches = get_local_groups_size(output_mem);
                    kd.gws0 = output_mem.count() / (get_neurons_per_work_item(output_mem) * get_batches_per_work_item(output_mem) * groups_per_batches);
                    kd.gws1 = groups_per_batches;
                    kd.lws0 = get_local_work_group_size(output_mem);
                    kd.lws1 = 1;
                    kd.kernel_name = kernel_name_xb_xb_b8_x8_vload;
                }
                else
                {
                    kd.kernel_name = kernel_name_xb_xb;
                }
                break;
            }
            case memory::format::bfyx_f32:
            {
                kd.kernel_name = kernel_name_yxfn;
                break;
            }
            default:
                throw std::invalid_argument("Weight memory format is not supported");
            }
            break;
        }

        case memory::format::yxfb_f16:
        case memory::format::xb_f16:
        case memory::format::x_f16:
        {
            switch (weight_mem.argument.format)
            {
            case memory::format::yxfb_f16:
            case memory::format::xb_f16:
            {
                kd.kernel_name = kernel_name_xb_xb;
                break;
            }

            default:
                throw std::invalid_argument("Weight memory format is not supported");
            }
            break;
        }

        default:
            throw std::invalid_argument("Input memory format is not supported");
        }
        return kd;
    }

    // how many neurons for a single batch will a single work item produce 
    static int get_neurons_per_work_item(const neural::memory &output_mem)
    {
        int batch_size = output_mem.argument.size.batch[0];
        auto out_elements_count_per_batch = output_mem.count() / batch_size;
        if (out_elements_count_per_batch % 16 == 0)
            return 2;
        else
            return 1;
    }

    // how many batches will a single work item compute
    static int get_batches_per_work_item(const neural::memory &output_mem)
    {
        int batch_size = output_mem.argument.size.batch[0];
        return std::min(batch_size, 32);
    }

    static int get_local_work_group_size(const neural::memory &output_mem)
    {
        int batch_size = output_mem.argument.size.batch[0];
        if (batch_size >= 16)
            return 8;
        auto out_elements_count_per_batch = output_mem.count() / batch_size;
        if (out_elements_count_per_batch % 16 == 0)
            return 16;
        else
            return 8;
    }

    static int get_local_groups_size(const neural::memory &output_mem)
    {
        int batch_size = output_mem.argument.size.batch[0];
        return std::max(1, batch_size / get_batches_per_work_item(output_mem));
    }

    static gpu::jit_constants get_jit_constants(const fully_connected& outer, const kernel_data& data)
    {
        gpu_info_helper gpu_info;
        auto engine_info = gpu_info.get_engine_info();

        const auto& input_mem  = outer.input_memory(0);   // input
        const auto& weight_mem = outer.input_memory(1);   // weights
        const auto& output_mem = outer.output_memory(0);  // output

        if (!engine_info.supports_fp16 && data.fp16_unit_used)
            throw std::invalid_argument("GPU device does not support half precision floating-point formats (cl_khr_fp16 extension)");

        gpu::jit_constants mem_consts{
            gpu::make_jit_constant("INPUT",                input_mem.argument.size),
            gpu::make_jit_constant("OUTPUT",               output_mem.argument.size),
            gpu::make_jit_constant("INPUT_ELEMENTS_COUNT", input_mem.count() / input_mem.argument.size.batch[0]),
            gpu::make_jit_constant("WEIGHTS",              weight_mem.argument.size),
            gpu::make_jit_constant("FP16_SUPPORTED",       static_cast<int>(engine_info.supports_fp16)),
            gpu::make_jit_constant("FP16_UNIT_USED",       static_cast<int>(data.fp16_unit_used)),
            gpu::make_jit_constant("UNIT_TYPE",            data.fp16_unit_used ? "half" : "float"),
            gpu::make_jit_constant("UNIT_SUFFIX",          data.fp16_unit_used ? "h" : "f")
        };

        if (outer.argument.use_relu)
        {
            mem_consts.add_constant(gpu::make_jit_constant("RELU",           ""));
            mem_consts.add_constant(gpu::make_jit_constant("NEGATIVE_SLOPE", outer.argument.negative_slope));
        }

        if (data.kernel_name == kernel_name_xb_xb_b8_x8_vload ||
            data.kernel_name == kernel_name_xb_xb_b16)
        {
            int batch_size = input_mem.argument.size.batch[0];
            const int batches_per_work_item = get_batches_per_work_item(output_mem);
            const int local_work_group_size = get_local_work_group_size(output_mem);

            mem_consts.add_constant(gpu::make_jit_constant("LOCAL_WORK_GROUP_SIZE",                         local_work_group_size));
            mem_consts.add_constant(gpu::make_jit_constant("NEURONS_PER_WORK_ITEM",                         get_neurons_per_work_item(output_mem)));                                     // how many neurons for a single batch will a single work item produce
            mem_consts.add_constant(gpu::make_jit_constant("BATCHES_PER_WORK_ITEM",                         batches_per_work_item));                                                     // how many batches will a single work item compute
            mem_consts.add_constant(gpu::make_jit_constant("LOCAL_WORK_GROUPS_PER_SINGLE_BATCHES_ELEMENTS", std::max((batch_size / batches_per_work_item) / local_work_group_size, 1))); // how many local work groups we need to compute single element for each batch
            mem_consts.add_constant(gpu::make_jit_constant("WORK_ITEMS_PER_SINGLE_BATCHES_ELEMENTS",        batch_size / batches_per_work_item));                                        // how many work items we need to compute single element for each batch
        }
        return mem_consts;
    }

    static void implementation(const void *ptr)
    {
        auto me = static_cast<const fully_connected_gpu *>(ptr);
        const auto& outer = me->_outer;
        const auto& kd    = me->_kernel_data;

        const auto& input_mem  = outer.input_memory(0);   // input
        const auto& weight_mem = outer.input_memory(1);   // weights
        const auto& bias_mem   = outer.input_memory(2);   // biases
        const auto& output_mem = outer.output_memory(0);  // output

        me->_kernel.run<gpu::input_mem, gpu::output_mem, gpu::input_mem, gpu::input_mem>
            ({{kd.gws0, kd.gws1}, {kd.lws0, kd.lws1}}, input_mem, output_mem, weight_mem, bias_mem);
    }

    task_group work() override {
        return{ { task{ implementation, this } }, schedule::single };
    }

    static is_an_implementation *create(fully_connected &arg) {

        auto& input_mem = arg.input_memory(0);
        auto& input_size = input_mem.argument.size;

        // validate arguments
        if (input_mem.argument.format == memory::format::yxfb_f32 ||
            input_mem.argument.format == memory::format::yxfb_f16)
        {
            // weights
            auto& weight_size = arg.input_memory(1).argument.size;
            if (   input_size.feature.size() != weight_size.feature.size()
                || input_size.batch.size()   != weight_size.batch.size()
                || input_size.feature[0]     != weight_size.feature[0])
                throw std::invalid_argument("Input and weights sizes do not match");
        }
        else {
            // int a,b,c; a*b*c = 1  => a=b=c=1
            if (1 != input_size.feature.size() * input_size.batch.size() * input_size.feature[0])
                throw std::invalid_argument("Wrong input size");
        }

        return new fully_connected_gpu(arg);
    };
};

namespace {
    struct attach {
        attach() {
            auto val_fw = fully_connected_gpu::create;

            implementation_map<fully_connected>::add({
                { std::make_tuple(engine::gpu, memory::format::yxfb_f32, memory::format::xb_f32), val_fw },
                { std::make_tuple(engine::gpu, memory::format::xb_f32, memory::format::xb_f32), val_fw },
                { std::make_tuple(engine::gpu, memory::format::x_f32,  memory::format::x_f32), val_fw },

                { std::make_tuple(engine::gpu, memory::format::yxfb_f16, memory::format::xb_f16), val_fw },
                { std::make_tuple(engine::gpu, memory::format::xb_f16, memory::format::xb_f16), val_fw },
                { std::make_tuple(engine::gpu, memory::format::x_f16,  memory::format::x_f16), val_fw }
            });
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