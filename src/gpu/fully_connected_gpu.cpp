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
#include "fully_connected_common_gpu.h"
#include "relu_gpu.h"
#include "implementation_map.h"
#include "kernel.h"

const std::string kernelName_xb_xb_memory = "Fully_Connected_GPU_xb_xb_memory";
const std::string kernelCode_xb_xb_memory_Begin = R"__krnl(
KERNEL (Fully_Connected_GPU_xb_xb_memory)(
    const __global float* input, 
    __global float* output, 
    const __global float* weight,
    const __global float* bias)
{
)__krnl";

const std::string kernelName_xb_bx_memory = "Fully_Connected_GPU_xb_bx_memory";
const std::string kernelCode_xb_bx_memory_Begin = R"__krnl(
KERNEL (Fully_Connected_GPU_xb_bx_memory)(
    const __global float* input, 
    __global float* output, 
    const __global float* weight,
    const __global float* bias)
{
)__krnl";

const std::string kernelName_xb_bx_b8_memory = "Fully_Connected_GPU_xb_bx_b8_memory";
const std::string kernelCode_xb_bx_b8_memory_Begin = R"__krnl(
__attribute__((reqd_work_group_size(8, 1, 1)))
KERNEL (Fully_Connected_GPU_xb_bx_b8_memory)(
    const __global float* input, 
    __global float* output, 
    const __global float* weight,
    const __global float* bias)
{
)__krnl";

const std::string kernelName_xb_xb_b8_x8_memory = "Fully_Connected_GPU_xb_xb_b8_x8_memory";
const std::string kernelCode_xb_xb_b8_x8_memory_Begin = R"__krnl(
__attribute__((reqd_work_group_size(8, 1, 1)))
KERNEL (Fully_Connected_GPU_xb_xb_b8_x8_memory)(
    const __global float* input, 
    __global float* output, 
    const __global float* weight,
    const __global float* bias)
{
)__krnl";

const std::string kernelName_xb_xb_b16_memory = "Fully_Connected_GPU_xb_xb_b16_memory";
const std::string kernelCode_xb_xb_b16_memory_Begin = R"__krnl(
KERNEL (Fully_Connected_GPU_xb_xb_b16_memory)(
    const __global float* input, 
    __global float* output, 
    const __global float* weights,
    const __global float* bias)
{
)__krnl";

const std::string KernelName_xb_xb_b8_x8_memory_vload = "Fully_Connected_GPU_xb_xb_b8_x8_memory_vload";
const std::string kernelCode_xb_xb_b8_x8_memory_vload_Begin = R"__krnl(
__attribute__((reqd_work_group_size(LOCAL_WORK_GROUP_SIZE, 1, 1)))
KERNEL (Fully_Connected_GPU_xb_xb_b8_x8_memory_vload)(
    const __global float* input, 
    __global float* output, 
    const __global float* weight,
    const __global float* bias)
{
)__krnl";

const std::string kernelName_yxfn_memory = "Fully_Connected_GPU_yxfn_memory";
const std::string kernelCode_yxfn_memory_Begin = R"__krnl(
KERNEL (Fully_Connected_GPU_yxfn_memory)(
    const __global float* input, 
    __global float* output, 
    const __global float* weight,
    const __global float* bias)
{
)__krnl";

const std::string kernelCode_End = R"__krnl(
}
)__krnl";

namespace neural {

    struct fully_connected_gpu : is_an_implementation {
        fully_connected &outer;
        gpu::kernel _kernel;
       
        fully_connected_gpu(fully_connected &arg)
            : is_an_implementation(neural::type_id<fully_connected_gpu>())
            , outer(arg)
            , _kernel(select_kernel_name(), get_jit_constants())
        {}

        const std::string& select_kernel_name() const {
            // input
            auto& input_mem = outer.input_memory(0);
            auto& weight_mem = outer.input_memory(1);
            auto& output_mem = outer.output_memory(0);

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
                    if (batch_multiple_of_8)
                    {
                        return kernelName_xb_bx_b8_memory;
                    }
                    else
                    {
                        return kernelName_xb_bx_memory;
                    }
                    break;
                }
                case memory::format::yxfb_f32:
                case memory::format::xb_f32:
                {
                    if (batch_multiple_of_8 &&
                        (output_mem.count() / output_mem.argument.size.batch[0]) % 8 == 0)
                    {
                        return KernelName_xb_xb_b8_x8_memory_vload;
                    }
                    else
                    {
                        return kernelName_xb_xb_memory;
                    }
                    break;
                }
                case memory::format::bfyx_f32:
                {
                    return kernelName_yxfn_memory;
                }
                default:
                    throw std::invalid_argument("Weight memory format is not supported");
                }
                break;
            }
            default:
                throw std::invalid_argument("Input memory format is not supported");
            }
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

        gpu::jit_constants get_jit_constants() const {
            auto& input_mem = outer.input_memory(0);
            auto& output_mem = outer.output_memory(0);

            // weights
            auto& weight_mem = outer.input_memory(1);

            float negative_slope = outer.argument.negative_slope;

            gpu::jit_constants mem_consts{
                gpu::make_jit_constant("INPUT", input_mem.argument.size),
                gpu::make_jit_constant("OUTPUT", output_mem.argument.size),
                gpu::make_jit_constant("INPUT_ELEMENTS_COUNT", input_mem.count() / input_mem.argument.size.batch[0])
            };

            if (outer.argument.use_relu)
            {
                mem_consts.add_constant(gpu::make_jit_constant("RELU", ""));
                mem_consts.add_constant(gpu::make_jit_constant("NEGATIVE_SLOPE", negative_slope));
            }

            mem_consts.add_constant(gpu::make_jit_constant("WEIGHTS", weight_mem.argument.size));

            // temporary values
            if (weight_mem.argument.format == memory::format::type::yxfb_f32 ||
                weight_mem.argument.format == memory::format::type::xb_f32)
            {
                int batch_size = input_mem.argument.size.batch[0];
                const int batches_per_work_item = get_batches_per_work_item(output_mem);
                const int local_work_group_size = get_local_work_group_size(output_mem);
                mem_consts.add_constant(gpu::make_jit_constant("LOCAL_WORK_GROUP_SIZE", local_work_group_size));
                mem_consts.add_constant(gpu::make_jit_constant("NEURONS_PER_WORK_ITEM", get_neurons_per_work_item(output_mem))); // how many neurons for a single batch will a single work item produce
                mem_consts.add_constant(gpu::make_jit_constant("BATCHES_PER_WORK_ITEM", batches_per_work_item)); // how many batches will a single work item compute
                mem_consts.add_constant(gpu::make_jit_constant("LOCAL_WORK_GROUPS_PER_SINGLE_BATCHES_ELEMENTS", std::max((batch_size / batches_per_work_item) / local_work_group_size, 1))); // how many local work groups we need to compute single element for each batch
                mem_consts.add_constant(gpu::make_jit_constant("WORK_ITEMS_PER_SINGLE_BATCHES_ELEMENTS", batch_size / batches_per_work_item)); // how many work items we need to compute single element for each batch
            }
            return mem_consts;
        }

        static void implementation(const void *ptr) {
            auto me = static_cast<const fully_connected_gpu *>(ptr);
            auto& outer = me->outer;

            // input
            auto& input_mem = outer.input_memory(0);
            // weights
            auto& weight_mem = outer.input_memory(1);
            // bias
            auto& bias_mem = outer.input_memory(2);
            // output
            auto& output_mem = outer.output_memory(0);

            auto output_bufSize = output_mem.count();

            size_t gws0 = output_bufSize;
            size_t gws1 = 1;
            size_t lws0 = 32;
            size_t lws1 = 1;

            // calculate local workgroup size
            while (output_bufSize % lws0) {
                lws0--;
            }

            switch (input_mem.argument.format) {
            case memory::format::yxfb_f32:
            case memory::format::xb_f32:
                switch (weight_mem.argument.format)
                {
                case memory::format::byxf_f32:
                case memory::format::bx_f32:
                    if (input_mem.argument.size.batch[0] % 8 == 0)
                    {
                        gws0 = output_mem.argument.size.batch[0];
                        gws1 = output_mem.argument.size.spatial[0];
                        lws0 = 8;
                        lws1 = 1;
                    }
                    break;
                    case memory::format::yxfb_f32:
                    case memory::format::xb_f32:
                        if (input_mem.argument.size.batch[0] % 8 == 0)
                        {
                            size_t groups_per_batches = get_local_groups_size(output_mem);
                            gws0 = output_bufSize / (get_neurons_per_work_item(output_mem) * get_batches_per_work_item(output_mem) * groups_per_batches);
                            gws1 = groups_per_batches;
                            lws0 = get_local_work_group_size(output_mem);
                            lws1 = 1;
                        }
                        break;
                }
                break;
            case memory::format::x_f32:
                break;
            default:
                throw std::invalid_argument("Input memory format is not supported");
            }

            me->_kernel.run<gpu::input_mem, gpu::output_mem, gpu::input_mem, gpu::input_mem>
                ({ {gws0, gws1 }, {lws0, lws1 } }, input_mem, output_mem, weight_mem, bias_mem);

        }

        task_group work() override {
            return{ { task{ implementation, this } }, schedule::single };
        }

        static is_an_implementation *create(fully_connected &arg) {
            // input
            auto& input_mem = arg.input_memory(0);
            auto& input_size = input_mem.argument.size;
            // validate arguments
            if (input_mem.argument.format == memory::format::yxfb_f32) {
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
            gpu::kernel_templates::add(kernelName_xb_xb_memory, inline_utils_float + kernelCode_xb_xb_memory_Begin + fully_connected_code_xb_xb_memory + kernelCode_End + inline_utils_float_end);
            gpu::kernel_templates::add(kernelName_xb_bx_memory, inline_utils_float + kernelCode_xb_bx_memory_Begin + fully_connected_code_xb_bx_memory + kernelCode_End + inline_utils_float_end);
            gpu::kernel_templates::add(kernelName_xb_bx_b8_memory, inline_utils_float + kernelCode_xb_bx_b8_memory_Begin + fully_connected_code_xb_bx_b8_memory + kernelCode_End + inline_utils_float_end);
            gpu::kernel_templates::add(kernelName_xb_xb_b8_x8_memory, inline_utils_float + kernelCode_xb_xb_b8_x8_memory_Begin + fully_connected_code_xb_xb_b8_x8_memory + kernelCode_End + inline_utils_float_end);
            gpu::kernel_templates::add(KernelName_xb_xb_b8_x8_memory_vload, inline_utils_float + kernelCode_xb_xb_b8_x8_memory_vload_Begin + fully_connected_code_xb_xb_b8_x8_memory_vload + kernelCode_End + inline_utils_float_end);
            gpu::kernel_templates::add(kernelName_xb_xb_b16_memory, inline_utils_float + kernelCode_xb_xb_b16_memory_Begin + fully_connected_code_xb_xb_b16_memory + kernelCode_End + inline_utils_float_end);
            gpu::kernel_templates::add(kernelName_yxfn_memory, inline_utils_float + kernelCode_yxfn_memory_Begin + fully_connected_code_yxfn_memory + kernelCode_End + inline_utils_float_end);

            auto val_fw = fully_connected_gpu::create;

            implementation_map<fully_connected>::add({
                { std::make_tuple(engine::gpu, memory::format::yxfb_f32, memory::format::xb_f32), val_fw },
                { std::make_tuple(engine::gpu, memory::format::xb_f32, memory::format::xb_f32), val_fw },
                { std::make_tuple(engine::gpu, memory::format::x_f32,  memory::format::x_f32), val_fw }
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