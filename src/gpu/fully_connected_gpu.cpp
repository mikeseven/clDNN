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

const std::string kernelName_xb = "Fully_Connected_GPU_xb";
const std::string kernelCode_xb_Begin = R"__krnl(
KERNEL (Fully_Connected_GPU_xb)(
    const __global neural_memory* input_mem,
    __global neural_memory* dst_mem)
{
)__krnl";

const std::string kernelName_xb_bx = "Fully_Connected_GPU_xb_bx";
const std::string kernelCode_xb_bx_Begin = R"__krnl(
KERNEL (Fully_Connected_GPU_xb_bx)(
    const __global neural_memory* input_mem,
    __global neural_memory* dst_mem)
{
)__krnl";

const std::string kernelName_yxfn = "Fully_Connected_GPU_yxfn";
const std::string kernelCode_yxfn_Begin = R"__krnl(
KERNEL (Fully_Connected_GPU_yxfn)(
    const __global neural_memory* input_mem,
    __global neural_memory* dst_mem)
{
)__krnl";

const std::string kernelName_xb_memory = "Fully_Connected_GPU_xb_memory";
const std::string kernelCode_xb_memory_Begin = R"__krnl(
KERNEL (Fully_Connected_GPU_xb_memory)(
    const __global neural_memory* input_mem, 
    __global neural_memory* dst_mem, 
    const __global neural_memory* weights_mem,
    const __global neural_memory* bias_mem)
{
)__krnl";

const std::string kernelName_xb_bx_memory = "Fully_Connected_GPU_xb_bx_memory";
const std::string kernelCode_xb_bx_memory_Begin = R"__krnl(
KERNEL (Fully_Connected_GPU_xb_bx_memory)(
    const __global neural_memory* input_mem, 
    __global neural_memory* dst_mem, 
    const __global neural_memory* weights_mem,
    const __global neural_memory* bias_mem)
{
)__krnl";

const std::string kernelName_xb_bx_b8_memory = "Fully_Connected_GPU_xb_bx_b8_memory";
const std::string kernelCode_xb_bx_b8_memory_Begin = R"__krnl(
__attribute__((reqd_work_group_size(8, 1, 1)))
KERNEL (Fully_Connected_GPU_xb_bx_b8_memory)(
    const __global neural_memory* input_mem, 
    __global neural_memory* dst_mem, 
    const __global neural_memory* weights_mem,
    const __global neural_memory* bias_mem)
{
)__krnl";

const std::string kernelName_xb_xb_b8_x8_memory = "Fully_Connected_GPU_xb_xb_b8_x8_memory";
const std::string kernelCode_xb_xb_b8_x8_memory_Begin = R"__krnl(
__attribute__((reqd_work_group_size(8, 1, 1)))
KERNEL (Fully_Connected_GPU_xb_xb_b8_x8_memory)(
    const __global neural_memory* input_mem, 
    __global neural_memory* dst_mem, 
    const __global neural_memory* weights_mem,
    const __global neural_memory* bias_mem)
{
)__krnl";

const std::string kernelName_yxfn_memory = "Fully_Connected_GPU_yxfn_memory";
const std::string kernelCode_yxfn_memory_Begin = R"__krnl(
KERNEL (Fully_Connected_GPU_yxfn_memory)(
    const __global neural_memory* input_mem, 
    __global neural_memory* dst_mem, 
    const __global neural_memory* weights_mem,
    const __global neural_memory* bias_mem)
{
)__krnl";

const std::string kernelName_yxfn_byxf_memory = "Fully_Connected_GPU_yxfn_byxf_memory";
const std::string kernelCode_yxfn_byxf_memory_Begin = R"__krnl(
KERNEL (Fully_Connected_GPU_yxfn_byxf_memory)(
    const __global neural_memory* input_mem, 
    __global neural_memory* dst_mem, 
    const __global neural_memory* weights_mem,
    const __global neural_memory* bias_mem)
{
)__krnl";

const std::string kernelName_yxfn_byxf_b8_f8_memory = "Fully_Connected_GPU_yxfn_byxf_b8_f8_memory";
const std::string kernelCode_yxfn_byxf_b8_f8_memory_Begin = R"__krnl(
__attribute__((reqd_work_group_size(8, 1, 1)))
KERNEL (Fully_Connected_GPU_yxfn_byxf_b8_f8_memory)(
    const __global neural_memory* input_mem, 
    __global neural_memory* dst_mem, 
    const __global neural_memory* weights_mem,
    const __global neural_memory* bias_mem)
{
)__krnl";
const std::string kernelCode_End = R"__krnl(
}
)__krnl";

namespace neural {

    struct fully_connected_gpu : is_an_implementation {
        fully_connected &outer;
        bool inline_memory;
        gpu::kernel _kernel;
       
        fully_connected_gpu(fully_connected &arg)
            : is_an_implementation(neural::type_id<fully_connected_gpu>())
            , outer(arg)
            , inline_memory(can_inline_memory(arg.input_memory(1)))
            , _kernel(select_kernel_name(), get_jit_constants())
        {}

        static bool can_inline_memory(const neural::memory& mem) {
            return mem.count() <= 1024;
        }

        const std::string& select_kernel_name() const {
            // input
            auto& input_mem = outer.input_memory(0);
            auto& weight_mem = outer.input_memory(1);
            auto& output_mem = outer.output_memory(0);
            if (input_mem.argument.format == memory::format::yxfb_f32) {
                if (weight_mem.argument.format == memory::format::byxf_f32)
                {
                    if (input_mem.argument.size.batch[0] % 8 == 0)
                    {
                        return kernelName_xb_bx_b8_memory;
                    }
                    else
                    {
                        return kernelName_yxfn_byxf_memory;
                    }
                }
                else if (weight_mem.argument.format == memory::format::yxfb_f32)
                {
                    if (input_mem.argument.size.batch[0] % 8 == 0 &&
                        (output_mem.count() / output_mem.argument.size.batch[0]) % 8 == 0)
                        return kernelName_xb_xb_b8_x8_memory;
                    else
                        throw std::runtime_error("yxfb_f32 for input not multiple of 64 not implemented yet");
                }
                else
                {
                    return inline_memory ? kernelName_yxfn : kernelName_yxfn_memory;
                }
            }
            else {
                if (outer.input_memory(1).argument.format == memory::format::bx_f32)
                {
                    if (input_mem.argument.size.batch[0] % 8 == 0)
                    {
                        return kernelName_xb_bx_b8_memory;
                    }
                    else
                        return inline_memory ? kernelName_xb_bx : kernelName_xb_bx_memory;
                }
                else if (outer.input_memory(1).argument.format == memory::format::xb_f32)
                {
                    if (input_mem.argument.size.batch[0] % 8 == 0 &&
                        (output_mem.count() / output_mem.argument.size.batch[0]) % 8 == 0)
                        return kernelName_xb_xb_b8_x8_memory;
                    else
                        throw std::runtime_error("xb_f32 for input not multiple of 64 not implemented yet");
                }
                else
                    return inline_memory ? kernelName_xb : kernelName_xb_memory;
            }
        }

        gpu::jit_constants get_jit_constants() const {
            auto& input_mem = outer.input_memory(0);
            auto& output_mem = outer.output_memory(0);

            // weights
            auto& weight_mem = outer.input_memory(1);
            // bias
            auto& bias_mem = outer.input_memory(2);

            float negative_slope = outer.argument.negative_slope;

            gpu::jit_constants mem_consts{
                gpu::make_jit_constant("INPUT", input_mem.argument.size),
                gpu::make_jit_constant("OUTPUT", output_mem.argument.size),
                gpu::make_jit_constant("INPUT_ELEMENTS_COUNT", std::to_string(input_mem.count() / input_mem.argument.size.batch[0]))
            };

            if (weight_mem.argument.format == memory::format::type::yxfb_f32 ||
                weight_mem.argument.format == memory::format::type::xb_f32)
            {
                auto out_elements_count_per_batch = output_mem.count() / output_mem.argument.size.batch[0];
                if (out_elements_count_per_batch % 16 == 0)
                    mem_consts.add_constant(gpu::make_jit_constant("NEURONS_PER_WORK_ITEM", std::to_string(16)));
                else if(out_elements_count_per_batch % 8 == 0)
                    mem_consts.add_constant(gpu::make_jit_constant("NEURONS_PER_WORK_ITEM", std::to_string(8)));
            }
            if (outer.argument.use_relu)
            {
                mem_consts.add_constant(gpu::make_jit_constant("RELU", ""));
                mem_consts.add_constant(gpu::make_jit_constant("NEGATIVE_SLOPE", std::to_string(negative_slope)));
            }
            if (inline_memory)
            {
                mem_consts.add_constant(gpu::make_jit_constant("WEIGHTS", weight_mem));
                mem_consts.add_constant(gpu::make_jit_constant("BIASES", bias_mem));
            }
            else
            {
                mem_consts.add_constant(gpu::make_jit_constant("WEIGHTS", weight_mem.argument.size));
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

            // calculate local workgroup size
            int lws = 16;
            while (output_bufSize % lws) {
                lws--;
            }

            switch (input_mem.argument.format) {
            case memory::format::yxfb_f32:
            case memory::format::xb_f32:
                if ((weight_mem.argument.format == memory::format::byxf_f32 ||
                    weight_mem.argument.format == memory::format::bx_f32) &&
                    input_mem.argument.size.batch[0] % 8 == 0)
                {
                    me->_kernel.run<gpu::input_mem, gpu::output_mem, gpu::input_mem, gpu::input_mem>
                        ({ output_bufSize, 8 }, input_mem, output_mem, weight_mem, bias_mem);
                }
                else if (weight_mem.argument.format == memory::format::yxfb_f32 ||
                        weight_mem.argument.format == memory::format::xb_f32)
                {
                    auto out_elements_count_per_batch = output_mem.count() / output_mem.argument.size.batch[0];
                    auto neurons_per_workitem=1;
                    if (out_elements_count_per_batch % 16 == 0)
                        neurons_per_workitem = 16;
                    else if (out_elements_count_per_batch % 8 == 0)
                        neurons_per_workitem = 8;
                    me->_kernel.run<gpu::input_mem, gpu::output_mem, gpu::input_mem, gpu::input_mem>
                        ({ output_bufSize / neurons_per_workitem, 8 }, input_mem, output_mem, weight_mem, bias_mem);
                }
                else
                {
                    if (me->inline_memory) {
                        me->_kernel.run<gpu::input_mem, gpu::output_mem >
                            ({ output_bufSize, std::min(output_bufSize, static_cast<size_t>(lws)) }, input_mem, output_mem);
                    }
                    else {
                        me->_kernel.run<gpu::input_mem, gpu::output_mem, gpu::input_mem, gpu::input_mem>
                            ({ output_bufSize, std::min(output_bufSize, static_cast<size_t>(lws)) }, input_mem, output_mem, weight_mem, bias_mem);
                    }
                }
                break;
            case memory::format::x_f32:
                if (me->inline_memory) {
                    me->_kernel.run<gpu::input_mem, gpu::output_mem>
                        ({ output_bufSize, std::min(output_bufSize, static_cast<size_t>(lws)) }, input_mem, output_mem);
                } else {
                    me->_kernel.run<gpu::input_mem, gpu::output_mem, gpu::input_mem, gpu::input_mem>
                        ({ output_bufSize, std::min(output_bufSize, static_cast<size_t>(lws)) }, input_mem, output_mem, weight_mem, bias_mem);
                }
                break;
            default:
                throw std::invalid_argument("Input memory format is not supported");
            }
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
            gpu::kernel_templates::add(kernelName_xb, inline_utils_float + kernelCode_xb_Begin + fully_connected_code_xb + kernelCode_End + inline_utils_float_end);
            gpu::kernel_templates::add(kernelName_xb_bx, inline_utils_float + kernelCode_xb_bx_Begin + fully_connected_code_xb_bx + kernelCode_End + inline_utils_float_end);
            gpu::kernel_templates::add(kernelName_yxfn, inline_utils_float + kernelCode_yxfn_Begin + fully_connected_code_yxfn + kernelCode_End + inline_utils_float_end);
            gpu::kernel_templates::add(kernelName_xb_memory, inline_utils_float + kernelCode_xb_memory_Begin + fully_connected_code_xb_memory + kernelCode_End + inline_utils_float_end);
            gpu::kernel_templates::add(kernelName_xb_bx_memory, inline_utils_float + kernelCode_xb_bx_memory_Begin + fully_connected_code_xb_bx_memory + kernelCode_End + inline_utils_float_end);
            gpu::kernel_templates::add(kernelName_xb_bx_b8_memory, inline_utils_float + kernelCode_xb_bx_b8_memory_Begin + fully_connected_code_xb_bx_b8_memory + kernelCode_End + inline_utils_float_end);
            gpu::kernel_templates::add(kernelName_xb_xb_b8_x8_memory, inline_utils_float + kernelCode_xb_xb_b8_x8_memory_Begin + fully_connected_code_xb_xb_b8_x8_memory + kernelCode_End + inline_utils_float_end);
            gpu::kernel_templates::add(kernelName_yxfn_memory, inline_utils_float + kernelCode_yxfn_memory_Begin + fully_connected_code_yxfn_memory + kernelCode_End + inline_utils_float_end);
            gpu::kernel_templates::add(kernelName_yxfn_byxf_memory, inline_utils_float + kernelCode_yxfn_byxf_memory_Begin + fully_connected_code_yxfn_byxf_memory + kernelCode_End + inline_utils_float_end);
            gpu::kernel_templates::add(kernelName_yxfn_byxf_b8_f8_memory, inline_utils_float + kernelCode_yxfn_byxf_b8_f8_memory_Begin + fully_connected_code_yxfn_byxf_b8_f8_memory + kernelCode_End + inline_utils_float_end);

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