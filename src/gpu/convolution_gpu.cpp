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
#include "multidimensional_counter.h"
#include "convolution_common_gpu.h"
#include "implementation_map.h"
#include "kernel.h"

namespace neural {

const std::string kernelName_YXFB = "Convolution_GPU_YXFB";
const std::string kernelCode_YXFB_Begin = R"__krnl(
#define INPUT_SIZE_X input_size[2]
KERNEL(Convolution_GPU_YXFB)(
    const __global neural_memory* input_mem,
    __global neural_memory* dst_mem)
{)__krnl";

const std::string kernelName_BFXY = "Convolution_GPU_BFXY";
const std::string kernelCode_BFXY_Begin = R"__krnl(
KERNEL(Convolution_GPU_BFXY)(
    const __global neural_memory* input_mem,
    __global neural_memory* dst_mem)
{
)__krnl";

const std::string kernelName_YXFB_memory = "Convolution_GPU_YXFB_memory";
const std::string kernelCode_YXFB_memory_Begin = R"__krnl(
#define INPUT_SIZE_X input_size[2]
KERNEL(Convolution_GPU_YXFB_memory)(
    const __global neural_memory* input_mem,
    __global neural_memory* dst_mem,
    const __global neural_memory* filter_mem,
    const __global neural_memory* bias_mem,
    uint split_idx)
{)__krnl";

const std::string kernelCodeEnd = R"__krnl(
}
)__krnl";


struct convolution_gpu : is_an_implementation {
    convolution &outer;
    bool inline_memory;
    gpu::kernel _kernel;

    convolution_gpu(convolution &arg): is_an_implementation(neural::type_id<convolution_gpu>())
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
        auto& padding = outer.argument.padding;

        if (padding != padding::zero)
            throw std::invalid_argument("Unknown padding mode in convolution.");

        switch (input_mem.argument.format) {
        case memory::format::bfyx_f32:
            if (!inline_memory)
                throw std::logic_error("Not supported yet");
            return kernelName_BFXY;
        case memory::format::yxfb_f32:
                return inline_memory ? kernelName_YXFB : kernelName_YXFB_memory;
        default:
            throw std::invalid_argument("Input memory format is not supported");
        }
    }

    gpu::jit_constants get_jit_constants() const {

        auto& input_mem = outer.input_memory(0);
        auto& input_offset = outer.argument.input_offset;
        auto split = outer.argument.split;
        auto& output_offset = outer.argument.output_offset;
        auto& output_size   = outer.argument.output_size;

        neural::vector<uint32_t> stride(outer.argument.stride);
        stride.spatial[0] = std::min(stride.spatial[0], input_mem.argument.size.spatial[0]);
        stride.spatial[1] = std::min(stride.spatial[1], input_mem.argument.size.spatial[1]);

        // weights neural::vector is: {b}, {ofm, ifm} {spatials}
        // ofm - output feature maps
        // ifm - input feature maps
        // b = 1 always
        // (weights can't have batch so it is equall to 1)
        // Ofm and batch is cropped, ofm will be hold manually
        // Batch is included in output size

        gpu::jit_constants mem_consts{
            gpu::make_jit_constant("STRIDE", stride),
            gpu::make_jit_constant("INPUT_OFFSET", input_offset)
            gpu::make_jit_constant("OUTPUT_OFFSET", output_offset),
            gpu::make_jit_constant("OUTPUT_SIZE", output_size)
        };

        if (inline_memory)
        {
            std::vector<std::reference_wrapper<const neural::memory>> biases_mem;
            std::vector<std::reference_wrapper<const neural::memory>> filters_mem;
            for (uint32_t i = 0; i < split; i++)
            {
                filters_mem.push_back(outer.input_memory(i * 2 + 1));
                biases_mem.push_back(outer.input_memory(i * 2 + 2));
            }

            mem_consts.add_constant(gpu::make_jit_constant("BIAS", biases_mem));
            mem_consts.add_constant(gpu::make_jit_constant("FILTER", filters_mem));
        }
        else
        {
            mem_consts.add_constant(gpu::make_jit_constant("FILTER", outer.input_memory(1).argument.size));
            mem_consts.add_constant(gpu::make_jit_constant("FILTER_ARRAY_NUM", std::to_string(split)));
        }
        return mem_consts;
    }

    static void implementation(const void *ptr) {
        auto me = static_cast<const convolution_gpu*>(ptr);
        auto& outer = me->outer;

        auto split = outer.argument.split;

        auto& input_mem = outer.input_memory(0);
        auto& output_mem = outer.output_memory(0);

        // weights neural::vector is: {b}, {ofm, ifm} {spatials}
        // ofm - output feature maps
        // ifm - input feature maps
        // b = 1 always
        // (weights can't have batch so it is equall to 1)
        // Ofm and batch is cropped, ofm will be hold manually
        // Batch is included in output size

        if (outer.argument.padding != padding::zero)
            throw std::invalid_argument("Unknown padding mode in convolution.");

        auto dstSize = output_mem.count();

        switch (input_mem.argument.format) {
        case memory::format::bfyx_f32:
            if (!me->inline_memory)
                throw std::logic_error("Not supported yet");
            me->_kernel.run<gpu::input_mem, gpu::output_mem>
                ({ { dstSize, 1 } ,{ std::min(dstSize, static_cast<size_t>(16)), 1 } }, input_mem, output_mem);
            break;
        case memory::format::yxfb_f32:
            if (me->inline_memory) {
                me->_kernel.run<gpu::input_mem, gpu::output_mem>
                    ({ { dstSize, 1 } ,{ std::min(dstSize, static_cast<size_t>(16)), 1 } }, input_mem, output_mem);
            }
            else {
                size_t workitems_per_enqueue = dstSize / split;
                for (uint32_t i = 0; i < split; i++) {
                    me->_kernel.run<gpu::input_mem, gpu::output_mem, gpu::input_mem, gpu::input_mem, uint32_t>
                        ({ { workitems_per_enqueue, 1 } ,{ std::min(workitems_per_enqueue, static_cast<size_t>(16)), 1 } },
                            input_mem,
                            output_mem,
                            outer.input_memory(i * 2 + 1), //filters
                            outer.input_memory(i * 2 + 2), //biases
                            i);
                }
            }
            break;
        default:
            throw std::invalid_argument("Input memory format is not supported");
        }
    }

    static is_an_implementation *create(convolution &arg) {
        auto& filter_arg = arg.input_memory(1).argument; //convolution filter

        assert(arg.argument.output_size.feature[0] / arg.argument.split == filter_arg.size.feature[0]); // memory::format oixy
        // todo remove
        if (filter_arg.format != memory::format::oiyx_f32) throw std::runtime_error("conv weights arent oiyx_f32 format");

        return new convolution_gpu(arg);
    }

    task_group work() override {
        return{ { task{ implementation, this } }, schedule::single };
    }
};


namespace{
struct attach{
    attach(){
        gpu::kernel_templates::add(kernelName_YXFB, kernelCode_YXFB_Begin + convolution_code_yxfb + kernelCodeEnd);
        gpu::kernel_templates::add(kernelName_BFXY, kernelCode_BFXY_Begin + convolution_code_bfxy + kernelCodeEnd);
        gpu::kernel_templates::add(kernelName_YXFB_memory, kernelCode_YXFB_memory_Begin + convolution_code_yxfb_memory + kernelCodeEnd);

        auto val_fw = convolution_gpu::create;

        auto key_fw = std::make_tuple(engine::gpu, memory::format::yxfb_f32, memory::format::yxfb_f32);
        implementation_map<convolution>::add(key_fw, val_fw);

        auto key_fw2 = std::make_tuple(engine::gpu, memory::format::bfyx_f32, memory::format::bfyx_f32);
        implementation_map<convolution>::add(key_fw2, val_fw);
    }
    ~attach(){}
};

#ifdef __GNUC__
    __attribute__((visibility("default"))) //todo meybe dll_sym?
#elif _MSC_VER
#   pragma section(".nn_init$m", read, write)
#endif
attach attach_impl;

}
}
