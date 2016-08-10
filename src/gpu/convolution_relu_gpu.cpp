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
#include "memory_utils.h"
#include "kernel.h"

namespace neural {

const std::string kernelName_YXFB = "Convolution_Relu_GPU_YXFB";
const std::string kernelCode_YXFB_Begin = R"__krnl(
#define INPUT_SIZE_X input_size[2]
KERNEL(Convolution_Relu_GPU_YXFB)(
    const __global neural_memory* input_mem,
    __global neural_memory* dst_mem)
{)__krnl";

const std::string kernelName_BFXY = "Convolution_Relu_GPU_BFXY";
const std::string kernelCode_BFXY_Begin = R"__krnl(
KERNEL(Convolution_Relu_GPU_BFXY)(
    const __global neural_memory* input_mem,
    __global neural_memory* dst_mem)
{)__krnl";

const std::string kernelCode_Relu = "    pDst[global_id] = max(pDst[global_id], 0.0f) + NEGATIVE_SLOPE * min(pDst[global_id], 0.0f);";

const std::string kernelCode_End = R"__krnl(
}
)__krnl";

struct convolution_relu_gpu : is_an_implementation {
    convolution_relu &outer;
    gpu::kernel _kernel;

    convolution_relu_gpu(convolution_relu &arg)
        : is_an_implementation(neural::type_id<convolution_relu_gpu>())
        , outer(arg)
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
            return kernelName_BFXY;
        case memory::format::yxfb_f32:
            return kernelName_YXFB;
        default:
            throw std::invalid_argument("Input memory format is not supported");
        }
    }

    gpu::jit_constants get_jit_constants() const {

        auto& input_mem = outer.input_memory(0);
        auto& input_offset = outer.argument.input_offset;
        auto split = outer.argument.split;
        auto negative_slope = outer.argument.negative_slope;

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

        std::vector<std::reference_wrapper<const neural::memory>> biases_mem;
        std::vector<std::reference_wrapper<const neural::memory>> filters_mem;
        for (uint32_t i = 0; i < split; i++)
        {
            filters_mem.push_back(outer.input_memory(i * 2 + 1));
            biases_mem.push_back(outer.input_memory(i * 2 + 2));
        }

        return gpu::jit_constants{
            gpu::make_jit_constant("STRIDE", stride),
            gpu::make_jit_constant("INPUT_OFFSET", input_offset),
            gpu::make_jit_constant("BIAS", biases_mem),
            gpu::make_jit_constant("FILTER", filters_mem),
            gpu::make_jit_constant("NEGATIVE_SLOPE", std::to_string(negative_slope))
        };
    }

    static void implementation(const void *ptr) {
        auto me = static_cast<const convolution_relu_gpu*>(ptr);
        auto& outer = me->outer;

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

        if (input_mem.argument.format == memory::format::bfyx_f32) {
            me->_kernel.run<gpu::input_mem, gpu::output_mem>
                ({ dstSize, std::min(dstSize, static_cast<size_t>(16)) }, input_mem, output_mem);
        } else {
            me->_kernel.run<gpu::input_mem, gpu::output_mem>
                ({ {dstSize, 1} , {std::min(dstSize, static_cast<size_t>(16)), 1} }, input_mem, output_mem);
        }
    }

    static is_an_implementation *create(convolution_relu &arg) {
        auto& filter_arg = arg.input_memory(1).argument; //convolution filter

        assert(arg.argument.output_size.feature[0] / arg.argument.split == filter_arg.size.feature[0]); // memory::format oixy
        // todo remove
        if (filter_arg.format != memory::format::oiyx_f32) throw std::runtime_error("conv weights arent oiyx_f32 format");

        return new convolution_relu_gpu(arg);
    }

    task_group work() override {
        return{ { task{ implementation, this } }, schedule::single };
    }
};

namespace{
struct attach{
    attach(){
        gpu::kernel_templates::add(kernelName_YXFB, kernelCode_YXFB_Begin + convolution_code_yxfb + kernelCode_Relu + kernelCode_End);
        auto val_fw = convolution_relu_gpu::create;

        auto key_fw = std::make_tuple(engine::gpu, memory::format::yxfb_f32, memory::format::yxfb_f32);
        implementation_map<convolution_relu>::add(key_fw, val_fw);

        gpu::kernel_templates::add(kernelName_BFXY, kernelCode_BFXY_Begin + convolution_code_bfxy + kernelCode_Relu + kernelCode_End);
        auto key_fw2 = std::make_tuple(engine::gpu, memory::format::bfyx_f32, memory::format::bfyx_f32);
        implementation_map<convolution_relu>::add(key_fw2, val_fw);
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
