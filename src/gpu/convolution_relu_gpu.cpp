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

#include "convolution_relu_gpu.h"
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

convolution_relu_gpu::convolution_relu_gpu(convolution_relu &arg)
        : is_an_implementation(neural::type_id<convolution_relu_gpu>())
        , outer(arg) {};
convolution_relu_gpu::~convolution_relu_gpu() {};
void convolution_relu_gpu::implementation(const void *ptr) {

    auto this_conv = static_cast<const convolution_relu *>(ptr);

    auto& input_offset = this_conv->argument.input_offset;
    auto& output_size   = this_conv->argument.output_size;
    output_size;
    auto& padding       = this_conv->argument.padding;
    auto& stride        = this_conv->argument.stride;
    auto negative_slope = this_conv->argument.negative_slope;

    auto split = this_conv->argument.split;
    auto& filter_arg = this_conv->argument.input[1].primitive.as<const memory&>().argument; //convolution filter

    assert( output_size.feature[0] == filter_arg.size.feature[0] ); // memory::format oixy

    // todo remove
    if(filter_arg.format != memory::format::oiyx_f32) throw std::runtime_error("conv weights arent oiyx_f32 format");

    auto& input_mem = this_conv->input_memory(0);
    auto& output_mem = this_conv->output_memory(0);
    std::vector<std::reference_wrapper<const neural::memory>> biases_mem;
    std::vector<std::reference_wrapper<const neural::memory>> filters_mem;
    for (size_t i = 0; i < split; i++)
    {
        filters_mem.push_back(this_conv->argument.input[i * 2 + 1].primitive.as<const memory&>());
        biases_mem.push_back(this_conv->argument.input[i * 2 + 2].primitive.as<const memory&>());
    }

    neural::vector<uint32_t> _stride(stride);
    if ( (stride.spatial[0] > input_mem.argument.size.spatial[0]) ||
         (stride.spatial[1] > input_mem.argument.size.spatial[1]) )
    {
        _stride.spatial[0] = input_mem.argument.size.spatial[0];
        _stride.spatial[1] = input_mem.argument.size.spatial[1];
    }
 

    // weights neural::vector is: {b}, {ofm, ifm} {spatials}
    // ofm - output feature maps
    // ifm - input feature maps
    // b = 1 always
    // (weights can't have batch so it is equall to 1)
    // Ofm and batch is cropped, ofm will be hold manually
    // Batch is included in output size

    auto dstSize = output_mem.count();

    gpu::jit_constants mem_consts{
        gpu::make_jit_constant("STRIDE", _stride),
        gpu::make_jit_constant("INPUT_OFFSET", input_offset),
        gpu::make_jit_constant("BIAS", biases_mem),
        gpu::make_jit_constant("FILTER", filters_mem),
        gpu::make_jit_constant("NEGATIVE_SLOPE", std::to_string(negative_slope))
    };

    switch(padding){
        case padding::zero:
        {
            if (input_mem.argument.format == memory::format::bfyx_f32)
            {
                auto kernel = gpu::kernel<gpu::input_mem, gpu::output_mem>(kernelName_BFXY, mem_consts);
                kernel({ dstSize, std::min(dstSize, static_cast<size_t>(16)) }, input_mem, output_mem);
            }
            else
            {
                auto kernel = gpu::kernel<gpu::input_mem, gpu::output_mem>(kernelName_YXFB, mem_consts);
                kernel({ {dstSize, 1} , {std::min(dstSize, static_cast<size_t>(16)), 1} }, input_mem, output_mem);
            }
        }
            break;
        default:
            throw std::runtime_error("Unknown padding mode in convolution.");
    }
}

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
