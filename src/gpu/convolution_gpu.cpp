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

#include "convolution_gpu.h"
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

convolution_gpu::convolution_gpu(convolution &arg)
        : is_an_implementation(neural::type_id<convolution_gpu>())
        , outer(arg) {};
convolution_gpu::~convolution_gpu() {};
void convolution_gpu::implementation(const void *ptr) {

    auto this_conv = static_cast<const convolution *>(ptr);

    auto& input_offset  = this_conv->argument.input_offset;
    auto& output_size   = this_conv->argument.output_size;
    output_size;

    auto& padding       = this_conv->argument.padding;
    auto& stride        = this_conv->argument.stride;

    auto split = this_conv->argument.split;
    auto& filter_arg = this_conv->argument.input[1].primitive.as<const memory&>().argument; //convolution filter

    assert( output_size.feature[0] / split == filter_arg.size.feature[0] ); // memory::format oixy

    // todo remove
    if(filter_arg.format != memory::format::oiyx_f32) throw std::runtime_error("conv weights arent oiyx_f32 format");

    auto& input_mem  = this_conv->input_memory(0);
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

    bool inline_memory = filters_mem[0].get().count() > 1024 ? false : true;

    gpu::jit_constants mem_consts{
        gpu::make_jit_constant("STRIDE", _stride),
        gpu::make_jit_constant("INPUT_OFFSET", input_offset)
    };
    if (inline_memory)
    {
        mem_consts.add_constant(gpu::make_jit_constant("BIAS", biases_mem));
        mem_consts.add_constant(gpu::make_jit_constant("FILTER", filters_mem));
    }
    else
    {
        mem_consts.add_constant(gpu::make_jit_constant("FILTER", filters_mem[0].get().argument.size));
        mem_consts.add_constant(gpu::make_jit_constant("FILTER_ARRAY_NUM", std::to_string(split)));
    }

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
                if (inline_memory)
                {
                    auto kernel = gpu::kernel<gpu::input_mem, gpu::output_mem>(kernelName_YXFB, mem_consts);
                    kernel({ { dstSize, 1 } ,{ std::min(dstSize, static_cast<size_t>(16)), 1 } }, input_mem, output_mem);
                }
                else
                {
                    size_t workitems_per_enqueue = dstSize / split;
                    auto kernel = gpu::kernel<gpu::input_mem, gpu::output_mem, gpu::input_mem, gpu::input_mem, cl_uint>(kernelName_YXFB_memory, mem_consts);
                    for (size_t i = 0; i < filters_mem.size(); i++)
                    {
                        kernel({ { workitems_per_enqueue, 1 } ,{ std::min(workitems_per_enqueue, static_cast<size_t>(16)), 1 } }, input_mem, output_mem, filters_mem[i].get(), biases_mem[i].get(), (cl_uint)i);
                    }
                }
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
