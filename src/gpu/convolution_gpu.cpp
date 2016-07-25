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

#include <iterator>
#include "convolution_gpu.h"
#include "multidimensional_counter.h"
#include "memory_utils.h"
#include "kernel.h"

#pragma warning(disable: 4189)
#pragma warning(disable: 4100)

namespace neural {

const std::string kernelName = "Convolution_GPU";
const std::string kernelCode = R"__krnl(
#define INPUT_SIZE_X input_size[2]
KERNEL(Convolution_GPU)(
    const __global neural_memory* input_mem,
    __global neural_memory* dst_mem)
{
//
    const __global uint* input_size = get_raw(input_mem);
    const __global uint* dst_size = get_raw(dst_mem);
    const __global float* input = (const __global float*)get_data(input_mem);
    __global float* pDst = (__global float*)get_data(dst_mem);
 //

    int global_id = get_global_id(0);
    const int batch_num = dst_size[0];
    const int batch_offset = global_id % batch_num;

    const int ofm_num = dst_size[1];
    const int ofm_offset = ((global_id / batch_num) % ofm_num) / FILTER_ARRAY_NUM;

    const int f_ofm_offset = (global_id % FILTER_OUTPUT_FEATURE_NUM) * FILTER_SIZE_Y * FILTER_SIZE_X * FILTER_INPUT_FEATURE_NUM;

    const int idx = (global_id / batch_num) / FILTER_ARRAY_NUM;

    const int i_ifm_num = input_size[1];

    const int x = ((idx / FILTER_OUTPUT_FEATURE_NUM) % dst_size[2]) * STRIDE_SIZE_X + INPUT_OFFSET_SIZE_X;
    const int y = (((idx / FILTER_OUTPUT_FEATURE_NUM) * STRIDE_SIZE_Y) / INPUT_SIZE_X) * STRIDE_SIZE_Y + INPUT_OFFSET_SIZE_Y;

    int divider = FILTER_ARRAY_NUM > FILTER_INPUT_FEATURE_NUM ? 1 : FILTER_INPUT_FEATURE_NUM / FILTER_ARRAY_NUM;
    const int split_idx = ((global_id / batch_num) / divider) % FILTER_ARRAY_NUM;

    pDst[global_id] = 0;
    for (uint h = 0; h < FILTER_INPUT_FEATURE_NUM; h++)
    {
        const int f_ifm_offset = h * FILTER_SIZE_Y * FILTER_SIZE_X;
        for (uint i = 0; i < FILTER_SIZE_Y; i++)
        {
            for (uint j = 0; j < FILTER_SIZE_X; j++)
            {
                int input_offset_x = x + j;
                int input_offset_y = y + i;

                bool zero = false;
                zero = input_offset_x < 0 ? true : zero;
                zero = input_offset_y < 0 ? true : zero;
                zero = input_offset_x >= input_size[2] ? true : zero;
                zero = input_offset_y >= input_size[3] ? true : zero;

                int input_idx = (input_offset_x + (input_offset_y * INPUT_SIZE_X)) * batch_num * i_ifm_num;
                input_idx += split_idx * batch_num;
                input_idx += h * batch_num;
                input_idx += batch_offset;
                int filter_idx = (i * FILTER_SIZE_X + j) + f_ofm_offset + f_ifm_offset;
                pDst[global_id] += zero ? 0 : input[input_idx] * FILTER[split_idx][filter_idx];
            }
        }
    }
   pDst[global_id] += BIAS[split_idx][ofm_offset];
}
)__krnl";

const std::string kernelName_BFXY_f32 = "Convolution_GPU_bfxy_f32";
const std::string kernelCode_BFXY_f32 = R"__krnl(
KERNEL(Convolution_GPU_bfxy_f32)(
    const __global neural_memory* input_mem,
    __global neural_memory* dst_mem)
{
//
    const __global uint* input_size = get_raw(input_mem);
    const __global uint* dst_size = get_raw(dst_mem);
    const __global float* input = (const __global float*)get_data(input_mem);
    __global float* pDst = (__global float*)get_data(dst_mem);
 //

    const int global_id = get_global_id(0);

    const int output_feature_num = dst_size[1];
    const int output_feature_size = dst_size[2] * dst_size[3];
    const int output_batch_size = output_feature_num * output_feature_size;

    const int output_feature_idx = (global_id / output_feature_size ) % output_feature_num;
    const int batch_idx = global_id / output_batch_size;

    const int filter_input_feature_size = FILTER_SIZE_X * FILTER_SIZE_Y;

    const int filter_output_feature_num = FILTER_OUTPUT_FEATURE_NUM;
    const int filter_output_feature_size = FILTER_INPUT_FEATURE_NUM * filter_input_feature_size;
    const int filter_output_feature_offset = output_feature_idx * filter_output_feature_size;
    
    const int input_feature_num = input_size[1];
    const int input_feature_size = input_size[2] * input_size[3];

    const int input_batch_size = input_feature_num * input_feature_size;
    const int input_batch_offset = input_batch_size * batch_idx;

    const int input_x_offset = global_id % (input_size[2] / STRIDE_SIZE_X) * STRIDE_SIZE_X + INPUT_OFFSET_SIZE_X;    
    const int input_y_offset = ((global_id / (input_size[2] / STRIDE_SIZE_X)) % (input_size[3] / STRIDE_SIZE_Y)) * STRIDE_SIZE_Y + INPUT_OFFSET_SIZE_Y;
    
    const int input_offset = input_batch_offset + input_y_offset * input_size[2] + input_x_offset;
    
    pDst[global_id] = 0;
    for(uint h = 0; h < FILTER_INPUT_FEATURE_NUM; h++)
    {
        const int filter_input_feature_offset = h * filter_input_feature_size;   
        const int input_feature_offset = h * input_feature_size;
        for( uint i = 0; i < FILTER_SIZE_Y; i++)
        {
            for (uint j = 0; j < FILTER_SIZE_X; j++)
            {
                int input_idx = j + i * input_size[2] + input_offset + input_feature_offset;
                int filter_idx = (i * FILTER_SIZE_X + j) + filter_output_feature_offset + filter_input_feature_offset;
                pDst[global_id] += input[input_idx] * FILTER[0][filter_idx];
            }
        }
    }
    // TODO!!!! change [0] from BIAS and FILTER to something that works - [0] is for temporary compilation
    pDst[global_id] += BIAS[0][output_feature_idx];
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
    for (int i = 0; i < split; i++)
    {
        filters_mem.push_back(this_conv->argument.input[i * 2 + 1].primitive.as<const memory&>());
        biases_mem.push_back(this_conv->argument.input[i * 2 + 2].primitive.as<const memory&>());
    }

    const int f_pos = 1; // neural::vector format is b,f,spatials. In input and output 'b' and 'f' fields are always scalars.
    namespace nd = ndimensional;
    nd::value<uint32_t> range (output_size.raw);

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

    size_t dstSize = output_mem.count();

    gpu::jit_constants mem_consts{
        gpu::make_jit_constant("STRIDE", _stride),
        gpu::make_jit_constant("INPUT_OFFSET", input_offset),
        gpu::make_jit_constant("BIAS", biases_mem), 
        gpu::make_jit_constant("FILTER", filters_mem)
    };

    switch(padding){
        case padding::zero:
        {
            if (input_mem.argument.format == memory::format::bfyx_f32)
            {
                auto kernel = gpu::kernel<gpu::input_mem, gpu::output_mem>("Convolution_GPU_bfxy_f32", mem_consts);
                kernel({ dstSize, std::min(dstSize, (size_t)16) }, input_mem, output_mem);
            }
            else
            {
                auto kernel = gpu::kernel<gpu::input_mem, gpu::output_mem>("Convolution_GPU", mem_consts);
                kernel({ {dstSize, 1} , {std::min(dstSize, (size_t)16), 1} }, input_mem, output_mem);
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
        gpu::kernel_templates::add(kernelName, kernelCode);
        auto key_fw = std::make_tuple(engine::gpu, memory::format::yxfb_f32, memory::format::yxfb_f32);
        auto val_fw = convolution_gpu::create;

        conv_fw_implementation_map.insert( {key_fw, val_fw} );

        gpu::kernel_templates::add(kernelName_BFXY_f32, kernelCode_BFXY_f32);
        auto key_fw2 = std::make_tuple(engine::gpu, memory::format::bfyx_f32, memory::format::bfyx_f32);
        conv_fw_implementation_map.insert({ key_fw2, val_fw });
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
