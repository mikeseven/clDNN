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
#include "ocl_toolkit.h"

#pragma warning(disable: 4189)
namespace neural {

    /*int get_global_id(int dim)
    {
        static int x = 0;
        if (dim == 0)
            return x++;
        else
            return 0;
    }

    void test_convolution_GPU(float* input, cl_uint4 input_size, float* filter, cl_uint4 filter_size, float* bias, float* pDst, cl_uint4 dst_size, cl_uint2 spatial_stride)
    {
        int global_id = get_global_id(0);
        const int batch_num = dst_size.s0;
        const int batch_offset = global_id % dst_size.s0;

        const int idx = global_id / batch_num;

        const int x = (idx % input_size.s2) * spatial_stride.s0;
        const int y = (idx * spatial_stride.s1) / input_size.s2;

        const int out_offset = idx * batch_num + batch_offset;

        pDst[out_offset] = 0;
        for (cl_uint i = 0; i < filter_size.s3; i++)
        {
            for (cl_uint j = 0; j < filter_size.s2; j++)
            {
                int input_idx = (x + j + ((y + i) * input_size.s2)) * batch_num + batch_offset;
                int filter_idx = i * filter_size.s2 + j;
                pDst[out_offset] += input[input_idx] * filter[filter_idx];
            }
        }
        pDst[out_offset] += bias[0];
    }*/

    const std::string kernelCode = R"__krnl(
__kernel void Convolution_GPU(
    const __global neural_memory* input_mem,
    const __global neural_memory* filter_mem,
    float bias,
    __global neural_memory* dst_mem,
    uint2 spatial_stride)
{

//
    const __global uint* input_size = get_raw(input_mem);
    const __global uint* filter_size = get_raw(filter_mem);
    const __global uint* dst_size = get_raw(dst_mem);
    const __global float* input = (__global float*)get_data(input_mem);
    const __global float* filter = (__global float*)get_data(filter_mem);
    __global float* pDst = (__global float*)get_data(dst_mem);
//

    int global_id = get_global_id(0);
    const int batch_num = dst_size[0];
    const int batch_offset = global_id % dst_size[0];

    const int idx = global_id / batch_num;

    const int x = (idx % input_size[2]) * spatial_stride.s0;
    const int y = (idx * spatial_stride.s1) / input_size[2];

    const int out_offset = idx * batch_num + batch_offset;

    pDst[out_offset] = 0;
    for (uint i = 0; i < filter_size[4]; i++)
    {
        for (uint j = 0; j < filter_size[3]; j++)
        {
            int input_idx = (x + j + ((y + i) * input_size[2])) * batch_num + batch_offset;
            int filter_idx = i * filter_size[3] + j;
            pDst[out_offset] += input[input_idx] * filter[filter_idx];
        }
    }
    pDst[out_offset] += bias;
}
)__krnl";

static cl_uint4 get_memory_sizes(const neural::memory& mem)
{
    auto size = mem.argument.size;
    if (size.raw.size() > 4) {
        throw std::runtime_error("Want to get size, but this is not a vector of sizes, because size is greater than 4!");
    }

    cl_uint4 x = { 0 };
    std::copy(std::begin(size.raw), std::end(size.raw), x.s);
    return x;
}

convolution_gpu::convolution_gpu(convolution &arg)
        : is_an_implementation(neural::type_id<convolution_cpu_reference>())
        , outer(arg) {};
convolution_gpu::~convolution_gpu() {};
void convolution_gpu::implementation(const void *ptr) {

    auto this_conv = static_cast<const convolution *>(ptr);

    auto& output_size   = this_conv->argument.output_size;
    auto& padding       = this_conv->argument.padding;
    auto& stride        = this_conv->argument.stride;

    auto& filter_arg = this_conv->argument.input[1].primitive.as<const memory&>().argument; //convolution filter

    assert( output_size.feature[0] == filter_arg.size.feature[0] ); // memory::format oixy

    // todo remove
    if(filter_arg.format != memory::format::oiyx_f32) throw std::runtime_error("conv weights arent oiyx_f32 format");

    auto& input_mem  = this_conv->input_memory(0);
    auto& output_mem = this_conv->output_memory(0);
    auto& filter_mem = this_conv->argument.input[1].primitive.as<const memory&>();
    auto bias   = static_cast<float*>(this_conv->argument.input[2].primitive.as<const memory&>().pointer);

    const int f_pos = 1; // neural::vector format is b,f,spatials. In input and output 'b' and 'f' fields are always scalars.
    namespace nd = ndimensional;
    nd::value<uint32_t> range (output_size.raw);

    // weights neural::vector is: {b}, {ofm, ifm} {spatials}
    // ofm - output feature maps
    // ifm - input feature maps
    // b = 1 always
    // (weights can't have batch so it is equall to 1)
    // Ofm and batch is cropped, ofm will be hold manually
    // Batch is included in output size

    //
    //auto inBuffer = toolkit.create_input_buffer<float>(input_mem);
    //auto inputSizes = toolkit.get_memory_sizes(input_mem);

    //auto filterBuffer = toolkit.create_input_buffer<float>(filter_mem);
    //auto filterSizes = toolkit.get_memory_sizes(filter_mem);

    //auto outBuffer = toolkit.create_output_buffer<float>(output_mem);
    auto dstSizes = get_memory_sizes(output_mem);
    size_t dstSize = dstSizes.s0 * dstSizes.s1 * dstSizes.s2 * dstSizes.s3;

    auto biasSizes = get_memory_sizes(this_conv->argument.input[2].primitive.as<const memory&>());
    cl_uint2 spatialStride = { stride.spatial[0], stride.spatial[1] };
    
    //for(int i = 0; i < dstSize; i++)
    //    test_convolution_GPU((float*)input_mem.pointer, inputSizes, (float*)filter_mem.pointer, filterSizes, bias, (float*)output_mem.pointer, dstSizes, spatialStride);
    //

    switch(padding){
        case padding::zero:
        {
            auto kernel = gpu::kernel<gpu::input_mem, gpu::input_mem, cl_float, gpu::output_mem, cl_uint2>("Convolution_GPU");
            kernel({ dstSize, 1 }, input_mem, filter_mem, bias[0], output_mem, spatialStride);
        }
            break;
        default:
            throw std::runtime_error("Unknown padding mode in convolution.");
    }
}

/*convolution_backward_cpu_reference::convolution_backward_cpu_reference(convolution_backward &arg)
    : is_an_implementation(neural::type_id<convolution_backward_cpu_reference>())
    , outer(arg) {};
convolution_backward_cpu_reference::~convolution_backward_cpu_reference() {};
void convolution_backward_cpu_reference::implementation(const void *ptr) { //todo tests
    auto this_bw_conv = static_cast<const convolution_backward *>(ptr);

    auto& bw_input_size    = this_bw_conv->argument.input_size;  // todo output or input?
    auto& bw_input_offset  = this_bw_conv->argument.input_offset;
    auto& bw_output_offset = this_bw_conv->argument.output_offset;
    auto& stride           = this_bw_conv->argument.stride;
    auto& padding          = this_bw_conv->argument.padding;

    auto& bw_input_arg     = this_bw_conv->input_memory(0).argument;
    auto& fw_input_arg     = this_bw_conv->input_memory(1).argument;
    auto& filter_arg       = this_bw_conv->input_memory(2).argument;
    auto& bias_arg         = this_bw_conv->input_memory(3).argument; //todo bias isn't needed in bw conv. It is only used to compare its size with bias_diff. Remove?

    auto& bw_output_arg    = this_bw_conv->output_memory(0).argument;
    auto& filter_diff_arg  = this_bw_conv->output_memory(1).argument;
    auto& bias_diff_arg    = this_bw_conv->output_memory(2).argument;

    assert( 1 == bw_input_size.feature.size() );
    assert( 1 == bw_input_size.batch.size()   );

    if(bw_input_size.raw.size()   != bw_output_arg.size.raw.size())   throw std::runtime_error("Backward convolution bw_input/bw_output number of dimension does not match.");
    if(stride.raw.size()          != bw_output_arg.size.raw.size())   throw std::runtime_error("Backward convolution stride/bw_output number of dimension does not match.");
    if(bw_input_size.raw.size()   != fw_input_arg.size.raw.size())    throw std::runtime_error("Backward convolution bw_input/fw_output number of dimension does not match.");
    if(filter_arg.size.raw.size() != bw_output_arg.size.raw.size())   throw std::runtime_error("Backward convolution filter size/bw_output number of dimension does not match.");
    if(filter_arg.size.raw.size() != filter_diff_arg.size.raw.size()) throw std::runtime_error("Backward convolution weights/weights_diff number of dimension does not match.");
    if(bw_input_arg.format        != bw_output_arg.format)            throw std::runtime_error("Backward convolution bw_input/bw_output data format does not match.");
    if(bw_input_arg.format        != filter_arg.format)               throw std::runtime_error("Backward convolution bw_input/weights data format does not match.");
    if(bw_input_arg.format        != fw_input_arg.format)             throw std::runtime_error("Backward convolution bw_input/fw_output data format does not match.");
    if(bias_arg.size.raw.size()   != 3 &&
       bias_arg.size.batch[0]     != 1 &&
       bias_arg.size.feature[0]   != 1)                               throw std::runtime_error("Backward convolution biases isn't 1D vector.");
    if(bias_arg.size.raw.size()   != bias_diff_arg.size.raw.size())   throw std::runtime_error("Backward convolution bias/bias_diff number dimensions doesn't match.");
    if(bias_arg.size.spatial[0]   != bw_input_arg.size.feature[0])    throw std::runtime_error("Backward convolution biases/bw_input dimensions does not match.");
    if(bias_arg.size              != bias_diff_arg.size)              throw std::runtime_error("Backward convolution bias/bias_diff size doesn't match.");

    auto bw_input     = static_cast<float*>(this_bw_conv->input_memory(0).pointer);
    auto fw_input     = static_cast<float*>(this_bw_conv->input_memory(1).pointer);
    auto weights      = static_cast<float*>(this_bw_conv->input_memory(2).pointer);
    //todo fw bias is used only for size check, is it needed?

    auto bw_output    = static_cast<float*>(this_bw_conv->output_memory(0).pointer);
    auto weights_diff = static_cast<float*>(this_bw_conv->output_memory(1).pointer);
    auto bias_diff    = static_cast<float*>(this_bw_conv->output_memory(2).pointer);

    //todo review conditions below
    for(size_t i = 0; i < bw_output_offset.raw.size(); ++i){
        // general formula for forward: output size = (input size - filter size) / step + 1
        if(bw_input_size.raw[i] <
            std::abs(static_cast<int32_t>(bw_output_arg.size.raw[i] - bw_output_offset.raw[i] - filter_arg.size.raw[i])) / stride.raw[i] + 1) //todo is it safe?
            if(filter_arg.size.raw[i] <= bw_input_size.raw[i])
                throw std::runtime_error("Output size of bw convolution is to small.");

        if(bw_input_arg.size.raw[i] < bw_input_size.raw[i] + bw_output_offset.raw[i])
            throw std::runtime_error("Backward convolution bw_input buffer size is to small.");

        if(bw_output_arg.size.raw[i] != fw_input_arg.size.raw[i])
            throw std::runtime_error("Sizes of BW output and FW input buffers in convolution bw must be equal.");
    }

    // initializie gradients with 0
    fill(this_bw_conv->output_memory(0), 0.0f);
    fill(this_bw_conv->output_memory(1), 0.0f);
    fill(this_bw_conv->output_memory(2), 0.0f);

    const int F_POS = 1;
    namespace nd = ndimensional;
    nd::value<uint32_t> bias_range (bias_arg.size);
    nd::value<uint32_t> range (bw_input_size); //todo in/out size?
    nd::value<uint32_t> window_range (filter_arg.size);
    auto calc_in_idx   = nd::choose_calculate_idx(bw_input_arg.format);
    auto calc_out_idx  = nd::choose_calculate_idx(bw_output_arg.format);
    auto calc_win_idx  = nd::choose_calculate_idx(filter_arg.format);

    switch(padding){
        case padding::zero:
        {
            for(auto pos : range) {
                auto in_idx = calc_in_idx(bw_input_arg.size.raw , pos + bw_input_offset);

                for(auto win_pos : window_range){
                    const std::vector<uint32_t> arg_out_idx = nd::value<uint32_t>(bw_output_offset) + pos*stride + win_pos;

                    if( nd::is_out_of_range(bw_output_arg.size, arg_out_idx) )
                        continue;

                    auto out_idx = calc_out_idx(bw_output_arg.size.raw, arg_out_idx);
                    auto win_idx = calc_win_idx(filter_arg.size.raw, win_pos);

                    auto sensitivity = bw_input[in_idx] * weights[win_idx];

                    bw_output[out_idx] += sensitivity;
                    weights_diff[win_idx] += fw_input[out_idx] * sensitivity;
                }
                bias_diff[ pos[F_POS] ] += bw_input[in_idx];
            }
            break;
        }
        default:
            throw std::runtime_error("Unknown padding mode in backward convolution.");
    }
}*/

namespace{
struct attach{
    attach(){
        gpu::gpu_toolkit::get().add_kernel(kernelCode);
        auto key_fw = std::make_tuple(engine::gpu, memory::format::yxfb_f32, memory::format::yxfb_f32);
        auto val_fw = convolution_gpu::create;

        conv_fw_implementation_map.insert( {key_fw, val_fw} );
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
