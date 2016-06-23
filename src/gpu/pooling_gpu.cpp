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

#include "pooling_gpu.h"
#include "multidimensional_counter.h"
#include "ocl_toolkit.h"
#include "kernel.h"

const std::string kernelName = "Pooling_GPU_max";
const std::string kernelCode = R"__krnl(
__kernel void Pooling_GPU_max(__global neural_memory* input_mem, __global neural_memory* output_mem, const __global neural_vector* stride, const __global neural_vector* window)
{
    __global uint* input_size = get_raw(input_mem);
    __global float* input = (__global float*)get_data(input_mem);
    __global uint* output_size = get_raw(output_mem);
    __global float* output = (__global float*)get_data(output_mem);

    const int global_id = get_global_id(0);

    const int batch_num = output_size[0];
    const int batch_offset = global_id % batch_num;

    const int ofm_num = output_size[1];
    const int ofm_offset = (global_id / batch_num) % ofm_num;

    const int idx = (global_id / batch_num);

    const int i_fm_num = input_size[1];

    const __global uint* window_size = get_spatial(window);
    const __global uint* spatial_stride = get_spatial(stride);
    

    const int filter_application_count_x = output_size[2]; // how many times we need to apply filter in X dimension
    const int filter_application_count_y = output_size[3]; // how many times we need to apply filter in Y dimension

    const int offset_x = (idx / ofm_num) % filter_application_count_x * spatial_stride[0];
    const int offset_y = ((idx / ofm_num) / filter_application_count_x) % filter_application_count_y * spatial_stride[1]; 

    output[global_id] = -FLT_MAX;
    for(uint j = 0; j < window_size[1]; j++)
    {
        for(uint i = 0; i < window_size[0]; i++)
        {
            int input_idx = (i + offset_x + (j + offset_y) * input_size[2]) * batch_num * i_fm_num + ofm_offset * batch_num + batch_offset;
            output[global_id] = max(output[global_id], input[input_idx]);
        }
    }
}
)__krnl";

namespace neural {

    pooling_gpu::pooling_gpu(pooling &arg)
        : is_an_implementation(neural::type_id<pooling_gpu>())
        , outer(arg) {};
    pooling_gpu::~pooling_gpu() {};
    void pooling_gpu::implementation(const void *ptr) {
            auto this_pooling = static_cast<const pooling *>(ptr);
            
            // input
            auto& input_mem = this_pooling->input_memory(0);

            // output
            auto& output_mem = this_pooling->output_memory(0);



            auto& input_arg = this_pooling->argument.input[0].primitive.as<const memory&>().argument;

            auto& input_buffer_size = input_arg.size;
            auto& input_offset = this_pooling->argument.input_offset;

            auto& output_arg = this_pooling->argument.output[0].as<const memory&>().argument;
            auto& output_buffer_size = output_arg.size;
            auto& output_offset = this_pooling->argument.output_offset;
            auto& output_size = this_pooling->argument.output_size;

            auto& stride  = this_pooling->argument.stride;
            auto& window  = this_pooling->argument.size;
            auto& padding = this_pooling->argument.padding;

            if (padding::zero != padding)                                      throw std::runtime_error("Pooling support only zero padding.");
            if (input_arg.format != memory::format::yxfb_f32)                  throw std::runtime_error("Pooling reference uses yxfb_f32 format."); //todo, only this format?
            if (input_buffer_size.raw.size() != output_buffer_size.raw.size()) throw std::runtime_error("Pooling input/output number of dimension does not match.");
            if (stride.raw.size() != output_buffer_size.raw.size())            throw std::runtime_error("Pooling stride/output number of dimension does not match.");
            if (window.raw.size() != output_buffer_size.raw.size())            throw std::runtime_error("Pooling window_size/output number of dimension does not match.");
            if (input_arg.format != output_arg.format)                         throw std::runtime_error("Pooling input/output data format does not match.");

            size_t dstSize = output_mem.count();

            // general formula: output size = (input size - window size) / step + 1
            for (size_t i = 0; i < input_offset.raw.size(); ++i) {
                if (output_buffer_size.raw[i] < output_size.raw[i] + output_offset.raw[i])
                    throw std::runtime_error("Pooling output buffer size is to small.");
            }

            namespace nd = ndimensional;
            if (this_pooling->argument.mode == pooling::mode::max)
            {
                auto kernel = gpu::kernel<gpu::input_mem, gpu::output_mem, gpu::vector_arg, gpu::vector_arg>{ kernelName };
                kernel({ dstSize, std::min( dstSize, (size_t)16 ) }, input_mem, output_mem, stride, window);
            }
            else if (this_pooling->argument.mode == pooling::mode::average)
            {
            }
            else
            {
                throw std::runtime_error("Unknown pooling mode.");
            }
    };


namespace
{

    struct attach
    {
        attach()
        {
            gpu::kernel_templates::add(kernelName, kernelCode);
            auto key_fw = std::make_tuple(engine::gpu, memory::format::yxfb_f32, memory::format::yxfb_f32);
            auto val_fw = pooling_gpu::create;

            pool_fw_implementation_map::instance().insert( {key_fw, val_fw} );
        }

        ~attach()
        {
        }
    };

#ifdef __GNUC__
    __attribute__((visibility("default")))
#elif _MSC_VER
#   pragma section(".nn_init$m", read, write)
#endif
    attach attach_impl;

}
}
