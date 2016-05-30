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
#include <cstring>
#include <immintrin.h>
#include <mutex>

#include "pooling_cpu_avx2_batch24.h"
#include "pooling.h"

const uint64_t BATCH_ACCEPTED_BLOCK = 24;                         //the batch size that is minimal required for usage with jit version
const uint64_t BATCH_SHIFT = 8;                                   //the size of register used for shifting with batch layout / number if pics/floats that are processed at the same time
const uint64_t BATCH_BLOCKS = BATCH_ACCEPTED_BLOCK / BATCH_SHIFT; //number of registers (blocks to process) in the batch format
const uint64_t BUFFERS_ALIGNMENT = BATCH_SHIFT * sizeof(float);   //required alignment of all buffers used by jit primitives

namespace {

int b_pos = 0; // todo typetraits
int f_pos = 1;
int x_pos = 2;
int y_pos = 3;


void naive(float* input, float* output, neural::vector<uint32_t> input_dims, neural::vector<uint32_t> pooling_dims) {
    const uint64_t batch_accs = BATCH_ACCEPTED_BLOCK / BATCH_SHIFT;
    const uint64_t in_pixel_size = input_dims.raw[1] * BATCH_ACCEPTED_BLOCK;

    std::memcpy(output, input, in_pixel_size * sizeof(float));
    for (uint64_t i = 0; i < pooling_dims.raw[3]; ++i)
    {
        for (uint64_t j = 0; j < pooling_dims.raw[2]; ++j)
        {
            if ((i == 0) && (j == 0)) continue;
            auto curr_output = output;
            auto curr_input = input + (i * input_dims.raw[2] + j) * in_pixel_size;
            for (uint64_t feat = 0u; feat < input_dims.raw[1]; ++feat)
            {
                for (uint64_t acc_index = 0; acc_index < batch_accs; ++acc_index)
                {
                    auto result = _mm256_max_ps(_mm256_loadu_ps(curr_output), _mm256_loadu_ps(curr_input));
                    _mm256_storeu_ps(curr_output, result);

                    curr_output += BATCH_SHIFT;
                    curr_input += BATCH_SHIFT;
                }
            }
        }
    }
}
}

namespace neural {

pooling_cpu_avx2_batch24::pooling_cpu_avx2_batch24(pooling &arg)
    : is_an_implementation(neural::type_id<pooling_cpu_avx2_batch24>())
    , outer(arg)
{
    uint64_t output_view_width  = arg.argument.output_size.raw[x_pos];
    uint64_t output_view_height = arg.argument.output_size.raw[y_pos];

    uint64_t input_width  = arg.input_memory(0).argument.size.raw[x_pos];
    uint64_t input_height = arg.input_memory(0).argument.size.raw[y_pos];
    uint64_t input_depth  = arg.input_memory(0).argument.size.raw[f_pos];

    uint64_t output_width  = arg.output_memory(0).argument.size.raw[x_pos];
    uint64_t output_height = arg.output_memory(0).argument.size.raw[y_pos];
    uint64_t output_depth  = arg.output_memory(0).argument.size.raw[f_pos];

    uint64_t column_end = arg.argument.output_offset.raw[x_pos] + output_view_width;
    uint64_t row_end    = arg.argument.output_offset.raw[y_pos] + output_view_height;

    uint64_t row_begin    = arg.argument.output_offset.raw[y_pos];
    uint64_t column_begin = arg.argument.output_offset.raw[x_pos];

    uint64_t input_pixel_size  = BATCH_ACCEPTED_BLOCK * input_depth;
    uint64_t output_pixel_size = BATCH_ACCEPTED_BLOCK * output_depth;

    if(arg.input_memory(0).argument.size.raw[b_pos] != arg.output_memory(0).argument.size.raw[b_pos])
        throw std::runtime_error("blabla");

    uint64_t num_batch_packages = arg.input_memory(0).argument.size.raw[b_pos] / BATCH_ACCEPTED_BLOCK;

    auto& stride = arg.argument.stride;

    for(uint64_t batch = 0; batch < num_batch_packages; ++batch)
        for(uint64_t row = row_begin; row < row_end; ++row)
            for(uint64_t column = column_begin; column < column_end; ++column)
            {
                uint64_t output_offset = ((batch * output_height + row) * output_width + column) * output_pixel_size;
                uint64_t in_base_row = row * stride.raw[3];
                uint64_t in_base_col = column * stride.raw[2];
                uint64_t input_offset = ((batch * input_height + in_base_row) * input_width + in_base_col) * input_pixel_size;

                tasks_parameters.push_back(
                    parameters_pooling_f32
                    {
                        &arg.input_memory(0).pointer,
                        &arg.output_memory(0).pointer,
                        input_offset,
                        output_offset,
                        &arg
                    });
            }

    for(auto& parameter : tasks_parameters)
        tasks.push_back(task{implementation, &parameter});
};

pooling_cpu_avx2_batch24::~pooling_cpu_avx2_batch24() {};
void pooling_cpu_avx2_batch24::implementation(const void *ptr) 
{
    auto this_handle        = static_cast<const parameters_pooling_f32*>(ptr);
    auto this_pooling       = static_cast<const pooling *>(this_handle->pooling_data);

    auto& window            = this_pooling->argument.size;

    auto& input_memory_arg  = this_pooling->input_memory(0).argument;
    auto& input_buffer_size = input_memory_arg.size;

    auto input  = reinterpret_cast<float**>(this_handle->input_ptr);
    auto output = reinterpret_cast<float**>(this_handle->output_ptr);

    uint64_t input_offset = this_handle->input_offset;
    uint64_t output_offset = this_handle->output_offset;

    naive(*input + input_offset, *output + output_offset, input_buffer_size, window);
}

namespace
{

    struct attach
    {
        attach()
        {
            auto key_fw = std::make_tuple(engine::cpu, memory::format::bs_yxf_bv24_f32, memory::format::bs_yxf_bv24_f32);
            auto val_fw = pooling_cpu_avx2_batch24::create;

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
