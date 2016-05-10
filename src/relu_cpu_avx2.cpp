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

#include "relu_cpu_avx2.h"
#include "multidimensional_counter.h"

namespace neural {

const uint32_t C_simd_width = sizeof(__m256) / sizeof(float);
const uint32_t C_num_max_acc = 15;


template<uint32_t T_num_acc>
void forward_inner_macro(float *input, float *output)
{
    __m256 acc[T_num_acc];
    __m256 zero = _mm256_setzero_ps();

    for (uint32_t acc_id = 0; acc_id < T_num_acc; ++acc_id)
        acc[acc_id] = _mm256_max_ps(zero, _mm256_load_ps(input + acc_id * C_simd_width));

    for (uint32_t acc_id = 0; acc_id < T_num_acc; ++acc_id)
        _mm256_storeu_ps(output + acc_id * C_simd_width, acc[acc_id]);
}


template<uint32_t T_num_acc>
void backward_inner_macro(float *forward_input, float *backward_input, float *backward_output)
{
    __m256 acc[T_num_acc];
    __m256 zero = _mm256_setzero_ps();

    for (uint32_t acc_id = 0; acc_id < T_num_acc; ++acc_id)
        acc[acc_id] = _mm256_cmp_ps(zero, _mm256_load_ps(forward_input + acc_id * C_simd_width), _CMP_LT_OQ);

    for (uint32_t acc_id = 0; acc_id < T_num_acc; ++acc_id)
        acc[acc_id] = _mm256_and_ps(acc[acc_id], _mm256_load_ps(backward_input + acc_id * C_simd_width));

    for (uint32_t acc_id = 0; acc_id < T_num_acc; ++acc_id)
        _mm256_storeu_ps(backward_output + acc_id * C_simd_width, acc[acc_id]);
}


relu_cpu_avx2::relu_cpu_avx2(relu &arg)
    : is_an_implementation(neural::type_id<relu_cpu_avx2>())
    , outer(arg)
{};


relu_cpu_avx2::~relu_cpu_avx2() {}


void relu_cpu_avx2::implementation(const void *ptr) {
    auto this_relu = static_cast<const relu *>(ptr);

    auto& input_offset = this_relu->argument.input_offset;
    auto& output_offset = this_relu->argument.output_offset;
    auto& output_size = this_relu->argument.output_size;

    //auto& input_mem_arg  = this_relu->input_memory(0).argument;
    //auto& output_mem_arg = this_relu->output_memory(0).argument;
    auto& input_mem_arg = this_relu->argument.input[0].primitive.as<const memory&>().argument; //todo tmp solution
    auto& output_mem_arg = this_relu->argument.output[0].as<const memory&>().argument;

    if (input_mem_arg.format != memory::format::yxfb_f32)                throw std::runtime_error("ReLU reference uses yxfb_f32 format.");
    if (input_mem_arg.format != output_mem_arg.format)                   throw std::runtime_error("ReLU input/output data format does not match.");
    if (input_mem_arg.size.raw.size() != output_mem_arg.size.raw.size()) throw std::runtime_error("ReLU input/output number of dimension does not match.");
    for (auto &x : input_offset.raw)                         if (x == 0) throw std::runtime_error("ReLU input offset must be equal to zero.");
    for (auto &x : output_offset.raw)                        if (x == 0) throw std::runtime_error("ReLU output offset must be equal to zero.");

    assert(1 == this_relu->argument.input.size());
    assert(1 == this_relu->argument.output.size());
    assert(1 == output_size.feature.size());
    assert(1 == output_size.batch.size());

    //auto input  = static_cast<float*>(this_relu->input_memory(0).pointer);
    //auto output = static_cast<float*>(this_relu->output_memory(0).pointer);
    auto input = static_cast<float*>(this_relu->argument.input[0].primitive.as<const memory&>().pointer);  //todo tmp solution
    auto output = static_cast<float*>(this_relu->argument.output[0].as<const memory&>().pointer);

    auto output_mem_count = this_relu->argument.output[0].as<const memory&>().count();

    auto full_passes = output_mem_count / (C_num_max_acc * C_simd_width);
    auto partial_pass_size = output_mem_count % (C_num_max_acc * C_simd_width) / C_simd_width;

    for (uint32_t pass = 0; pass < full_passes; ++pass)
    {
        forward_inner_macro<C_num_max_acc>(input, output);

        input += C_num_max_acc * C_simd_width;
        output += C_num_max_acc * C_simd_width;
    }

    switch (partial_pass_size)
    {
        case  1: forward_inner_macro< 1>(input, output); break;
        case  2: forward_inner_macro< 2>(input, output); break;
        case  3: forward_inner_macro< 3>(input, output); break;
        case  4: forward_inner_macro< 4>(input, output); break;
        case  5: forward_inner_macro< 5>(input, output); break;
        case  6: forward_inner_macro< 6>(input, output); break;
        case  7: forward_inner_macro< 7>(input, output); break;
        case  8: forward_inner_macro< 8>(input, output); break;
        case  9: forward_inner_macro< 9>(input, output); break;
        case 10: forward_inner_macro<10>(input, output); break;
        case 11: forward_inner_macro<11>(input, output); break;
        case 12: forward_inner_macro<12>(input, output); break;
        case 13: forward_inner_macro<13>(input, output); break;
        case 14: forward_inner_macro<14>(input, output); break; //C_num_max_acc - 1
    }
}


relu_backward_cpu_avx2::relu_backward_cpu_avx2(relu_backward &arg)
: is_an_implementation(neural::type_id<relu_backward_cpu_avx2>())
, outer(arg)
{};


relu_backward_cpu_avx2::~relu_backward_cpu_avx2() {}


void relu_backward_cpu_avx2::implementation(const void *ptr)
{
    auto this_relu = static_cast<const relu_backward *>(ptr);

    if (this_relu->input().size() != 2)
        throw std::runtime_error("ReLU backward: number of inputs is incorrect.");

    if (this_relu->output().size() != 1)
        throw std::runtime_error("ReLU backward: number of outputs is incorrect.");

    //auto backward_input = static_cast<float*>(this_relu->input_memory(0).pointer);
    //auto forward_input       = static_cast<float*>(this_relu->input_memory(1).pointer);
    //auto backward_output  = static_cast<float*>(this_relu->output_memory(0).pointer);
    auto backward_input = static_cast<float*>(this_relu->argument.input[0].primitive.as<const memory&>().pointer);
    auto forward_input = static_cast<float*>(this_relu->argument.input[1].primitive.as<const memory&>().pointer);
    auto backward_output = static_cast<float*>(this_relu->argument.output[0].as<const memory&>().pointer);

    //auto forward_output_grad_arg    = this_relu->input_memory(0).argument;
    auto forward_output_grad_arg = this_relu->argument.input[0].primitive.as<const memory&>().argument;
    auto forward_output_grad_offset = this_relu->argument.input_offset[0];

    //auto forward_input_arg    = this_relu->input_memory(1).argument;
    auto forward_input_arg = this_relu->argument.input[1].primitive.as<const memory&>().argument;
    auto forward_input_offset = this_relu->argument.input_offset[1];

    //auto forward_input_grad_arg    = this_relu->output_memory(0).argument;
    auto forward_input_grad_arg = this_relu->argument.output[0].as<const memory&>().argument;
    auto forward_input_grad_offset = this_relu->argument.output_offset;

    auto processed_window_sizes = this_relu->argument.output_size;

    if (forward_output_grad_arg.size.raw.size() != forward_input_arg.size.raw.size() || forward_input_arg.size.raw.size() != forward_input_grad_arg.size.raw.size())
        throw std::runtime_error("ReLU backward: number of IO dimension does not match.");

    if (forward_output_grad_arg.format != forward_input_arg.format || forward_input_arg.format != forward_input_grad_arg.format)
        throw std::runtime_error("ReLU backward: IO data format does not match.");

    for (size_t i = 0; i < forward_output_grad_arg.size.raw.size(); ++i) {
        if (forward_output_grad_arg.size.raw[i] < processed_window_sizes.raw[i] + forward_output_grad_offset.raw[i]) throw std::runtime_error("ReLU backward: backward_input size does not match the offset.");
        if (forward_input_arg.size.raw[i]       < processed_window_sizes.raw[i] + forward_input_offset.raw[i])       throw std::runtime_error("ReLU backward: forward_input size does not match the offset.");
        if (forward_input_grad_arg.size.raw[i]  < processed_window_sizes.raw[i] + forward_input_grad_offset.raw[i])  throw std::runtime_error("ReLU backward: backward_output size does not match the offset.");
    }

    auto output_mem_count = this_relu->argument.output[0].as<const memory&>().count();

    auto full_passes = output_mem_count / (C_num_max_acc * C_simd_width);
    auto partial_pass_size = output_mem_count % (C_num_max_acc * C_simd_width) / C_simd_width;

    for (uint32_t pass = 0; pass < full_passes; ++pass)
    {
        backward_inner_macro<C_num_max_acc>(forward_input, backward_input, backward_output);

        forward_input += C_num_max_acc * C_simd_width;
        backward_output += C_num_max_acc * C_simd_width;
        backward_input += C_num_max_acc * C_simd_width;
    }

    switch (partial_pass_size)
    {
        case  1: backward_inner_macro< 1>(forward_input, backward_input, backward_output); break;
        case  2: backward_inner_macro< 2>(forward_input, backward_input, backward_output); break;
        case  3: backward_inner_macro< 3>(forward_input, backward_input, backward_output); break;
        case  4: backward_inner_macro< 4>(forward_input, backward_input, backward_output); break;
        case  5: backward_inner_macro< 5>(forward_input, backward_input, backward_output); break;
        case  6: backward_inner_macro< 6>(forward_input, backward_input, backward_output); break;
        case  7: backward_inner_macro< 7>(forward_input, backward_input, backward_output); break;
        case  8: backward_inner_macro< 8>(forward_input, backward_input, backward_output); break;
        case  9: backward_inner_macro< 9>(forward_input, backward_input, backward_output); break;
        case 10: backward_inner_macro<10>(forward_input, backward_input, backward_output); break;
        case 11: backward_inner_macro<11>(forward_input, backward_input, backward_output); break;
        case 12: backward_inner_macro<12>(forward_input, backward_input, backward_output); break;
        case 13: backward_inner_macro<13>(forward_input, backward_input, backward_output); break;
        case 14: backward_inner_macro<14>(forward_input, backward_input, backward_output); break; //C_num_max_acc - 1
    }
}


namespace {
struct attach {
    attach() {
        auto key_fw = std::make_tuple(engine::reference, memory::format::yxfb_f32, memory::format::yxfb_f32);
        auto key_bw = std::make_tuple(engine::reference, memory::format::yxfb_f32, memory::format::yxfb_f32);
        auto val_fw = relu_cpu_avx2::create;
        auto val_bw = relu_backward_cpu_avx2::create;

        relu_fw_implementation_map::instance().insert( {key_fw, val_fw} ); //todo keys should be different
        relu_bw_implementation_map::instance().insert( {key_bw, val_bw} );
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
