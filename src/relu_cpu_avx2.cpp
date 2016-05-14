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

const uint16_t C_simd_width = sizeof(__m256) / sizeof(float);

static __m256i simd_masks[] = {
    _mm256_setr_epi32(-1,  0,  0,  0,  0,  0,  0, 0),
    _mm256_setr_epi32(-1, -1,  0,  0,  0,  0,  0, 0),
    _mm256_setr_epi32(-1, -1, -1,  0,  0,  0,  0, 0),
    _mm256_setr_epi32(-1, -1, -1, -1,  0,  0,  0, 0),
    _mm256_setr_epi32(-1, -1, -1, -1, -1,  0,  0, 0),
    _mm256_setr_epi32(-1, -1, -1, -1, -1, -1,  0, 0),
    _mm256_setr_epi32(-1, -1, -1, -1, -1, -1, -1, 0)
};

// process chunks of data
struct relu_avx2_worker : public neural::is_an_implementation
{
    #pragma pack(push, 1)
    struct task_data_t {
        const relu *relu_layer;
        size_t offset;
        size_t size;
    };
    #pragma pack(pop)

    std::vector<neural::task> tasks;
    std::vector<task_data_t> task_data;

    relu_avx2_worker(const void *outher, bool forward) : is_an_implementation(neural::type_id<relu_avx2_worker>()) {
        auto relu_layer = static_cast<const relu *>(outher);

        auto output_mem_count = relu_layer->output_memory(0).count();

        //todo: determine chunks of data from analysis
        auto chunks_count = 1u;

        auto chunk_size = output_mem_count/chunks_count;

        tasks.resize(chunks_count);
        task_data.resize(chunks_count);
        for (auto i = 0u; i < tasks.size(); ++i) {
            auto offset = i * chunk_size;
            auto size = (i < chunks_count-1) ? (i+1) * chunk_size - offset : output_mem_count - offset;

            task_data[i] = {relu_layer, offset, size};
            if(forward) tasks[i] = { reinterpret_cast<void(*)(const void*)>(process_forward_chunk_of_task_data), &task_data[i] };
            else tasks[i] = { reinterpret_cast<void(*)(const void*)>(process_backward_chunk_of_task_data), &task_data[i] };
        }
    }

    static void calculate_activation_result(uint16_t vector_size, float *input, float *output)
    {
        if(vector_size == 0) return;

        __m256 zero = _mm256_setzero_ps();

        if(vector_size == C_simd_width) {
            __m256 dst = _mm256_max_ps(zero, _mm256_load_ps(input));
            _mm256_storeu_ps(output, dst);
        }
        else {
            __m256 dst = _mm256_max_ps(zero, _mm256_maskload_ps(input, simd_masks[vector_size-1]));
            _mm256_maskstore_ps(output, simd_masks[vector_size-1], dst);
        }
    }

    static void calculate_activation_result(uint16_t vector_size, float *forward_input, float *backward_input, float *backward_output)
    {
        if(vector_size == 0) return;

        __m256 zero = _mm256_setzero_ps();

        if(vector_size == C_simd_width) {
            __m256 dst = _mm256_cmp_ps(zero, _mm256_load_ps(forward_input), _CMP_LT_OQ);
            dst = _mm256_and_ps(dst, _mm256_load_ps(backward_input));
            _mm256_storeu_ps(backward_output, dst);
        }
        else {
            __m256 dst = _mm256_cmp_ps(zero, _mm256_maskload_ps(forward_input, simd_masks[vector_size-1]), _CMP_LT_OQ);
            dst = _mm256_and_ps(dst, _mm256_maskload_ps(backward_input, simd_masks[vector_size-1]));
            _mm256_maskstore_ps(backward_output, simd_masks[vector_size-1], dst);
        }
    }

    static void process_forward_chunk_of_task_data(const task_data_t *data) {
        auto input = static_cast<float*>(data->relu_layer->input_memory(0).pointer);
        auto output = static_cast<float*>(data->relu_layer->output_memory(0).pointer);

        input += data->offset;
        output += data->offset;
        auto output_mem_count = data->size;
        auto chunks_count = output_mem_count/C_simd_width;

        for (auto i = 0u; i < chunks_count; ++i) {
            calculate_activation_result(C_simd_width, input, output);

            input += C_simd_width;
            output += C_simd_width;
        }
        calculate_activation_result(output_mem_count % C_simd_width, input, output);
    }

    static void process_backward_chunk_of_task_data(const task_data_t *data) {
        auto backward_input  = static_cast<float*>(data->relu_layer->input_memory(0).pointer);
        auto forward_input   = static_cast<float*>(data->relu_layer->input_memory(1).pointer);
        auto backward_output = static_cast<float*>(data->relu_layer->output_memory(0).pointer);

        backward_input += data->offset;
        forward_input += data->offset;
        backward_output += data->offset;

        auto output_mem_count = data->size;
        auto chunks_count = output_mem_count/C_simd_width;

        for (auto i = 0u; i < chunks_count; ++i) {
            calculate_activation_result(C_simd_width, forward_input, backward_input, backward_output);

            forward_input += C_simd_width;
            backward_output += C_simd_width;
            backward_input += C_simd_width;
        }
        calculate_activation_result(output_mem_count % C_simd_width, forward_input, backward_input, backward_output);
    }

    std::vector<neural::task> work() {
        return this->tasks;
    }
};

relu_cpu_avx2::relu_cpu_avx2(relu &arg)
    : is_an_implementation(neural::type_id<relu_cpu_avx2>())
    , outer(arg)
{
    auto this_relu = static_cast<const relu *>(&outer);

    auto& input_offset = this_relu->argument.input_offset;
    auto& output_offset = this_relu->argument.output_offset;
    auto& output_size = this_relu->argument.output_size;

    auto& input_mem_arg  = this_relu->input_memory(0).argument;
    auto& output_mem_arg = this_relu->output_memory(0).argument;

    if (input_mem_arg.format != memory::format::yxfb_f32)                throw std::runtime_error("ReLU reference uses yxfb_f32 format.");
    if (input_mem_arg.format != output_mem_arg.format)                   throw std::runtime_error("ReLU input/output data format does not match.");
    if (input_mem_arg.size.raw.size() != output_mem_arg.size.raw.size()) throw std::runtime_error("ReLU input/output number of dimension does not match.");
    for (auto &x : input_offset.raw)                          if (x > 0) throw std::runtime_error("ReLU input offset must be equal to zero.");
    for (auto &x : output_offset.raw)                         if (x > 0) throw std::runtime_error("ReLU output offset must be equal to zero.");

    assert(1 == this_relu->argument.input.size());
    assert(1 == this_relu->argument.output.size());
    assert(1 == output_size.feature.size());
    assert(1 == output_size.batch.size());

    relu_ptr.reset(new relu_avx2_worker(this_relu, true));
};

relu_cpu_avx2::~relu_cpu_avx2() {}

relu_backward_cpu_avx2::relu_backward_cpu_avx2(relu_backward &arg)
: is_an_implementation(neural::type_id<relu_backward_cpu_avx2>())
, outer(arg)
{
    auto this_relu = static_cast<const relu_backward *>(&outer);

    if (this_relu->input().size() != 2)
        throw std::runtime_error("ReLU backward: number of inputs is incorrect.");

    if (this_relu->output().size() != 1)
        throw std::runtime_error("ReLU backward: number of outputs is incorrect.");

    auto forward_output_grad_arg = this_relu->argument.input[0].primitive.as<const memory&>().argument;
    auto forward_output_grad_offset = this_relu->argument.input_offset[0];

    auto forward_input_arg = this_relu->argument.input[1].primitive.as<const memory&>().argument;
    auto forward_input_offset = this_relu->argument.input_offset[1];

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

    relu_ptr.reset(new relu_avx2_worker(this_relu, false));
};

relu_backward_cpu_avx2::~relu_backward_cpu_avx2() {}

namespace {
struct attach {
    attach() {
        auto key_fw = std::make_tuple(engine::cpu, memory::format::yxfb_f32, memory::format::yxfb_f32);
        auto key_bw = std::make_tuple(engine::cpu, memory::format::yxfb_f32, memory::format::yxfb_f32);
        auto val_fw = relu_cpu_avx2::create;
        auto val_bw = relu_backward_cpu_avx2::create;

        relu_fw_implementation_map::instance().insert( {key_fw, val_fw} );
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
