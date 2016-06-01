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

#include "softmax_cpu_avx2.h"
#include "multidimensional_counter.h"
#include "algorithm"

// NN_CODE_UNREACHABLE signal to supporting compiler that specific location in code cannot be reached
#if defined _MSC_VER
#   define NN_UNREACHABLE_CODE __assume(0)
#endif

#if defined __GNUC__
#   if (__GNUC__ * 100 + __GNUC_MINOR__) >= 405
#       define NN_UNREACHABLE_CODE __builtin_unreachable()
#   else
#       define NN_UNREACHABLE_CODE
#   endif
#endif

#if defined __clang__
#   if __has_builtin(__builtin_unreachable)
#       define NN_UNREACHABLE_CODE __builtin_unreachable()
#   else
#       define NN_UNREACHABLE_CODE
#   endif
#endif

namespace neural {
namespace normalization {

const uint16_t C_simd_width = sizeof(__m256) / sizeof(float);

static const auto C_max_acc_batch1  = 12u;
static const auto C_max_acc_batch8  = 12u;
static const auto C_max_acc_batch48 = 12u;

static const auto C_batch8_size = C_simd_width;
static const auto C_batch48_size = 6 * C_simd_width;

static const auto C_data_stride_batch1 = C_simd_width * C_max_acc_batch1;


// process chunks of data
struct softmax_avx2_worker : public neural::is_an_implementation
{
    #pragma pack(push, 1)
    struct task_data_t {
        const softmax *softmax_layer;
    };
    #pragma pack(pop)

    std::vector<neural::task> tasks;
    std::vector<task_data_t> task_data;

    softmax_avx2_worker(const void *outher) : is_an_implementation(neural::type_id<softmax_avx2_worker>()) {
        auto softmax_layer = static_cast<const softmax *>(outher);

        tasks.resize(1);
        task_data.resize(1);

        task_data[0] = {softmax_layer};

        auto batch_size = softmax_layer->input_memory(0).argument.size.batch[0];

        if(batch_size == 1)       tasks[0] = { reinterpret_cast<void(*)(const void*)>(run_softmax_work_item_latency), &task_data[0] };
        else if(batch_size == 8)  tasks[0] = { reinterpret_cast<void(*)(const void*)>(run_softmax_work_item_batch8),  &task_data[0] };
        else if(batch_size == 48) tasks[0] = { reinterpret_cast<void(*)(const void*)>(run_softmax_work_item_batch48), &task_data[0] };
    }

    static inline __m256 _inner_mm256_exp_ps(__m256 arg)
    {
        __m256 mask = _mm256_cmp_ps(arg, _mm256_set1_ps(-87.336f), _CMP_GT_OQ);

        arg = _mm256_mul_ps(arg, _mm256_set1_ps(1.4426950408889634073599246810018921374266459541529859f));

        __m256i e = _mm256_add_epi32(
            _mm256_castps_si256(_mm256_cmp_ps(arg, _mm256_set1_ps(0.0f), _CMP_LT_OQ)),
            _mm256_cvttps_epi32(arg));

        arg = _mm256_sub_ps(arg, _mm256_cvtepi32_ps(e));

        __m256 intermediate_result;
        intermediate_result = _mm256_fmadd_ps(_mm256_set1_ps(0.0136779459179717f), arg, _mm256_set1_ps(0.0517692205767896f));
        intermediate_result = _mm256_fmadd_ps(intermediate_result, arg, _mm256_set1_ps(0.241554388295527f));
        intermediate_result = _mm256_fmadd_ps(intermediate_result, arg, _mm256_set1_ps(0.692998430056128f));
        intermediate_result = _mm256_fmadd_ps(intermediate_result, arg, _mm256_set1_ps(0.999999804292074f));
        arg = intermediate_result;

        __m256 res = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(e, _mm256_set1_epi32(127)), 23));

        res = _mm256_mul_ps(res, arg);

        return _mm256_and_ps(res, mask);
    }

    template<uint32_t T_SIZE, uint32_t T_batch_width>
    static void softmax_finalize_block(float* &output_ptr, __m256 &acc_sum)
    {
        // We are not using table of registers and unroll pragmas
        // due to compiler which have issues with register allocation
        // and needs special, obvious treatment. Template immediate
        // arguments matching will remove all conditions in this code.
        __m256  acc0, acc1, acc2, acc3, acc4,
            acc5, acc6, acc7, acc8, acc9,
            acc10, acc11, acc12, acc13, acc14;

//__pragma(warning(push))
__pragma(warning(disable:4127))

        // Load outputs and perform multiplication.
        if (T_SIZE >=  1)  acc0 = _mm256_mul_ps(_mm256_loadu_ps(output_ptr +  0 * T_batch_width), acc_sum);
        if (T_SIZE >=  2)  acc1 = _mm256_mul_ps(_mm256_loadu_ps(output_ptr +  1 * T_batch_width), acc_sum);
        if (T_SIZE >=  3)  acc2 = _mm256_mul_ps(_mm256_loadu_ps(output_ptr +  2 * T_batch_width), acc_sum);
        if (T_SIZE >=  4)  acc3 = _mm256_mul_ps(_mm256_loadu_ps(output_ptr +  3 * T_batch_width), acc_sum);
        if (T_SIZE >=  5)  acc4 = _mm256_mul_ps(_mm256_loadu_ps(output_ptr +  4 * T_batch_width), acc_sum);
        if (T_SIZE >=  6)  acc5 = _mm256_mul_ps(_mm256_loadu_ps(output_ptr +  5 * T_batch_width), acc_sum);
        if (T_SIZE >=  7)  acc6 = _mm256_mul_ps(_mm256_loadu_ps(output_ptr +  6 * T_batch_width), acc_sum);
        if (T_SIZE >=  8)  acc7 = _mm256_mul_ps(_mm256_loadu_ps(output_ptr +  7 * T_batch_width), acc_sum);
        if (T_SIZE >=  9)  acc8 = _mm256_mul_ps(_mm256_loadu_ps(output_ptr +  8 * T_batch_width), acc_sum);
        if (T_SIZE >= 10)  acc9 = _mm256_mul_ps(_mm256_loadu_ps(output_ptr +  9 * T_batch_width), acc_sum);
        if (T_SIZE >= 11) acc10 = _mm256_mul_ps(_mm256_loadu_ps(output_ptr + 10 * T_batch_width), acc_sum);
        if (T_SIZE >= 12) acc11 = _mm256_mul_ps(_mm256_loadu_ps(output_ptr + 11 * T_batch_width), acc_sum);
        if (T_SIZE >= 13) acc12 = _mm256_mul_ps(_mm256_loadu_ps(output_ptr + 12 * T_batch_width), acc_sum);
        if (T_SIZE >= 14) acc13 = _mm256_mul_ps(_mm256_loadu_ps(output_ptr + 13 * T_batch_width), acc_sum);
        if (T_SIZE >= 15) acc14 = _mm256_mul_ps(_mm256_loadu_ps(output_ptr + 14 * T_batch_width), acc_sum);

        // Store results.
        if (T_SIZE >=  1) _mm256_storeu_ps(output_ptr +  0 * T_batch_width,  acc0);
        if (T_SIZE >=  2) _mm256_storeu_ps(output_ptr +  1 * T_batch_width,  acc1);
        if (T_SIZE >=  3) _mm256_storeu_ps(output_ptr +  2 * T_batch_width,  acc2);
        if (T_SIZE >=  4) _mm256_storeu_ps(output_ptr +  3 * T_batch_width,  acc3);
        if (T_SIZE >=  5) _mm256_storeu_ps(output_ptr +  4 * T_batch_width,  acc4);
        if (T_SIZE >=  6) _mm256_storeu_ps(output_ptr +  5 * T_batch_width,  acc5);
        if (T_SIZE >=  7) _mm256_storeu_ps(output_ptr +  6 * T_batch_width,  acc6);
        if (T_SIZE >=  8) _mm256_storeu_ps(output_ptr +  7 * T_batch_width,  acc7);
        if (T_SIZE >=  9) _mm256_storeu_ps(output_ptr +  8 * T_batch_width,  acc8);
        if (T_SIZE >= 10) _mm256_storeu_ps(output_ptr +  9 * T_batch_width,  acc9);
        if (T_SIZE >= 11) _mm256_storeu_ps(output_ptr + 10 * T_batch_width, acc10);
        if (T_SIZE >= 12) _mm256_storeu_ps(output_ptr + 11 * T_batch_width, acc11);
        if (T_SIZE >= 13) _mm256_storeu_ps(output_ptr + 12 * T_batch_width, acc12);
        if (T_SIZE >= 14) _mm256_storeu_ps(output_ptr + 13 * T_batch_width, acc13);
        if (T_SIZE >= 15) _mm256_storeu_ps(output_ptr + 14 * T_batch_width, acc14);

//__pragma(warning(pop))

        output_ptr += T_batch_width*T_SIZE;
    }

    template<uint32_t T_SIZE, uint32_t T_batch_width>
    static void softmax_compute_block(float* &output_ptr, __m256 &acc_sum)
    {
        // We are not using table of registers and unroll pragmas
        // due to compiler which have issues with register allocation
        // and needs special, obvious treatment. Template immediate
        // arguments matching will remove all conditions in this code.
        __m256  acc0, acc1, acc2, acc3, acc4,
                acc5, acc6, acc7, acc8, acc9,
                acc10, acc11, acc12, acc13, acc14;

        // Load inputs and perform e^x
        if (T_SIZE >=  1)  acc0 = _inner_mm256_exp_ps(_mm256_loadu_ps(output_ptr +  0 * T_batch_width));
        if (T_SIZE >=  2)  acc1 = _inner_mm256_exp_ps(_mm256_loadu_ps(output_ptr +  1 * T_batch_width));
        if (T_SIZE >=  3)  acc2 = _inner_mm256_exp_ps(_mm256_loadu_ps(output_ptr +  2 * T_batch_width));
        if (T_SIZE >=  4)  acc3 = _inner_mm256_exp_ps(_mm256_loadu_ps(output_ptr +  3 * T_batch_width));
        if (T_SIZE >=  5)  acc4 = _inner_mm256_exp_ps(_mm256_loadu_ps(output_ptr +  4 * T_batch_width));
        if (T_SIZE >=  6)  acc5 = _inner_mm256_exp_ps(_mm256_loadu_ps(output_ptr +  5 * T_batch_width));
        if (T_SIZE >=  7)  acc6 = _inner_mm256_exp_ps(_mm256_loadu_ps(output_ptr +  6 * T_batch_width));
        if (T_SIZE >=  8)  acc7 = _inner_mm256_exp_ps(_mm256_loadu_ps(output_ptr +  7 * T_batch_width));
        if (T_SIZE >=  9)  acc8 = _inner_mm256_exp_ps(_mm256_loadu_ps(output_ptr +  8 * T_batch_width));
        if (T_SIZE >= 10)  acc9 = _inner_mm256_exp_ps(_mm256_loadu_ps(output_ptr +  9 * T_batch_width));
        if (T_SIZE >= 11) acc10 = _inner_mm256_exp_ps(_mm256_loadu_ps(output_ptr + 10 * T_batch_width));
        if (T_SIZE >= 12) acc11 = _inner_mm256_exp_ps(_mm256_loadu_ps(output_ptr + 11 * T_batch_width));
        if (T_SIZE >= 13) acc12 = _inner_mm256_exp_ps(_mm256_loadu_ps(output_ptr + 12 * T_batch_width));
        if (T_SIZE >= 14) acc13 = _inner_mm256_exp_ps(_mm256_loadu_ps(output_ptr + 13 * T_batch_width));
        if (T_SIZE >= 15) acc14 = _inner_mm256_exp_ps(_mm256_loadu_ps(output_ptr + 14 * T_batch_width));

        // Store results.
        if (T_SIZE >=  1) _mm256_storeu_ps(output_ptr +  0 * T_batch_width,  acc0);
        if (T_SIZE >=  2) _mm256_storeu_ps(output_ptr +  1 * T_batch_width,  acc1);
        if (T_SIZE >=  3) _mm256_storeu_ps(output_ptr +  2 * T_batch_width,  acc2);
        if (T_SIZE >=  4) _mm256_storeu_ps(output_ptr +  3 * T_batch_width,  acc3);
        if (T_SIZE >=  5) _mm256_storeu_ps(output_ptr +  4 * T_batch_width,  acc4);
        if (T_SIZE >=  6) _mm256_storeu_ps(output_ptr +  5 * T_batch_width,  acc5);
        if (T_SIZE >=  7) _mm256_storeu_ps(output_ptr +  6 * T_batch_width,  acc6);
        if (T_SIZE >=  8) _mm256_storeu_ps(output_ptr +  7 * T_batch_width,  acc7);
        if (T_SIZE >=  9) _mm256_storeu_ps(output_ptr +  8 * T_batch_width,  acc8);
        if (T_SIZE >= 10) _mm256_storeu_ps(output_ptr +  9 * T_batch_width,  acc9);
        if (T_SIZE >= 11) _mm256_storeu_ps(output_ptr + 10 * T_batch_width, acc10);
        if (T_SIZE >= 12) _mm256_storeu_ps(output_ptr + 11 * T_batch_width, acc11);
        if (T_SIZE >= 13) _mm256_storeu_ps(output_ptr + 12 * T_batch_width, acc12);
        if (T_SIZE >= 14) _mm256_storeu_ps(output_ptr + 13 * T_batch_width, acc13);
        if (T_SIZE >= 15) _mm256_storeu_ps(output_ptr + 14 * T_batch_width, acc14);

        // Sum up accumulators.
        if (T_SIZE >=  1) acc_sum = _mm256_add_ps(acc0,  acc_sum);
        if (T_SIZE >=  2) acc_sum = _mm256_add_ps(acc1,  acc_sum);
        if (T_SIZE >=  3) acc_sum = _mm256_add_ps(acc2,  acc_sum);
        if (T_SIZE >=  4) acc_sum = _mm256_add_ps(acc3,  acc_sum);
        if (T_SIZE >=  5) acc_sum = _mm256_add_ps(acc4,  acc_sum);
        if (T_SIZE >=  6) acc_sum = _mm256_add_ps(acc5,  acc_sum);
        if (T_SIZE >=  7) acc_sum = _mm256_add_ps(acc6,  acc_sum);
        if (T_SIZE >=  8) acc_sum = _mm256_add_ps(acc7,  acc_sum);
        if (T_SIZE >=  9) acc_sum = _mm256_add_ps(acc8,  acc_sum);
        if (T_SIZE >= 10) acc_sum = _mm256_add_ps(acc9,  acc_sum);
        if (T_SIZE >= 11) acc_sum = _mm256_add_ps(acc10, acc_sum);
        if (T_SIZE >= 12) acc_sum = _mm256_add_ps(acc11, acc_sum);
        if (T_SIZE >= 13) acc_sum = _mm256_add_ps(acc12, acc_sum);
        if (T_SIZE >= 14) acc_sum = _mm256_add_ps(acc13, acc_sum);
        if (T_SIZE >= 15) acc_sum = _mm256_add_ps(acc14, acc_sum);

        output_ptr += T_batch_width*T_SIZE;
    }

    template<uint32_t T_NUM_ITERATIONS>
    static void softmax_compute_subsimd(
        float* &output_ptr,
        float &acc_sum)
    {
        for (auto iteration = 0u; iteration < T_NUM_ITERATIONS; ++iteration)
        {
            float acc0 = std::exp(*output_ptr);
            *output_ptr = acc0;
            acc_sum += acc0;

            ++output_ptr;
        }
    }

    template<uint32_t T_NUM_ITERATIONS>
    static void softmax_finalize_subsimd(
        float* &output_ptr,
        float &acc_sum)
    {
        for (auto iteration = 0u; iteration < T_NUM_ITERATIONS; ++iteration)
        {
            float acc0 = *output_ptr;
            acc0 *= acc_sum;
            *output_ptr = acc0;

            ++output_ptr;
        }
    }

    static void run_softmax_work_item_latency(const task_data_t *data) {
        const softmax *softmax_layer = data->softmax_layer;
        const auto input_arg = softmax_layer->argument.input[0].primitive.as<const memory&>().argument;

        const auto input_width  = input_arg.size.spatial[0];
        const auto output_width  = softmax_layer->argument.output_size.spatial[0];

        const auto num_full_blocks = output_width / C_data_stride_batch1;
        const auto partial_block_size = (output_width / C_simd_width) % C_max_acc_batch1;
        const auto subsimd_block_size = output_width % C_simd_width;

        // Find max value for each image. We will use 1 scalar accumulator (semi-naive way).
        {
            auto input_buffer = static_cast<float*>(softmax_layer->input_memory(0).pointer);
            auto output_buffer = static_cast<float*>(softmax_layer->output_memory(0).pointer);

            float max_value;
            uint32_t input = 0;
            max_value = *(input_buffer + (input++));

            while(input < input_width)
                max_value = std::max(max_value, *(input_buffer + (input++)));

            for(uint32_t output = 0; output < output_width; ++output)
            {
                *(output_buffer + output) = *(input_buffer + output) - max_value;
            }
        }

        __m256 acc_sum = _mm256_setzero_ps();
        float subsimd_sum = 0.0f;

        {
            auto output_buffer = static_cast<float*>(softmax_layer->output_memory(0).pointer);

            for (auto block = 0u; block < num_full_blocks; ++block)
            {
                // Run computation.
                softmax_compute_block<C_max_acc_batch1, C_simd_width>(output_buffer, acc_sum);
            }

            switch (partial_block_size)
            {
                case  0: break;
                case  1: softmax_compute_block< 1, C_simd_width>(output_buffer, acc_sum); break;
                case  2: softmax_compute_block< 2, C_simd_width>(output_buffer, acc_sum); break;
                case  3: softmax_compute_block< 3, C_simd_width>(output_buffer, acc_sum); break;
                case  4: softmax_compute_block< 4, C_simd_width>(output_buffer, acc_sum); break;
                case  5: softmax_compute_block< 5, C_simd_width>(output_buffer, acc_sum); break;
                case  6: softmax_compute_block< 6, C_simd_width>(output_buffer, acc_sum); break;
                case  7: softmax_compute_block< 7, C_simd_width>(output_buffer, acc_sum); break;
                case  8: softmax_compute_block< 8, C_simd_width>(output_buffer, acc_sum); break;
                case  9: softmax_compute_block< 9, C_simd_width>(output_buffer, acc_sum); break;
                case 10: softmax_compute_block<10, C_simd_width>(output_buffer, acc_sum); break;
                case 11: softmax_compute_block<11, C_simd_width>(output_buffer, acc_sum); break;
                case 12: softmax_compute_block<12, C_simd_width>(output_buffer, acc_sum); break;
                case 13: softmax_compute_block<13, C_simd_width>(output_buffer, acc_sum); break;
                case 14: softmax_compute_block<14, C_simd_width>(output_buffer, acc_sum); break;
                default: NN_UNREACHABLE_CODE;
            }

            switch (subsimd_block_size)
            {
                case 0: break;
                case 1: softmax_compute_subsimd<1>(output_buffer, subsimd_sum); break;
                case 2: softmax_compute_subsimd<2>(output_buffer, subsimd_sum); break;
                case 3: softmax_compute_subsimd<3>(output_buffer, subsimd_sum); break;
                case 4: softmax_compute_subsimd<4>(output_buffer, subsimd_sum); break;
                case 5: softmax_compute_subsimd<5>(output_buffer, subsimd_sum); break;
                case 6: softmax_compute_subsimd<6>(output_buffer, subsimd_sum); break;
                case 7: softmax_compute_subsimd<7>(output_buffer, subsimd_sum); break;
                default: NN_UNREACHABLE_CODE;
            }
        }

        {
            __m256 intermediate_sum = _mm256_hadd_ps(acc_sum, acc_sum);
            intermediate_sum = _mm256_permutevar8x32_ps(intermediate_sum, _mm256_set_epi32(0, 1, 4, 5, 2, 3, 6, 7));
            intermediate_sum = _mm256_hadd_ps(intermediate_sum, intermediate_sum);
            intermediate_sum = _mm256_hadd_ps(intermediate_sum, intermediate_sum);

            acc_sum = _mm256_add_ps(intermediate_sum, _mm256_set1_ps(subsimd_sum));
            subsimd_sum = _mm_cvtss_f32(_mm256_extractf128_ps(acc_sum, 0));

            acc_sum = _mm256_div_ps(_mm256_set1_ps(1.0f), acc_sum);
            subsimd_sum = 1.0f / subsimd_sum;
        }

        {
            auto output_buffer = static_cast<float*>(softmax_layer->output_memory(0).pointer);

            for (auto block = 0u; block < num_full_blocks; ++block)
            {
                // Run computation.
                softmax_finalize_block<C_max_acc_batch1, C_simd_width>(output_buffer, acc_sum);
            }

            switch (partial_block_size)
            {
                case  0: break;
                case  1: softmax_finalize_block< 1, C_simd_width>(output_buffer, acc_sum); break;
                case  2: softmax_finalize_block< 2, C_simd_width>(output_buffer, acc_sum); break;
                case  3: softmax_finalize_block< 3, C_simd_width>(output_buffer, acc_sum); break;
                case  4: softmax_finalize_block< 4, C_simd_width>(output_buffer, acc_sum); break;
                case  5: softmax_finalize_block< 5, C_simd_width>(output_buffer, acc_sum); break;
                case  6: softmax_finalize_block< 6, C_simd_width>(output_buffer, acc_sum); break;
                case  7: softmax_finalize_block< 7, C_simd_width>(output_buffer, acc_sum); break;
                case  8: softmax_finalize_block< 8, C_simd_width>(output_buffer, acc_sum); break;
                case  9: softmax_finalize_block< 9, C_simd_width>(output_buffer, acc_sum); break;
                case 10: softmax_finalize_block<10, C_simd_width>(output_buffer, acc_sum); break;
                case 11: softmax_finalize_block<11, C_simd_width>(output_buffer, acc_sum); break;
                case 12: softmax_finalize_block<12, C_simd_width>(output_buffer, acc_sum); break;
                case 13: softmax_finalize_block<13, C_simd_width>(output_buffer, acc_sum); break;
                case 14: softmax_finalize_block<14, C_simd_width>(output_buffer, acc_sum); break;
                default: NN_UNREACHABLE_CODE;
            }

            switch (subsimd_block_size)
            {
                case 0: break;
                case 1: softmax_finalize_subsimd<1>(output_buffer, subsimd_sum); break;
                case 2: softmax_finalize_subsimd<2>(output_buffer, subsimd_sum); break;
                case 3: softmax_finalize_subsimd<3>(output_buffer, subsimd_sum); break;
                case 4: softmax_finalize_subsimd<4>(output_buffer, subsimd_sum); break;
                case 5: softmax_finalize_subsimd<5>(output_buffer, subsimd_sum); break;
                case 6: softmax_finalize_subsimd<6>(output_buffer, subsimd_sum); break;
                case 7: softmax_finalize_subsimd<7>(output_buffer, subsimd_sum); break;
                default: NN_UNREACHABLE_CODE;
            }
        }
    }

    static void run_softmax_work_item_batch8(const task_data_t *data) {
        const softmax *softmax_layer = data->softmax_layer;
        const auto input_arg = softmax_layer->argument.input[0].primitive.as<const memory&>().argument;

        const auto input_width  = input_arg.size.spatial[0];
        const auto output_width = softmax_layer->argument.output_size.spatial[0];

        const auto num_full_blocks = output_width / C_max_acc_batch8;
        const auto partial_block_size = output_width % C_max_acc_batch8;

        // Find max value for each image. We will use 1 accumulator (1*8=8) so each image has its own avx component.
        {
            auto input_buffer = static_cast<float*>(softmax_layer->input_memory(0).pointer);
            auto output_buffer = static_cast<float*>(softmax_layer->output_memory(0).pointer);

            __m256 max_values;
            uint32_t input = 0;
            max_values = _mm256_load_ps(input_buffer + (input++) * C_batch8_size);

            while(input < input_width)
                max_values = _mm256_max_ps(max_values, _mm256_load_ps(input_buffer + (input++) * C_batch8_size));

            for(uint32_t output = 0; output < output_width; ++output)
            {
                    _mm256_storeu_ps(
                        output_buffer + output * C_batch8_size,
                        _mm256_sub_ps(
                            _mm256_load_ps(input_buffer + output * C_batch8_size),
                            max_values));
            }
        }

        __m256 acc_sum = _mm256_setzero_ps();

        {
            auto output_buffer = static_cast<float*>(softmax_layer->output_memory(0).pointer);

            for (auto block = 0u; block < num_full_blocks; ++block)
            {
                // Run computation.
                softmax_compute_block<C_max_acc_batch8, C_batch8_size>(output_buffer, acc_sum);
            }

            switch (partial_block_size)
            {
                case  0: break;
                case  1: softmax_compute_block< 1, C_batch8_size>(output_buffer, acc_sum); break;
                case  2: softmax_compute_block< 2, C_batch8_size>(output_buffer, acc_sum); break;
                case  3: softmax_compute_block< 3, C_batch8_size>(output_buffer, acc_sum); break;
                case  4: softmax_compute_block< 4, C_batch8_size>(output_buffer, acc_sum); break;
                case  5: softmax_compute_block< 5, C_batch8_size>(output_buffer, acc_sum); break;
                case  6: softmax_compute_block< 6, C_batch8_size>(output_buffer, acc_sum); break;
                case  7: softmax_compute_block< 7, C_batch8_size>(output_buffer, acc_sum); break;
                case  8: softmax_compute_block< 8, C_batch8_size>(output_buffer, acc_sum); break;
                case  9: softmax_compute_block< 9, C_batch8_size>(output_buffer, acc_sum); break;
                case 10: softmax_compute_block<10, C_batch8_size>(output_buffer, acc_sum); break;
                case 11: softmax_compute_block<11, C_batch8_size>(output_buffer, acc_sum); break;
                case 12: softmax_compute_block<12, C_batch8_size>(output_buffer, acc_sum); break;
                case 13: softmax_compute_block<13, C_batch8_size>(output_buffer, acc_sum); break;
                case 14: softmax_compute_block<14, C_batch8_size>(output_buffer, acc_sum); break;
                default: NN_UNREACHABLE_CODE;
            }
        }

        acc_sum = _mm256_div_ps(_mm256_set1_ps(1.0f), acc_sum);

        {
            auto output_buffer = static_cast<float*>(softmax_layer->output_memory(0).pointer);

            for (auto block = 0u; block < num_full_blocks; ++block)
            {
                // Run computation.
                softmax_finalize_block<C_max_acc_batch8, C_batch8_size>(output_buffer, acc_sum);
            }

            switch (partial_block_size)
            {
                case  0: break;
                case  1: softmax_finalize_block< 1, C_batch8_size>(output_buffer, acc_sum); break;
                case  2: softmax_finalize_block< 2, C_batch8_size>(output_buffer, acc_sum); break;
                case  3: softmax_finalize_block< 3, C_batch8_size>(output_buffer, acc_sum); break;
                case  4: softmax_finalize_block< 4, C_batch8_size>(output_buffer, acc_sum); break;
                case  5: softmax_finalize_block< 5, C_batch8_size>(output_buffer, acc_sum); break;
                case  6: softmax_finalize_block< 6, C_batch8_size>(output_buffer, acc_sum); break;
                case  7: softmax_finalize_block< 7, C_batch8_size>(output_buffer, acc_sum); break;
                case  8: softmax_finalize_block< 8, C_batch8_size>(output_buffer, acc_sum); break;
                case  9: softmax_finalize_block< 9, C_batch8_size>(output_buffer, acc_sum); break;
                case 10: softmax_finalize_block<10, C_batch8_size>(output_buffer, acc_sum); break;
                case 11: softmax_finalize_block<11, C_batch8_size>(output_buffer, acc_sum); break;
                case 12: softmax_finalize_block<12, C_batch8_size>(output_buffer, acc_sum); break;
                case 13: softmax_finalize_block<13, C_batch8_size>(output_buffer, acc_sum); break;
                case 14: softmax_finalize_block<14, C_batch8_size>(output_buffer, acc_sum); break;
                default: NN_UNREACHABLE_CODE;
            }
        }
    }

    static void run_softmax_work_item_batch48(const task_data_t *data) {
        const softmax *softmax_layer = data->softmax_layer;
        const auto input_arg = softmax_layer->argument.input[0].primitive.as<const memory&>().argument;

        const auto input_width  = input_arg.size.spatial[0];
        const auto output_width = softmax_layer->argument.output_size.spatial[0];

        const auto num_full_blocks = output_width / C_max_acc_batch48;
        const auto partial_block_size = output_width % C_max_acc_batch48;
        const auto num_batch_packages = 6;

        // Find max value for each image. We will use 6 accumulators (6*8=48) so each image has its own avx component.
        {
            auto input_buffer = static_cast<float*>(softmax_layer->input_memory(0).pointer);
            auto output_buffer = static_cast<float*>(softmax_layer->output_memory(0).pointer);

            __m256 max_values[num_batch_packages];
            uint32_t input = 0;

            for(auto acc = 0u; acc < num_batch_packages; ++acc)
                max_values[acc] = _mm256_load_ps(input_buffer + input * C_batch48_size + acc * C_simd_width);

            ++input;

            for(;input < input_width; ++input)
                for(auto acc = 0u; acc < num_batch_packages; ++acc)
                    max_values[acc] = _mm256_max_ps(max_values[acc], _mm256_load_ps(input_buffer + input * C_batch48_size + acc * C_simd_width));

            for(uint32_t output = 0; output < output_width; ++output)
            {
                for(auto acc = 0u; acc < num_batch_packages; ++acc)
                    _mm256_storeu_ps(
                        output_buffer + output * C_batch48_size + acc * C_simd_width,
                        _mm256_sub_ps(
                            _mm256_load_ps(input_buffer + output * C_batch48_size + acc * C_simd_width),
                            max_values[acc]));
            }
        }

        for (uint32_t batch_package = 0; batch_package < num_batch_packages; ++batch_package)
        {
            const auto output_view_start = batch_package * C_batch8_size;

            __m256 acc_sum = _mm256_setzero_ps();

            {
                auto output_buffer = static_cast<float*>(softmax_layer->output_memory(0).pointer) + output_view_start;

                for (auto block = 0u; block < num_full_blocks; ++block)
                {
                    // Run computation.
                    softmax_compute_block<C_max_acc_batch48, C_batch48_size>(output_buffer, acc_sum);
                }

                switch (partial_block_size)
                {
                    case  0: break;
                    case  1: softmax_compute_block< 1, C_batch48_size>(output_buffer, acc_sum); break;
                    case  2: softmax_compute_block< 2, C_batch48_size>(output_buffer, acc_sum); break;
                    case  3: softmax_compute_block< 3, C_batch48_size>(output_buffer, acc_sum); break;
                    case  4: softmax_compute_block< 4, C_batch48_size>(output_buffer, acc_sum); break;
                    case  5: softmax_compute_block< 5, C_batch48_size>(output_buffer, acc_sum); break;
                    case  6: softmax_compute_block< 6, C_batch48_size>(output_buffer, acc_sum); break;
                    case  7: softmax_compute_block< 7, C_batch48_size>(output_buffer, acc_sum); break;
                    case  8: softmax_compute_block< 8, C_batch48_size>(output_buffer, acc_sum); break;
                    case  9: softmax_compute_block< 9, C_batch48_size>(output_buffer, acc_sum); break;
                    case 10: softmax_compute_block<10, C_batch48_size>(output_buffer, acc_sum); break;
                    case 11: softmax_compute_block<11, C_batch48_size>(output_buffer, acc_sum); break;
                    case 12: softmax_compute_block<12, C_batch48_size>(output_buffer, acc_sum); break;
                    case 13: softmax_compute_block<13, C_batch48_size>(output_buffer, acc_sum); break;
                    case 14: softmax_compute_block<14, C_batch48_size>(output_buffer, acc_sum); break;
                    default: NN_UNREACHABLE_CODE;
                }
            }

            acc_sum = _mm256_div_ps(_mm256_set1_ps(1.0f), acc_sum);

            {
                auto output_buffer = static_cast<float*>(softmax_layer->output_memory(0).pointer) + output_view_start;

                for (auto block = 0u; block < num_full_blocks; ++block)
                {
                    // Run computation.
                    softmax_finalize_block<C_max_acc_batch48, C_batch48_size>(output_buffer, acc_sum);
                }

                switch (partial_block_size)
                {
                    case  0: break;
                    case  1: softmax_finalize_block< 1, C_batch48_size>(output_buffer, acc_sum); break;
                    case  2: softmax_finalize_block< 2, C_batch48_size>(output_buffer, acc_sum); break;
                    case  3: softmax_finalize_block< 3, C_batch48_size>(output_buffer, acc_sum); break;
                    case  4: softmax_finalize_block< 4, C_batch48_size>(output_buffer, acc_sum); break;
                    case  5: softmax_finalize_block< 5, C_batch48_size>(output_buffer, acc_sum); break;
                    case  6: softmax_finalize_block< 6, C_batch48_size>(output_buffer, acc_sum); break;
                    case  7: softmax_finalize_block< 7, C_batch48_size>(output_buffer, acc_sum); break;
                    case  8: softmax_finalize_block< 8, C_batch48_size>(output_buffer, acc_sum); break;
                    case  9: softmax_finalize_block< 9, C_batch48_size>(output_buffer, acc_sum); break;
                    case 10: softmax_finalize_block<10, C_batch48_size>(output_buffer, acc_sum); break;
                    case 11: softmax_finalize_block<11, C_batch48_size>(output_buffer, acc_sum); break;
                    case 12: softmax_finalize_block<12, C_batch48_size>(output_buffer, acc_sum); break;
                    case 13: softmax_finalize_block<13, C_batch48_size>(output_buffer, acc_sum); break;
                    case 14: softmax_finalize_block<14, C_batch48_size>(output_buffer, acc_sum); break;
                    default: NN_UNREACHABLE_CODE;
                }
            }
        }
    }

    task_group work() {
        return this->tasks;
    }
};



softmax_cpu_avx2::softmax_cpu_avx2(softmax &arg)
    : is_an_implementation(neural::type_id<softmax_cpu_avx2>())
    , outer(arg)
{
    auto this_softmax = static_cast<const softmax *>(&outer);

    auto& input_offset = this_softmax->argument.input_offset;
    auto& output_offset = this_softmax->argument.output_offset;

    for (auto &x : input_offset.raw)  if (x != 0) throw std::runtime_error("softmax input offset must be equal to zero.");
    for (auto &x : output_offset.raw) if (x > 0)  throw std::runtime_error("softmax output offset must be equal to zero.");

    auto batch_size = this_softmax->input_memory(0).argument.size.batch[0];
    if(batch_size != 1 && batch_size != 8 && batch_size != 48) throw std::runtime_error("softmax batch size must be equal to 1 or 8 or 48.");

    assert(1 == this_softmax->argument.input.size());
    assert(1 == this_softmax->argument.output.size());
    assert(1 == this_softmax->argument.output_size.batch.size());

    softmax_ptr.reset(new softmax_avx2_worker(this_softmax));
};

softmax_cpu_avx2::~softmax_cpu_avx2() {}

namespace {
struct attach {
    attach() {
        auto key_fw = std::make_tuple(engine::cpu, memory::format::xb_f32, memory::format::xb_f32);
        auto val_fw = softmax_cpu_avx2::create;

        softmax_fw_implementation_map::instance().insert( {key_fw, val_fw} );
    }
    ~attach() {}
};

#ifdef __GNUC__
    __attribute__((visibility("default")))
#elif _MSC_VER
#   pragma section(".nn_init$m", read, write)
#endif
attach attach_impl;
}

} // namespace normalization
} // namespace neural
