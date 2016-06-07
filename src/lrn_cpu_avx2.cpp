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
#include "multidimensional_counter.h"
#include "lrn_cpu_avx2.h"
#include <immintrin.h>

namespace neural {

    // SIMD width for this implementation
    const auto C_simd_width = sizeof(__m256) / sizeof(float);

    // process chunks of data
    struct lrn_cpu_avx2_worker : public neural::is_an_implementation {

#pragma pack(push, 1)
        struct task_data_t {
            const normalization::response *lrn_cpu_layer;
            uint64_t output_batch_begin;
            uint64_t output_batch_end;
            uint64_t input_row_begin;
            uint64_t output_row_begin;
            uint64_t output_row_end;

        };
#pragma pack(pop)

        std::vector<neural::task> tasks;
        std::vector<task_data_t> task_data;

        lrn_cpu_avx2_worker(const void *outher) : is_an_implementation(neural::type_id<lrn_cpu_avx2_worker>()) {


            auto lrn_layer = static_cast<const normalization::response *>(outher);

            auto output_batch_begin=lrn_layer->argument.output_offset.batch[0];
            auto output_batch_end = lrn_layer->argument.output_size.batch[0] + output_batch_begin;
            auto input_row_begin = lrn_layer->argument.input_offset.spatial[1];
            auto output_row_begin = lrn_layer->argument.output_offset.spatial[1];
            auto output_row_end = lrn_layer->argument.output_size.spatial[1] + output_row_begin;


            auto batch_view_length = lrn_layer->argument.output_size.batch[0];
            auto row_view_length = lrn_layer->argument.output_size.spatial[1];

            auto chunks_count = batch_view_length * row_view_length;

            tasks.clear();
            task_data.resize(chunks_count);

            uint64_t index = 0;

            for (auto batch = output_batch_begin; batch < output_batch_end; ++batch)
                for (uint64_t row = input_row_begin, output_row = output_row_begin; output_row < output_row_end; row++, output_row++)
                    task_data[index++] = { lrn_layer, batch, batch+1, row, output_row, output_row+1 };

            for (auto& task : task_data)
                tasks.push_back({ reinterpret_cast<void(*)(const void*)>(run_3d_normalization_work_item), &task });
        }

        static __m256 _inner_mm256_invpow075_ps(__m256 arg)
        {
            __m256i e = _mm256_slli_epi32(
                _mm256_sub_epi32(
                    _mm256_and_si256(
                        _mm256_castps_si256(arg),
                        _mm256_set1_epi32(0x7f800000)),
                    _mm256_set1_epi32(0x3f800000)),
                1);

            __m256 p0 = _mm256_castsi256_ps(
                _mm256_srli_epi32(
                    _mm256_add_epi32(
                        _mm256_mullo_epi32(
                            _mm256_srai_epi32(
                                _mm256_and_si256(
                                    e,
                                    _mm256_set1_epi32(0xfc000000)),
                                2),
                            _mm256_set1_epi32(-3)),
                        _mm256_set1_epi32(0x7f000000)),
                    1));

            __m256 p1 = _mm256_blendv_ps(
                _mm256_set1_ps(0.59460355750136053335874998528f),
                _mm256_set1_ps(1.0f),
                _mm256_castsi256_ps(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            e,
                            _mm256_set1_epi32(1 << 24)),
                        _mm256_set1_epi32(0))));

            __m256 p2 = _mm256_blendv_ps(
                _mm256_set1_ps(0.35355339059327376220042218105f),
                _mm256_set1_ps(1.0f),
                _mm256_castsi256_ps(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            e,
                            _mm256_set1_epi32(2 << 24)),
                        _mm256_set1_epi32(0))));

            arg = _mm256_castsi256_ps(
                _mm256_or_si256(
                    _mm256_and_si256(
                        _mm256_castps_si256(arg),
                        _mm256_set1_epi32(0x007fffff)),
                    _mm256_set1_epi32(0x3f800000)));

            __m256 intermediate_result;
            intermediate_result = _mm256_fmadd_ps(arg, _mm256_set1_ps(-0.06251362156237f), _mm256_set1_ps(0.56657226995864f));
            intermediate_result = _mm256_fmadd_ps(arg, intermediate_result, _mm256_set1_ps(-2.12314847503624f));
            intermediate_result = _mm256_fmadd_ps(arg, intermediate_result, _mm256_set1_ps(4.22879355263332f));
            intermediate_result = _mm256_fmadd_ps(arg, intermediate_result, _mm256_set1_ps(-4.79039952143706f));
            intermediate_result = _mm256_fmadd_ps(arg, intermediate_result, _mm256_set1_ps(3.18069569544757f));

            intermediate_result =
                _mm256_mul_ps(
                    _mm256_mul_ps(
                        p0,
                        p1),
                    _mm256_mul_ps(
                        p2,
                        intermediate_result));

            return intermediate_result;
        }

        static void run_3d_normalization_work_item(const task_data_t *data)
        {
            auto input_buffer = static_cast<float*>(data->lrn_cpu_layer->input_memory(0).pointer);
            auto output_buffer = static_cast<float*>(data->lrn_cpu_layer->output_memory(0).pointer);

            auto &n = data->lrn_cpu_layer->argument.size;
            auto &k = data->lrn_cpu_layer->argument.k;
            auto &alpha = data->lrn_cpu_layer->argument.alpha;
            auto &beta = data->lrn_cpu_layer->argument.beta;

            const auto input_column_size = data->lrn_cpu_layer->input_memory(0).argument.size.feature[0];
            const auto input_row_size = data->lrn_cpu_layer->input_memory(0).argument.size.spatial[0] * input_column_size;
            const auto input_batch_size = data->lrn_cpu_layer->input_memory(0).argument.size.spatial[1] * input_row_size;

            const auto output_column_size = data->lrn_cpu_layer->output_memory(0).argument.size.feature[0];
            const auto output_row_size = data->lrn_cpu_layer->output_memory(0).argument.size.spatial[0] * output_column_size;
            const auto output_batch_size = data->lrn_cpu_layer->output_memory(0).argument.size.spatial[1] * output_row_size;

            const auto output_batch_begin = data->output_batch_begin;
            const auto output_batch_end = data->output_batch_end;

            const auto input_row_begin = data->input_row_begin;
            const auto output_row_begin = data->output_row_begin;
            const auto output_row_end = data->output_row_end;

            const auto input_column_begin = data->lrn_cpu_layer->argument.input_offset.spatial[0];
            const auto output_column_begin = data->lrn_cpu_layer->argument.output_offset.spatial[0];
            const auto output_column_end = data->lrn_cpu_layer->argument.output_size.spatial[0] + output_column_begin;

            const auto output_feature_map_size = data->lrn_cpu_layer->argument.output_size.feature[0];
            const auto input_feature_map_begin = data->lrn_cpu_layer->argument.input_offset.feature[0];
            const auto output_feature_map_begin = data->lrn_cpu_layer->argument.output_offset.feature[0];
            const auto output_feature_map_end = output_feature_map_begin + output_feature_map_size;

            // Const data.
            const uint32_t permutation_mask[8] = { 1, 2, 3, 4, 5, 6, 7, 0 };
            uint32_t first_load_mask[8] = { 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000 };
            uint32_t last_load_mask[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };

            const auto neighbourhood = n / 2;

            for (uint32_t neighbour = 0; neighbour < neighbourhood; ++neighbour)
            {
                first_load_mask[neighbour] ^= 0x80000000;
                last_load_mask[neighbour] ^= 0x80000000;
            }

            // Permuters and masks.
            const __m256i forward_permuter = _mm256_loadu_si256((__m256i*)permutation_mask);
            const __m256i first_masker = _mm256_loadu_si256((__m256i*)first_load_mask);
            const __m256i last_masker = _mm256_loadu_si256((__m256i*)last_load_mask);

            for (uint64_t batch = output_batch_begin; batch < output_batch_end; ++batch)
            {
                for (uint64_t row = input_row_begin, out_row = output_row_begin;
                out_row < output_row_end;
                    ++row, ++out_row)
                {
                    for (uint64_t column = input_column_begin, out_column = output_column_begin;
                    out_column < output_column_end;
                        ++column, ++out_column)
                    {
                        const auto input_address = &input_buffer[batch*input_batch_size + row*input_row_size + column*input_column_size];
                        const auto output_address = &output_buffer[batch*output_batch_size + out_row*output_row_size + out_column*output_column_size];

                        // Prepare first data chunk.
                        __m256 source_tmp = _mm256_maskload_ps(input_address - neighbourhood, first_masker);
                        source_tmp = _mm256_mul_ps(source_tmp, source_tmp);


                        for (uint32_t feature_map = input_feature_map_begin, out_feature_map = output_feature_map_begin;
                        out_feature_map < output_feature_map_end;
                            feature_map += C_simd_width, out_feature_map += C_simd_width)
                        {
                            // Initialize accumulator.
                            __m256 acc = _mm256_setzero_ps();

                            // Move previous saved chunk to first and load new one as a next.
                            __m256 source_first = source_tmp;
                            __m256 source_second =
                                (feature_map + C_simd_width < (input_feature_map_begin+ output_feature_map_size))
                                ? _mm256_loadu_ps(input_address + feature_map - neighbourhood + C_simd_width)
                                : _mm256_maskload_ps(input_address + feature_map - neighbourhood + C_simd_width, last_masker);

                            // Square of new chunk and save for next iteration.
                            source_tmp = source_second = _mm256_mul_ps(source_second, source_second);

                            // Required for final computation.
                            __m256 source_raw = _mm256_loadu_ps(input_address + feature_map);

                            // Forward permute - five times.
                            for (uint32_t i = 0; i < n; ++i)
                            {
                                acc = _mm256_add_ps(source_first, acc);
                                source_first = _mm256_permutevar8x32_ps(source_first, forward_permuter);
                                source_second = _mm256_permutevar8x32_ps(source_second, forward_permuter);
                                source_first = _mm256_blend_ps(source_first, source_second, 0x80);
                            }

                            // Do k + alpha * acc.
                            acc = _mm256_fmadd_ps(acc, _mm256_set1_ps(alpha), _mm256_set1_ps(k));

                            // acc ^ -beta
                            if (beta==0.75f)
                                acc = _inner_mm256_invpow075_ps(acc);
                            else
                                throw std::runtime_error("Value of beta = " + std::to_string(beta) + " not supported. Set beta value to 0.75");

                            // Multiply with input data.
                            acc = _mm256_mul_ps(acc, source_raw);

                            // Save data.
                            _mm256_storeu_ps(output_address + out_feature_map, acc);
                        }
                    }
                }
            }
        }
        neural::task_group work() {
            return{ this->tasks, schedule::unordered };
        }
    };

    lrn_cpu_avx2::lrn_cpu_avx2(normalization::response &arg)
        : is_an_implementation(neural::type_id<lrn_cpu_avx2>())
        , outer(arg) {

        auto this_lrn = static_cast<const normalization::response *>(&outer);

        if (this_lrn->output_memory(0).argument.size.feature[0] % 8)
            throw std::runtime_error("the value of featuremap=" + std::to_string(this_lrn->output_memory(0).argument.size.feature[0]) + " is not a multiple of 8, and it should be");

        lrn_ptr.reset(new lrn_cpu_avx2_worker(this_lrn));
    };

    lrn_cpu_avx2::~lrn_cpu_avx2() {};

    namespace {
        struct attach {
            attach() {
                auto key = std::make_tuple(engine::cpu, memory::format::byxf_f32, memory::format::byxf_f32);
                auto val_fw = lrn_cpu_avx2::create;
                //auto val_bw = lrn_backward_cpu_reference::create;

                lrn_fw_implementation_map::instance().insert({ key, val_fw }); //todo keys should be different
                //lrn_bw_implementation_map.insert({ key, val_bw });
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

