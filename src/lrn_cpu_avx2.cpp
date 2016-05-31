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

__m256 _inner_mm256_invpow075_ps(__m256 arg)
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


enum EXP_APPROX
{
    APPROX_NEGATIVE_0_75,
};

template<EXP_APPROX T_approx> __m256 _internal_mm256_pow_ps(__m256 base, __m256 exponent);
template<>                    __m256 _internal_mm256_pow_ps<APPROX_NEGATIVE_0_75>(__m256 base, __m256 exponent) { return _inner_mm256_invpow075_ps(base); }

template<EXP_APPROX T_approx, bool T_emit_intermediates>
void run_3d_normalization_work_item_template(
    const nn::workload_data<> *input_view,
    nn::workload_data<> *intermediate_data,
    nn::workload_data<> *output_view,
    uint32_t n,
    float alpha,
    uint32_t k,
    float beta)
{
    const auto input_column_size = input_view->parent->lengths.t[NN_DATA_COORD_z];
    const auto input_row_size = input_view->parent->lengths.t[NN_DATA_COORD_x] * input_column_size;
    const auto input_batch_size = input_view->parent->lengths.t[NN_DATA_COORD_y] * input_row_size;

    const auto output_column_size = output_view->parent->lengths.t[NN_DATA_COORD_z];
    const auto output_row_size = output_view->parent->lengths.t[NN_DATA_COORD_x] * output_column_size;
    const auto output_batch_size = output_view->parent->lengths.t[NN_DATA_COORD_y] * output_row_size;

    auto input_buffer = static_cast<float*>(input_view->parent->data_buffer);
    auto output_buffer = static_cast<float*>(output_view->parent->data_buffer);
    auto intermediate_buffer = (T_emit_intermediates) ? static_cast<float*>(intermediate_data->parent->data_buffer) : nullptr;

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

    const __m256 pow_vec = _mm256_set1_ps(-beta);

    for (uint32_t batch = output_view->view_begin.t[NN_DATA_COORD_n]; batch <= output_view->view_end.t[NN_DATA_COORD_n]; ++batch)
    {
        for (uint32_t row = input_view->view_begin.t[NN_DATA_COORD_y], out_row = output_view->view_begin.t[NN_DATA_COORD_y];
        out_row <= output_view->view_end.t[NN_DATA_COORD_y];
            ++row, ++out_row)
        {
            for (uint32_t column = input_view->view_begin.t[NN_DATA_COORD_x], out_column = output_view->view_begin.t[NN_DATA_COORD_x];
            out_column <= output_view->view_end.t[NN_DATA_COORD_x];
                ++column, ++out_column)
            {
                const auto input_address = &input_buffer[batch*input_batch_size + row*input_row_size + column*input_column_size];
                const auto output_address = &output_buffer[batch*output_batch_size + out_row*output_row_size + out_column*output_column_size];
                const auto intermediate_address =
                    (T_emit_intermediates)
                    ? &intermediate_buffer[batch*output_batch_size + out_row*output_row_size + out_column*output_column_size]
                    : nullptr;

                // Prepare first data chunk.
                __m256 source_tmp = _mm256_maskload_ps(input_address - neighbourhood, first_masker);
                source_tmp = _mm256_mul_ps(source_tmp, source_tmp);

#pragma forceinline recursive
                for (uint32_t feature_map = input_view->view_begin.t[NN_DATA_COORD_z], out_feature_map = output_view->view_begin.t[NN_DATA_COORD_z];
                out_feature_map <= output_view->view_end.t[NN_DATA_COORD_z];
                    feature_map += C_simd_width, out_feature_map += C_simd_width)
                {
                    // Initialize accumulator.
                    __m256 acc = _mm256_setzero_ps();

                    // Move previous saved chunk to first and load new one as a next.
                    __m256 source_first = source_tmp;
                    __m256 source_second =
                        (feature_map + C_simd_width <= input_view->view_end.t[NN_DATA_COORD_z])
                        ? _mm256_loadu_ps(input_address + feature_map - neighbourhood + C_simd_width)
                        : _mm256_maskload_ps(input_address + feature_map - neighbourhood + C_simd_width, last_masker);

                    // Square of new chunk and save for next iteration.
                    source_tmp = source_second = _mm256_mul_ps(source_second, source_second);

                    // Required for final computation.
                    __m256 source_raw = _mm256_loadu_ps(input_address + feature_map);

                    // Forward permute - five times.
                    for (int i = 0; i < n; ++i)
                    {
                        acc = _mm256_add_ps(source_first, acc);
                        source_first = _mm256_permutevar8x32_ps(source_first, forward_permuter);
                        source_second = _mm256_permutevar8x32_ps(source_second, forward_permuter);
                        source_first = _mm256_blend_ps(source_first, source_second, 0x80);
                    }

                    // Do k + alpha * acc.
                    acc = _mm256_fmadd_ps(acc, _mm256_set1_ps(alpha), _mm256_set1_ps(k));

                    if (T_emit_intermediates)
                    {
                        // Store this for later backprop use.
                        _mm256_stream_ps(intermediate_address + out_feature_map, acc);
                    }

                    // acc ^ -beta
                    acc = _internal_mm256_pow_ps<T_approx>(acc, pow_vec);

                    // Multiply with input data.
                    acc = _mm256_mul_ps(acc, source_raw);

                    // Save data.
                    _mm256_storeu_ps(output_address + out_feature_map, acc);
                }
            }
        }
    }
}


namespace neural {

    lrn_cpu_avx2::lrn_cpu_avx2(normalization::response &arg)
        : is_an_implementation(neural::type_id<lrn_cpu_avx2>())
        , outer(arg) {};

    lrn_cpu_avx2::~lrn_cpu_avx2() {};

    void lrn_cpu_avx2::implementation(const void *ptr) {

        auto this_lrn = static_cast<const normalization::response *>(ptr);

        auto& input_offset = this_lrn->argument.input_offset;
        auto& output_offset = this_lrn->argument.output_offset;
        auto& output_size = this_lrn->argument.output_size;
        auto& padding = this_lrn->argument.padding;
        auto& size = this_lrn->argument.size;
        auto& k = this_lrn->argument.k;
        auto& alpha = this_lrn->argument.alpha;
        auto& beta = this_lrn->argument.beta;

        auto input_arg  = this_lrn->input_memory(0).argument;
        auto output_arg = this_lrn->output_memory(0).argument;

        if (input_arg.size.raw.size() != output_arg.size.raw.size())
            throw std::runtime_error("lrn input/output number of dimension does not match [iput size=" + std::to_string(input_arg.size.raw.size())
                                     + ", output size=" + std::to_string(output_arg.size.raw.size()));

        auto input  = static_cast<float*>(this_lrn->input_memory(0).pointer);
        auto output = static_cast<float*>(this_lrn->output_memory(0).pointer);

        namespace nd = ndimensional;
        nd::value<uint32_t> range(output_size);

        auto calc_in_idx  = nd::choose_calculate_idx(input_arg.format);
        auto calc_out_idx = nd::choose_calculate_idx(output_arg.format);

        nd::value<uint32_t> window_range({ 1,{1,1},size });

        vector<int32_t> help_input_offset({ input_offset });

        switch (padding) {
        case padding::zero:
            // wytworzyæ listê tasków i wrzuciæ do listy tasków
            // run_3d_normalization_work_item_template
            break;
        default:
            throw std::runtime_error("Unknown padding mode in lrn");
        }
    }

    namespace {
        struct attach {
            attach() {
                auto key = std::make_tuple(engine::cpu, memory::format::yxfb_f32, memory::format::yxfb_f32);
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

