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

namespace {

// parallel implementation of x^(-0.75)
__m256 _invpow075_ps(__m256 arg) {
    // auto e = ((bitwise_cast<int>(arg)&0x7f800000)-0x3f800000)<<1;
    __m256i e = _mm256_slli_epi32(
        _mm256_sub_epi32(
            _mm256_and_si256(
                _mm256_castps_si256(arg),
                _mm256_set1_epi32(0x7f800000)),
            _mm256_set1_epi32(0x3f800000)),
        1);

    // auto p0 = bitwise_cast<float>(static_cast<unsigned int>((static_cast<int>(e&0xfc000000)>>2)*-3+0x7f000000)>>1);
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

    // auto p1 = e&(1<<24) ? 0.59460355750136053335874998528f : 1.0f;
    __m256 p1 = _mm256_blendv_ps(
        _mm256_set1_ps(0.59460355750136053335874998528f),
        _mm256_set1_ps(1.0f),
        _mm256_castsi256_ps(
            _mm256_cmpeq_epi32(
                _mm256_and_si256(
                    e,
                    _mm256_set1_epi32(1 << 24)),
                _mm256_set1_epi32(0))));

    // auto p2 = e&(2<<24) ? 0.35355339059327376220042218105f : 1.0f;
    __m256 p2 = _mm256_blendv_ps(
        _mm256_set1_ps(0.35355339059327376220042218105f),
        _mm256_set1_ps(1.0f),
        _mm256_castsi256_ps(
            _mm256_cmpeq_epi32(
                _mm256_and_si256(
                    e,
                    _mm256_set1_epi32(2 << 24)),
                _mm256_set1_epi32(0))));

    // arg = bitwise_cast<float>((bitwise_cast<int>(arg)&0x007fffff)|0x3f800000);
    arg = _mm256_castsi256_ps(
        _mm256_or_si256(
            _mm256_and_si256(
                _mm256_castps_si256(arg),
                _mm256_set1_epi32(0x007fffff)),
            _mm256_set1_epi32(0x3f800000)));

    // result = arg*(arg*(arg*(arg*(arg*-0.06335208676907f+0.57270562628927f)-2.14081144365050f)+4.25382491869498f)-4.80786079752800f)+3.18549378296333f;
    __m256 result = _mm256_fmadd_ps(arg, _mm256_set1_ps(-0.06251362156237f), _mm256_set1_ps(0.56657226995864f));
    result = _mm256_fmadd_ps(arg, result, _mm256_set1_ps(-2.12314847503624f));
    result = _mm256_fmadd_ps(arg, result, _mm256_set1_ps(4.22879355263332f));
    result = _mm256_fmadd_ps(arg, result, _mm256_set1_ps(-4.79039952143706f));
    result = _mm256_fmadd_ps(arg, result, _mm256_set1_ps(3.18069569544757f));

    // return p0*p1*p2*result;
    return _mm256_mul_ps(_mm256_mul_ps(p0, p1), _mm256_mul_ps(p2,result));
}

};











// local response normalization for byxf_b24_f32 format when beta=0.75, n=5
struct lrn_cpu_avx2_batch24 : public neural::is_an_implementation {

#pragma pack(push, 1)
    struct task_data_t {
        float *input_ptr;
        float *output_ptr;
        uint32_t feature_count;
        float alpha;
        float k;
    };
#pragma pack(pop)

    const neural::normalization::response &outer;
    std::vector<neural::task> tasks;
    std::vector<task_data_t> task_data;

    lrn_cpu_avx2_batch24(neural::normalization::response &outer)
        : is_an_implementation(neural::type_id<lrn_cpu_avx2_batch24>())
        , outer(outer)
    {
        if(outer.argument.size!=5) throw std::runtime_error("lrn_cpu_avx2_batch24: only normalization size of 5 currently supported");
        if(outer.argument.output_size.batch%24!=0) throw std::runtime_error("lrn_cpu_avx2_batch24: only batch size being multiple of 24 is currently supported");
        if(outer.argument.input_offset.batch!=0) throw std::runtime_error("lrn_cpu_avx2_batch24: only batch input offset 0 currently supported");
        if(outer.argument.output_offset.batch!=0) throw std::runtime_error("lrn_cpu_avx2_batch24: only batch output offset 0 currently supported");

        const uint32_t batch24_count = outer.argument.output_size.batch/24;
        auto size_y = outer.argument.output_size.spatial[1];
        auto size_x = outer.argument.output_size.spatial[0];
        task_data.resize(size_x*size_y*batch24_count);
        tasks.resize(size_x*size_y*batch24_count);
        const uint64_t value_stride = 24*outer.input_memory(0).argument.size.feature;
        const uint64_t batch24_stride = value_stride*size_x*size_y;
        for(uint32_t batch24_index=0; batch24_index<batch24_count; ++batch24_index)
            for(uint32_t y=0; y<size_y; ++y)
                for(uint32_t x=0; x<size_x; ++x) {
                    uint64_t offset = value_stride*(x+y*outer.input_memory(0).argument.size.spatial[0]) + batch24_stride*batch24_index;
                    float *input_ptr = static_cast<float *>(outer.input_memory(0).pointer) + offset;
                    float *outpt_ptr = static_cast<float *>(outer.output_memory(0).pointer) + offset;
                    auto index = x+size_x*(y+size_y*batch24_index);
                    task_data[index] = task_data_t{
                          input_ptr
                        , outpt_ptr
                        , outer.argument.output_size.feature
                        , outer.argument.alpha
                        , outer.argument.k
                    };
                    tasks[index] = { &implementation, &task_data[index] };
                }
    }

    task_group work() { return {tasks, schedule::unordered}; }
    static is_an_implementation *create(neural::normalization::response &arg) { return new lrn_cpu_avx2_batch24(arg); };

    static void implementation(const void *task_data) {
        auto ptr = static_cast<const task_data_t *>(task_data);
        auto feature_count  = ptr->feature_count;
        auto input_ptr      = ptr->input_ptr;
        auto output_ptr     = ptr->output_ptr;
        auto alpha          = ptr->alpha;
        auto k              = ptr->k;

        __m256 input[5][3];     // 15 registers
        __m256 squared[5][3];   // 15 registers

        // inputs: <0;1> = zero, <2;4> = data
        for(auto in=0; in<=1; ++in) for(auto at=0; at<=2; ++at) input[in][at] = _mm256_set1_ps(0.0f);
        for(auto in=2; in<=4; ++in) for(auto at=0; at<=2; ++at) {
            input[in][at] = _mm256_loadu_ps(input_ptr);
            input_ptr += 8;
        }

        // init
        for(auto in=0; in<=4; ++in) for(auto at=0; at<=2; ++at) squared[in][at] = _mm256_mul_ps(input[in][at], input[in][at]);

        // sum all squares
        __m256 output[3];
        for(auto at=0; at<=2; ++at) output[at] = squared[0][at];
        for(auto in=1; in<=4; ++in) for(auto at=0; at<=2; ++at) output[at] = _mm256_add_ps(output[at], squared[in][at]);

        // multiply by alpha, add k
        for(auto at=0; at<=2; ++at) output[at] = _mm256_add_ps(_mm256_mul_ps(output[at], _mm256_set1_ps(alpha)), _mm256_set1_ps(k));

        // output = x^-0.75
        for(auto at=0; at<=2; ++at) output[at] = _mm256_mul_ps(input[2][at], _invpow075_ps(output[at]));

        // store output
        for(auto at=0; at<=2; ++at) {
            _mm256_storeu_ps(output_ptr, output[at]);
            output_ptr += 8;
        }


        uint32_t current_reg = 0, current_feature=1;
        for(; current_feature<feature_count-2; ++current_feature) {
            for(auto at=0; at<=2; ++at) {
                input[current_reg][at] = _mm256_loadu_ps(input_ptr);
                input_ptr += 8;
                squared[current_reg][at] = _mm256_mul_ps(input[current_reg][at], input[current_reg][at]);
            }

            // sum all squares
            for(auto at=0; at<=2; ++at) output[at] = squared[0][at];
            for(auto in=1; in<=4; ++in) for(auto at=0; at<=2; ++at) output[at] = _mm256_add_ps(output[at], squared[in][at]);

            // multiply by alpha, add k
            for(auto at=0; at<=2; ++at) output[at] = _mm256_add_ps(_mm256_mul_ps(output[at], _mm256_set1_ps(alpha)), _mm256_set1_ps(k));

            // output = x^-0.75
            for(auto at=0; at<=2; ++at) output[at] = _mm256_mul_ps(input[(current_reg+3)%5][at], _invpow075_ps(output[at]));

            // store output
            for(auto at=0; at<=2; ++at) {
                _mm256_storeu_ps(output_ptr, output[at]);
                output_ptr += 8;
            }
            current_reg = (current_reg+1)%5;
        }

        for(; current_feature<feature_count; ++current_feature) {
            for(auto at=0; at<=2; ++at) {
                input[current_reg][at] = _mm256_setzero_ps();
                squared[current_reg][at] = _mm256_setzero_ps();
            }

            // sum all squares
            for(auto at=0; at<=2; ++at) output[at] = squared[0][at];
            for(auto in=1; in<=4; ++in) for(auto at=0; at<=2; ++at) output[at] = _mm256_add_ps(output[at], squared[in][at]);

            // multiply by alpha, add k
            for(auto at=0; at<=2; ++at) output[at] = _mm256_add_ps(_mm256_mul_ps(output[at], _mm256_set1_ps(alpha)), _mm256_set1_ps(k));

            // output = x^-0.75
            for(auto at=0; at<=2; ++at) output[at] = _mm256_mul_ps(input[(current_reg+3)%5][at], _invpow075_ps(output[at]));

            // store output
            for(auto at=0; at<=2; ++at) {
                _mm256_storeu_ps(output_ptr, output[at]);
                output_ptr += 8;
            }
            current_reg = (current_reg+1)%5;
        }

    };
};


namespace {
    struct attach {
        attach() {
            auto &map = lrn_fw_implementation_map::instance();
            map.insert({std::make_tuple(engine::cpu, memory::format::byxf_b24_f32, memory::format::byxf_b24_f32), lrn_cpu_avx2_batch24::create});
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

