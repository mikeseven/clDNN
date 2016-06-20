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

#include "api/neural.h"
#include "api/neural_base.h"
#include "reorder.h"

#define XBYAK_NO_OP_NAMES
#define XBYAK_USE_MMAP_ALLOCATOR

#include "xbyak/xbyak_util.h"
#include <immintrin.h>

using namespace neural;

namespace{
struct op_data_t {
    uint32_t batch;
    float **output;
    float **input;
    uint32_t size_b;
    uint32_t size_y;
    uint32_t size_x;
    uint32_t size_f;

    uint32_t row;
};

class avx2_scatter : public ::Xbyak::CodeGenerator
{
public:
    void (*callback)(float*, float*);

    avx2_scatter(int stride, int fm_packages)
    {
        if(fm_packages < 1)
            throw std::runtime_error("avx2_scatter: at least one package is required");

        if(fm_packages > 4)
            throw std::runtime_error("avx2_scatter: max 4 packages are supported now");

        using namespace ::Xbyak;
        util::Cpu Current_cpu;

        if(Current_cpu.has(util::Cpu::tAVX2))
        {
#ifdef _WIN32
            const Reg64&   regarg_output_ptr = rcx;
            const Reg64&   regarg_input_ptr  = rdx;
#else
            const Reg64&   regarg_output_ptr = rdi;
            const Reg64&   regarg_input_ptr  = rsi;
#endif

            for(int fm_package = 0; fm_package < fm_packages; ++fm_package) vmovups(Ymm(fm_package*4),  ptr[regarg_input_ptr + 4*fm_package*8]);

            for(int fm_package = 0; fm_package < fm_packages; ++fm_package) vmovss(ptr[regarg_output_ptr + 4*0*stride + 4*fm_package*8*stride], Xmm(fm_package*4));
            for(int fm_package = 0; fm_package < fm_packages; ++fm_package) vpalignr(Xmm(fm_package*4+1), Xmm(fm_package*4),  Xmm(fm_package*4), 4);
            for(int fm_package = 0; fm_package < fm_packages; ++fm_package) vmovss(ptr[regarg_output_ptr + 4*1*stride + 4*fm_package*8*stride],  Xmm(fm_package*4+1));
            for(int fm_package = 0; fm_package < fm_packages; ++fm_package) vpalignr(Xmm(fm_package*4+2), Xmm(fm_package*4), Xmm(fm_package*4), 8);
            for(int fm_package = 0; fm_package < fm_packages; ++fm_package) vmovss(ptr[regarg_output_ptr + 4*2*stride + 4*fm_package*8*stride],  Xmm(fm_package*4+2));
            for(int fm_package = 0; fm_package < fm_packages; ++fm_package) vpalignr(Xmm(fm_package*4+3), Xmm(fm_package*4), Xmm(fm_package*4), 12);
            for(int fm_package = 0; fm_package < fm_packages; ++fm_package) vmovss(ptr[regarg_output_ptr + 4*3*stride + 4*fm_package*8*stride],  Xmm(fm_package*4+3));

            for(int fm_package = 0; fm_package < fm_packages; ++fm_package) vextractf128(Xmm(fm_package*4), Ymm(fm_package*4), 1);

            for(int fm_package = 0; fm_package < fm_packages; ++fm_package) vmovss(ptr[regarg_output_ptr + 4*4*stride + 4*fm_package*8*stride],  Xmm(fm_package*4));
            for(int fm_package = 0; fm_package < fm_packages; ++fm_package) vpalignr(Xmm(fm_package*4+1), Xmm(fm_package*4), Xmm(fm_package*4), 4);
            for(int fm_package = 0; fm_package < fm_packages; ++fm_package) vmovss(ptr[regarg_output_ptr + 4*5*stride + 4*fm_package*8*stride],  Xmm(fm_package*4+1));
            for(int fm_package = 0; fm_package < fm_packages; ++fm_package) vpalignr(Xmm(fm_package*4+2), Xmm(fm_package*4), Xmm(fm_package*4), 8);
            for(int fm_package = 0; fm_package < fm_packages; ++fm_package) vmovss(ptr[regarg_output_ptr + 4*6*stride + 4*fm_package*8*stride],  Xmm(fm_package*4+2));
            for(int fm_package = 0; fm_package < fm_packages; ++fm_package) vpalignr(Xmm(fm_package*4+3), Xmm(fm_package*4), Xmm(fm_package*4), 12);
            for(int fm_package = 0; fm_package < fm_packages; ++fm_package) vmovss(ptr[regarg_output_ptr + 4*7*stride + 4*fm_package*8*stride],  Xmm(fm_package*4+3));

            ret();

            callback = getCode<void (*)(float*, float*)>();
        }
        else
            throw std::runtime_error("AVX2 not supported by this machine.");
    }
} global_avx2_scatter_stride24_32fm(24, 4);
}

struct ULTRAFAST_reorder_cpu_byxf_f32_to_byxf_b24_f32 : is_an_implementation {
    const reorder &outer;

    std::vector<task> tasks;
    std::vector<op_data_t> op_array;

    ULTRAFAST_reorder_cpu_byxf_f32_to_byxf_b24_f32(reorder &arg)
        : is_an_implementation(neural::type_id<ULTRAFAST_reorder_cpu_byxf_f32_to_byxf_b24_f32>())
        , outer(arg)
    {
        const uint32_t batch = outer.input_memory(0).argument.size.batch;

        tasks.resize(batch, {implementation, nullptr});
        op_array.resize(batch, {0
                        , reinterpret_cast<float**>(&outer.output_memory(0).pointer)
                        , reinterpret_cast<float**>(&outer.input_memory(0).pointer)
                        , outer.input_memory(0).argument.size.batch[0]
                        , outer.input_memory(0).argument.size.spatial[1]
                        , outer.input_memory(0).argument.size.spatial[0]
                        , outer.input_memory(0).argument.size.feature[0]
                        , 0
        });

        for(uint32_t b = 0; b < batch; ++b){
            op_array[b].batch = b;
            tasks[b].data     = &op_array[b];
        }
    }
    ~ULTRAFAST_reorder_cpu_byxf_f32_to_byxf_b24_f32() {}
    task_group work() { return {tasks, schedule::unordered}; }
    static is_an_implementation *create(reorder &arg) { return new ULTRAFAST_reorder_cpu_byxf_f32_to_byxf_b24_f32(arg); }

    static void implementation(const void *ptr) {

        auto data       = static_cast<const op_data_t* >(ptr);
        auto input_ptr  = *data->input;
        auto output_ptr = *data->output;

        const uint64_t size_y = data->size_y;
        const uint64_t size_x = data->size_x;
        const uint64_t size_f = data->size_f;
        const uint64_t b = data->batch;

        //auto outoffs = b%24 + 24 * (f + size_f * (x + size_x * (y + (b/24) * size_y)));
        //auto inoffs = f + size_f * (x + size_x * (y + size_y * b));

        if(size_f % 32 == 0)
        {
            // Precomputed constants.
            const uint64_t fm_package_size = 32;
            const uint64_t in_f_stride = fm_package_size;
            const uint64_t out_f_stride = fm_package_size*24;
            const uint64_t in_xf_stride = size_f;
            const uint64_t out_xf_stride = size_f * 24;
            const uint64_t in_yxf_stride = size_x * size_f;
            const uint64_t out_yxf_stride = size_x * size_f * 24;

            auto in_b_offset = b * size_y * size_x * size_f;
            auto out_b_offset = b%24 + (b/24) * size_y * size_x * size_f * 24;

            for(uint64_t y=0; y<size_y; ++y)
            {
                auto in_by_offset = in_b_offset;
                auto out_by_offset = out_b_offset;

                for(uint64_t x=0; x<size_x; ++x)
                {
                    auto in_byx_offset = in_by_offset;
                    auto out_byx_offset = out_by_offset;

                    for(uint64_t f = 0; f < size_f; f += fm_package_size)
                    {
                        global_avx2_scatter_stride24_32fm.callback(&output_ptr[out_byx_offset], &input_ptr[in_byx_offset]);

                        in_byx_offset += in_f_stride;
                        out_byx_offset += out_f_stride;
                    }

                    in_by_offset += in_xf_stride;
                    out_by_offset += out_xf_stride;
                }

                in_b_offset += in_yxf_stride;
                out_b_offset += out_yxf_stride;
            }
        }
        else
        {
            // Precomputed constants.
            const uint64_t fm_package_size = 1;
            const uint64_t in_f_stride = fm_package_size;
            const uint64_t out_f_stride = fm_package_size*24;
            const uint64_t in_xf_stride = size_f;
            const uint64_t out_xf_stride = size_f * 24;
            const uint64_t in_yxf_stride = size_x * size_f;
            const uint64_t out_yxf_stride = size_x * size_f * 24;

            auto in_b_offset = b * size_y * size_x * size_f;
            auto out_b_offset = b%24 + (b/24) * size_y * size_x * size_f * 24;

            for(uint64_t y=0; y<size_y; ++y)
            {
                auto in_by_offset = in_b_offset;
                auto out_by_offset = out_b_offset;

                for(uint64_t x=0; x<size_x; ++x)
                {
                    auto in_byx_offset = in_by_offset;
                    auto out_byx_offset = out_by_offset;

                    for(uint64_t f = 0; f < size_f; f += fm_package_size)
                    {
                        output_ptr[out_byx_offset] = input_ptr[in_byx_offset];

                        in_byx_offset += in_f_stride;
                        out_byx_offset += out_f_stride;
                    }

                    in_by_offset += in_xf_stride;
                    out_by_offset += out_xf_stride;
                }

                in_b_offset += in_yxf_stride;
                out_b_offset += out_yxf_stride;
            }
        }
    }
};

struct ULTRAFAST_reorder_cpu_byxf_b24_f32_to_fyxb_f32 : is_an_implementation {
    const reorder &outer;

    std::vector<task> tasks;
    std::vector<op_data_t> op_array;

    ULTRAFAST_reorder_cpu_byxf_b24_f32_to_fyxb_f32(reorder &arg)
        : is_an_implementation(neural::type_id<ULTRAFAST_reorder_cpu_byxf_b24_f32_to_fyxb_f32>())
        , outer(arg)
    {
        const uint32_t row = outer.input_memory(0).argument.size.spatial[1];

        tasks.resize(row, {implementation, nullptr});
        op_array.resize(row, {0
                        , reinterpret_cast<float**>(&outer.output_memory(0).pointer)
                        , reinterpret_cast<float**>(&outer.input_memory(0).pointer)
                        , outer.input_memory(0).argument.size.batch[0]
                        , outer.input_memory(0).argument.size.spatial[1]
                        , outer.input_memory(0).argument.size.spatial[0]
                        , outer.input_memory(0).argument.size.feature[0]
                        , 0
        });

        for(uint32_t r = 0; r < row; ++r){
            op_array[r].row = r;
            tasks[r].data     = &op_array[r];
        }
    }
    ~ULTRAFAST_reorder_cpu_byxf_b24_f32_to_fyxb_f32() {}
    task_group work() { return {tasks, schedule::unordered}; }
    static is_an_implementation *create(reorder &arg) { return new ULTRAFAST_reorder_cpu_byxf_b24_f32_to_fyxb_f32(arg); }

    static void implementation(const void *ptr) {
        auto data       = static_cast<const op_data_t* >(ptr);
        auto input_ptr  = *data->input;
        auto output_ptr = *data->output;

        const uint64_t size_y = data->size_y;
        const uint64_t size_x = data->size_x;
        const uint64_t size_f = data->size_f;
        const uint64_t size_b = data->size_b;
        const uint64_t y = data->row;

        //auto inoffs = b%24 + 24 * (f + size_f * (x + size_x * (y + (b/24) * size_y)));
        //auto outoffs = b + size_b * (x + size_x * (y + size_y * f));

        // Precomputed constants.
        const uint64_t in_f_stride = 24;
        const uint64_t out_f_stride = size_y * size_x * size_b;
        const uint64_t in_xf_stride = size_f * 24;
        const uint64_t out_xf_stride = size_b;
        const uint64_t in_yxf_stride = size_y * size_x * size_f * 24;
        const uint64_t out_yxf_stride = 24;

        {
            auto in_y_offset = y * size_x * size_f * 24;
            auto out_y_offset = y * size_x * size_b;

            for(uint64_t b=0; b<size_b; b+=24)
            {
                auto in_yb_offset = in_y_offset;
                auto out_yb_offset = out_y_offset;

                for(uint64_t x=0; x<size_x; ++x)
                {
                    auto in_xyb_offset = in_yb_offset;
                    auto out_xyb_offset = out_yb_offset;

                    for(uint64_t f=0; f<size_f; ++f)
                    {
                            __m256 acc0 = _mm256_loadu_ps(&input_ptr[in_xyb_offset +  0]);
                            __m256 acc1 = _mm256_loadu_ps(&input_ptr[in_xyb_offset +  8]);
                            __m256 acc2 = _mm256_loadu_ps(&input_ptr[in_xyb_offset + 16]);
                            _mm256_storeu_ps(&output_ptr[out_xyb_offset +  0], acc0);
                            _mm256_storeu_ps(&output_ptr[out_xyb_offset +  8], acc1);
                            _mm256_storeu_ps(&output_ptr[out_xyb_offset + 16], acc2);

                            in_xyb_offset += in_f_stride;
                            out_xyb_offset += out_f_stride;
                    }

                    in_yb_offset += in_xf_stride;
                    out_yb_offset += out_xf_stride;
                }

                in_y_offset += in_yxf_stride;
                out_y_offset += out_yxf_stride;
            }
        }
    }
};

struct reorder_cpu_byxf_f32_to_byxf_b24_f32 : is_an_implementation {
    const reorder &outer;

    std::vector<task> tasks;
    std::vector<op_data_t> op_array;

    reorder_cpu_byxf_f32_to_byxf_b24_f32(reorder &arg)
        : is_an_implementation(neural::type_id<reorder_cpu_byxf_f32_to_byxf_b24_f32>())
        , outer(arg)
    {
        const uint32_t batch = outer.input_memory(0).argument.size.batch;

        tasks.resize(batch, {implementation, nullptr});
        op_array.resize(batch, {0
                                , reinterpret_cast<float**>(&outer.output_memory(0).pointer)
                                , reinterpret_cast<float**>(&outer.input_memory(0).pointer)
                                , outer.input_memory(0).argument.size.batch[0]
                                , outer.input_memory(0).argument.size.spatial[1]
                                , outer.input_memory(0).argument.size.spatial[0]
                                , outer.input_memory(0).argument.size.feature[0]
                                , 0
                        });

        for(uint32_t b = 0; b < batch; ++b){
            op_array[b].batch = b;
            tasks[b].data     = &op_array[b];
        }
    }
    ~reorder_cpu_byxf_f32_to_byxf_b24_f32() {}
    task_group work() { return {tasks, schedule::unordered}; }
    static is_an_implementation *create(reorder &arg) { return new reorder_cpu_byxf_f32_to_byxf_b24_f32(arg); }

   static void implementation(const void *ptr) {
        auto data       = static_cast<const op_data_t* >(ptr);
        auto input_ptr  = *data->input;
        auto output_ptr = *data->output;

        const uint32_t size_y = data->size_y;
        const uint32_t size_x = data->size_x;
        const uint32_t size_f = data->size_f;
        const uint32_t b      = data->batch;

        for(uint32_t y=0; y<size_y; ++y)
            for(uint32_t x=0; x<size_x; ++x)
                for(uint32_t f=0; f<size_f; ++f) {
                    auto  input_index = f + size_f * (x + size_x * (y + size_y * b));
                    auto output_index = b%24 + 24 * (f + size_f * (x + size_x * (y + (b/24) * size_y)));
                    output_ptr[output_index] = input_ptr[input_index];
                }
    }

};

struct reorder_cpu_bfyx_f32_to_byxf_f32 : is_an_implementation {
    const reorder &outer;

    std::vector<task> tasks;
    std::vector<op_data_t> op_array;

    reorder_cpu_bfyx_f32_to_byxf_f32(reorder &arg)
        : is_an_implementation(neural::type_id<reorder_cpu_bfyx_f32_to_byxf_f32>())
        , outer(arg)
    {
        const uint32_t batch = outer.input_memory(0).argument.size.batch;

        tasks.resize(batch, {implementation, nullptr});
        op_array.resize(batch, {0
                                , reinterpret_cast<float**>(&outer.output_memory(0).pointer)
                                , reinterpret_cast<float**>(&outer.input_memory(0).pointer)
                                , outer.input_memory(0).argument.size.batch[0]
                                , outer.input_memory(0).argument.size.spatial[1]
                                , outer.input_memory(0).argument.size.spatial[0]
                                , outer.input_memory(0).argument.size.feature[0]
                                , 0
                        });

        for(uint32_t b = 0; b < batch; ++b){
            op_array[b].batch = b;
            tasks[b].data     = &op_array[b];
        }
    }
    ~reorder_cpu_bfyx_f32_to_byxf_f32() {}
    task_group work() { return {tasks, schedule::unordered}; }
    static is_an_implementation *create(reorder &arg) { return new reorder_cpu_bfyx_f32_to_byxf_f32(arg); }

    static void implementation(const void *ptr) {
        auto data       = static_cast<const op_data_t* >(ptr);
        auto input_ptr  = *data->input;
        auto output_ptr = *data->output;

        const uint32_t size_y = data->size_y;
        const uint32_t size_x = data->size_x;
        const uint32_t size_f = data->size_f;
        const uint32_t b      = data->batch;

        for(uint32_t y=0; y<size_y; ++y)
            for(uint32_t x=0; x<size_x; ++x)
                for(uint32_t f=0; f<size_f; ++f) {
                    auto  input_index = x + size_x * (y + size_y * (f + size_f * b));
                    auto output_index = f + size_f * (x + size_x * (y + size_y * b));
                    output_ptr[output_index] = input_ptr[input_index];
                }
    }
};

struct reorder_cpu_byxf_f32_to_fyxb_f32 : is_an_implementation {
    const reorder &outer;

    std::vector<task> tasks;
    std::vector<op_data_t> op_array;

    reorder_cpu_byxf_f32_to_fyxb_f32(reorder &arg)
        : is_an_implementation(neural::type_id<reorder_cpu_byxf_f32_to_fyxb_f32>())
        , outer(arg)
    {
        const uint32_t batch = outer.input_memory(0).argument.size.batch;

        tasks.resize(batch, {implementation, nullptr});
        op_array.resize(batch, {0
                                , reinterpret_cast<float**>(&outer.output_memory(0).pointer)
                                , reinterpret_cast<float**>(&outer.input_memory(0).pointer)
                                , outer.input_memory(0).argument.size.batch[0]
                                , outer.input_memory(0).argument.size.spatial[1]
                                , outer.input_memory(0).argument.size.spatial[0]
                                , outer.input_memory(0).argument.size.feature[0]
                                , 0
                        });

        for(uint32_t b = 0; b < batch; ++b){
            op_array[b].batch = b;
            tasks[b].data     = &op_array[b];
        }
    }
    ~reorder_cpu_byxf_f32_to_fyxb_f32() {}
    task_group work() { return {tasks, schedule::unordered}; }
    static is_an_implementation *create(reorder &arg) { return new reorder_cpu_byxf_f32_to_fyxb_f32(arg); }

   static void implementation(const void *ptr) {
        auto data       = static_cast<const op_data_t* >(ptr);
        auto input_ptr  = *data->input;
        auto output_ptr = *data->output;

        const uint32_t size_y = data->size_y;
        const uint32_t size_x = data->size_x;
        const uint32_t size_f = data->size_f;
        const uint32_t size_b = data->size_b;
        const uint32_t b      = data->batch;

        for(uint32_t y=0; y<size_y; ++y)
            for(uint32_t x=0; x<size_x; ++x)
                for(uint32_t f=0; f<size_f; ++f) {
                    auto  input_index = f + size_f * (x + size_x * (y + size_y * b));
                    auto output_index = b + size_b * (x + size_x * (y + size_y * f));
                    output_ptr[output_index] = input_ptr[input_index];
                }
    }

};


namespace {
    struct attach {
        attach() {
            auto &map = reorder_fw_implementation_map::instance();
                                      // engine       output                        input                      create function
            map.insert( {std::make_tuple(engine::cpu, memory::format::byxf_b24_f32, memory::format::byxf_f32), ULTRAFAST_reorder_cpu_byxf_f32_to_byxf_b24_f32::create} );
            map.insert( {std::make_tuple(engine::cpu, memory::format::fyxb_f32,     memory::format::byxf_b24_f32), ULTRAFAST_reorder_cpu_byxf_b24_f32_to_fyxb_f32::create} );
        }
        ~attach(){}
    };
#ifdef __GNUC__
    __attribute__((visibility("default")))
#elif _MSC_VER
#   pragma section(".nn_init$m", read, write)
#endif
    attach attach_impl;

}
