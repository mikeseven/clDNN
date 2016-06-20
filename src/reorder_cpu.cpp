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
};
}

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
            map.insert( {std::make_tuple(engine::cpu, memory::format::byxf_b24_f32, memory::format::byxf_f32), reorder_cpu_byxf_f32_to_byxf_b24_f32::create} );
            map.insert( {std::make_tuple(engine::cpu, memory::format::byxf_f32,     memory::format::fyxb_f32), reorder_cpu_byxf_f32_to_fyxb_f32    ::create} );
            map.insert( {std::make_tuple(engine::cpu, memory::format::byxf_f32,     memory::format::bfyx_f32), reorder_cpu_bfyx_f32_to_byxf_f32    ::create} );
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
