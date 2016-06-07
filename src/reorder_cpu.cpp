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
#include "reorder.h"

using namespace neural;

struct reorder_cpu_byxf_f32_to_byxf_b24_f32 : is_an_implementation {
    const reorder &outer;
    reorder_cpu_byxf_f32_to_byxf_b24_f32(reorder &arg)
        : is_an_implementation(neural::type_id<reorder_cpu_byxf_f32_to_byxf_b24_f32>())
        , outer(arg)
    {};
    ~reorder_cpu_byxf_f32_to_byxf_b24_f32() {}

    static void implementation(const void *ptr) {
        auto this_reorder = static_cast<const reorder *>(ptr);
        auto input = static_cast<float*>(this_reorder->input_memory(0).pointer);
        auto output = static_cast<float*>(this_reorder->output_memory(0).pointer);
        auto& input_memory_arg  = this_reorder->input_memory(0).argument;

        uint32_t size_b = input_memory_arg.size.batch;
        uint32_t size_y = input_memory_arg.size.spatial[1];
        uint32_t size_x = input_memory_arg.size.spatial[0];
        uint32_t size_f = input_memory_arg.size.feature;
        for(uint32_t b=0; b<size_b; ++b)
            for(uint32_t y=0; y<size_y; ++y)
                for(uint32_t x=0; x<size_x; ++x)
                    for(uint32_t f=0; f<size_f; ++f) {
                        auto  input_index = f + size_f * (x + size_x * (y + size_y * b));
                        auto output_index = b%24 + 24 * (f + size_f * (x + size_x * (y + (b/24) * size_y)));
                        output[output_index] = input[input_index];
                    }
    }

    task_group work() {
        return {{task{implementation, &outer}}, schedule::unordered};
    }

    static is_an_implementation *create(reorder &arg) { return new reorder_cpu_byxf_f32_to_byxf_b24_f32(arg); };
};


namespace {

    struct attach {
        attach() {
            auto key_fw = std::make_tuple(engine::cpu, memory::format::byxf_b24_f32, memory::format::byxf_f32);
            auto val_fw = reorder_cpu_byxf_f32_to_byxf_b24_f32::create;
            reorder_fw_implementation_map::instance().insert( {key_fw, val_fw} );
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
