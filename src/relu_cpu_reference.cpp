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

#include "relu_cpu_reference.h"
#include "multidimensional_counter.h"
#include "implementation_map.h"

namespace neural {

relu_cpu_reference::relu_cpu_reference(relu &arg)
    : is_an_implementation(neural::type_id<relu_cpu_reference>())
    , outer(arg) {}

relu_cpu_reference::~relu_cpu_reference() {}

void relu_cpu_reference::implementation(const void *ptr) {
    auto this_relu = static_cast<const relu *>(ptr);

    auto& input_offset  = this_relu->argument.input_offset;
    auto& output_offset = this_relu->argument.output_offset;
    auto& output_size   = this_relu->argument.output_size;

    auto& input_arg  = this_relu->input_memory(0).argument;
    auto& output_arg = this_relu->output_memory(0).argument;

    assert( 1 == output_size.feature.size() );
    assert( 1 == output_size.batch.size()   );

    auto input  = this_relu->input_memory(0).pointer<float>();
    auto output = this_relu->output_memory(0).pointer<float>();

    namespace nd = ndimensional;
    nd::value<uint32_t> range ( output_size );
    auto calc_in_idx  = nd::choose_calculate_idx(input_arg.format);
    auto calc_out_idx = nd::choose_calculate_idx(output_arg.format);

    for(auto pos : range) {
        auto in_idx  = calc_in_idx (input_arg.size.raw , pos + input_offset );
        auto out_idx = calc_out_idx(output_arg.size.raw, pos + output_offset);

        output[out_idx] = std::max( input[in_idx], 0.0f) + this_relu->argument.negative_slope * std::min( input[in_idx], 0.0f);
    }
}

namespace {
struct attach {
    attach() {
        auto key_fw = std::make_tuple(engine::reference, memory::format::yxfb_f32, memory::format::yxfb_f32);
        auto key_bw = std::make_tuple(engine::reference, memory::format::yxfb_f32, memory::format::yxfb_f32);
        auto val_fw = relu_cpu_reference::create;

        implementation_map<relu>::add(key_fw, val_fw); //todo keys should be different
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

