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

#include "softmax_cpu_reference.h"
#include "multidimensional_counter.h"
#include "implementation_map.h"

namespace neural {
namespace normalization {

softmax_cpu_reference::softmax_cpu_reference(softmax &arg)
    : is_an_implementation(neural::type_id<softmax_cpu_reference>())
    , outer(arg) {}

softmax_cpu_reference::~softmax_cpu_reference() {}

void softmax_cpu_reference::implementation(const void *ptr) {
    auto this_softmax = static_cast<const softmax *>(ptr);
    auto input        = this_softmax->input_memory(0).pointer<float>();
    auto output       = this_softmax->output_memory(0).pointer<float>();

    auto& input_offset  = this_softmax->argument.input_offset;
    auto& output_offset = this_softmax->argument.output_offset;
    auto& output_size   = this_softmax->argument.output_size;

    auto& input_arg  = this_softmax->input_memory(0).argument;
    auto& output_arg = this_softmax->output_memory(0).argument;

    assert( 1 == output_size.feature.size() );
    assert( 1 == output_size.batch.size()   );
    int batch_index = 0;
    std::vector<float> v_max( output_size.batch[0], -std::numeric_limits<float>::max() );
    std::vector<float> v_acc( output_size.batch[0] );

    namespace nd = ndimensional;
    nd::value<uint32_t> range (output_size);
    auto calc_in_idx  = nd::choose_calculate_idx(input_arg.format);
    auto calc_out_idx = nd::choose_calculate_idx(output_arg.format);

    // find max val per batch
    for(auto pos : range) {
        auto in_idx  = calc_in_idx (input_arg.size.raw , pos + input_offset );
        v_max[ pos[batch_index] ] = std::max( v_max[pos[batch_index]], input[in_idx]);
    }
    for(auto pos : range) {
        auto in_idx  = calc_in_idx (input_arg.size.raw , pos + input_offset );
        auto out_idx = calc_out_idx(output_arg.size.raw, pos + output_offset);

        output[out_idx] = input[in_idx] - v_max[ pos[batch_index] ]; // subtracte max val from every data point per batch
        output[out_idx] = std::exp(output[out_idx]);  // exp
        v_acc[ pos[batch_index] ] += output[out_idx]; // sum eveything per batch
    }
    for(auto pos : range) {
        auto out_idx = calc_out_idx(output_arg.size.raw, pos + output_offset);
        output[out_idx] /= v_acc[ pos[batch_index] ]; // compute softmax
    }
}

namespace {
struct attach {
    attach() {
        auto key_fw = std::make_tuple(engine::reference, memory::format::xb_f32, memory::format::xb_f32);
        auto val_fw = softmax_cpu_reference::create;

        implementation_map<softmax>::add(key_fw, val_fw);
    }
    ~attach() {}
};
}

#ifdef __GNUC__
    __attribute__((visibility("default"))) //todo meybe dll_sym?
#elif _MSC_VER
#   pragma section(".nn_init$m", read, write)
#endif
attach attach_impl;

} // namespace normalization
} // namespace neural
