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

#include <iterator>
#include "mean_subtract_cpu_reference.h"
#include "implementation_map.h"
#include "multidimensional_counter.h"
#include "memory_utils.h"

namespace neural {

    mean_subtract_cpu_reference::mean_subtract_cpu_reference(mean_subtract &arg)
        : is_an_implementation(neural::type_id<mean_subtract_cpu_reference>())
        , outer(arg) {};
    mean_subtract_cpu_reference::~mean_subtract_cpu_reference() {};
    void mean_subtract_cpu_reference::implementation(const void *ptr) {
        auto this_mean = static_cast<const mean_subtract *>(ptr);

        auto& output_arg = this_mean->output_memory(0).argument;

        auto& mean_arg = this_mean->argument.input[1].primitive.as<const memory&>().argument; //mean

        if (mean_arg.format != memory::format::yxfb_f32) throw std::runtime_error("mean_subtract mean isn't yxfb_f32 format");

        auto input = this_mean->input_memory(0).pointer<float>();
        auto output = this_mean->output_memory(0).pointer<float>();
        auto mean = this_mean->argument.input[1].primitive.as<const memory&>().pointer<float>();

        namespace nd = ndimensional;
        nd::value<uint32_t> range(this_mean->output_memory(0).argument.size.raw);

        auto calc_out_idx = nd::choose_calculate_idx(output_arg.format);

        for (auto pos : range) {
            auto out_idx = calc_out_idx(output_arg.size.raw, pos);
            // TODO: this is temporary solution. have to be done properly
            output[out_idx] = input[out_idx] - mean[(out_idx % 3) * mean_arg.size.spatial[0] * mean_arg.size.spatial[1]];
        }
    }


    namespace {
        struct attach {
            attach() {
                auto key_fw = std::make_tuple(engine::reference, memory::format::yxfb_f32, memory::format::yxfb_f32);
                auto val_fw = mean_subtract_cpu_reference::create;

                implementation_map<mean_subtract>::add(key_fw, val_fw);
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
