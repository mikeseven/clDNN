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
#include "lrn_cpu_reference.h"

namespace neural {

    lrn_cpu_reference::lrn_cpu_reference(normalization::response &arg)
        : is_an_implementation(neural::type_id<lrn_cpu_reference>())
        , outer(arg) {};

    lrn_cpu_reference::~lrn_cpu_reference() {};

    void lrn_cpu_reference::implementation(const void *ptr) {

        auto this_lrn = static_cast<const normalization::response *>(ptr);

        auto& input_offset = this_lrn->argument.input_offset;
        auto& output_offset = this_lrn->argument.output_offset;
        auto& output_size = this_lrn->argument.output_size;
        auto& padding = this_lrn->argument.padding;
        auto& size = this_lrn->argument.size;

        auto& k = this_lrn->argument.k;
        auto& alpha = this_lrn->argument.alpha;
        auto& beta = this_lrn->argument.beta;


        //auto input_arg  = this_lrn->input_memory(0).argument;
        //auto output_arg = this_lrn->output_memory(0).argument;
        auto input_arg = this_lrn->argument.input[0].primitive.as<const memory&>().argument; //todo tmp solution
        auto output_arg = this_lrn->argument.output[0].as<const memory&>().argument;

        if (input_arg.size.raw.size() != output_arg.size.raw.size())   throw std::runtime_error("lrn input/output number of dimension does not match.");
        if (input_arg.format != memory::format::yxfb_f32) throw std::runtime_error("lrn reference uses yxfb_f32 format.");             // only yxfb_f32 format is supported

        //auto input  = static_cast<float*>(this_lrn->input_memory(0).pointer);
        //auto output = static_cast<float*>(this_lrn->output_memory(0).pointer);
        auto input = static_cast<float*>(this_lrn->argument.input[0].primitive.as<const memory&>().pointer);  //todo tmp solution
        auto output = static_cast<float*>(this_lrn->argument.output[0].as<const memory&>().pointer);

        namespace nd = ndimensional;
        nd::value<uint32_t> range(output_size);
        nd::calculate_idx<uint32_t, memory::format::yxfb_f32> calc_in_idx(input_arg.size);
        nd::calculate_idx<uint32_t, memory::format::yxfb_f32> calc_out_idx(output_arg.size);
        nd::value<uint32_t> window_range({1,{1,1},size});

        vector<int32_t> help_input_offset({input_offset});

        switch (padding) {
        case padding::zero:
            help_input_offset.feature[0] -= static_cast<int32_t>(size / 2);
            for (auto pos : range) {
                auto out_idx = calc_out_idx(pos + output_offset);
                float acc = 0.f;
                float sum_of_products = 0.f;
                for (auto window_pos : window_range) {
                    auto input_pos = pos - help_input_offset + window_pos;
                    auto input_index = calc_in_idx(input_pos);
                    acc = input[input_index];
                };
            }
            /*
            for (auto pos : range) {
                float acc = 0;
                auto out_idx = calc_out_idx(pos + output_offset);

                for (auto win_pos : window_range) {
                    const std::vector<int32_t> arg_in_idx = nd::value<int32_t>(input_offset) + pos*stride + win_pos;

                    if (calc_in_idx.is_out_of_range(arg_in_idx))
                        continue;

                    auto in_idx = calc_in_idx(arg_in_idx);
                    auto win_idx = calc_win_idx(win_pos);
                    acc += input[in_idx] * filter[win_idx];
                }
                output[out_idx] = acc + bias[pos[f_pos]]; // todo need type traits for index of 'f' dimension
            }
            */
            break;
        default:
            throw std::runtime_error("Unknown padding mode in lrn");
        }
    }


    namespace {
        struct attach {
            attach() {
                auto key = std::make_tuple(engine::reference, memory::format::yxfb_f32, memory::format::yxfb_f32);
                auto val_fw = lrn_cpu_reference::create;
                //auto val_bw = lrn_backward_cpu_reference::create;

                lrn_fw_implementation_map.insert({ key, val_fw }); //todo keys should be different
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