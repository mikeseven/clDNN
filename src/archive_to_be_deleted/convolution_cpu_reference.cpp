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
#include "convolution_cpu_reference.h"
#include "implementation_map.h"
#include "multidimensional_counter.h"
#include "memory_utils.h"

namespace neural {

convolution_cpu_reference::convolution_cpu_reference(convolution &arg)
        : is_an_implementation(neural::type_id<convolution_cpu_reference>())
        , outer(arg) {}
convolution_cpu_reference::~convolution_cpu_reference() {}
void convolution_cpu_reference::implementation(const void *ptr) {
    auto this_conv = static_cast<const convolution *>(ptr);

    auto& input_offset  = this_conv->argument.input_offset;
    auto& output_offset = this_conv->argument.output_offset;
    auto& output_size   = this_conv->argument.output_size;
    auto& padding       = this_conv->argument.padding;
    auto& stride        = this_conv->argument.stride;

    auto& input_arg  = this_conv->input_memory(0).argument;
    auto& output_arg = this_conv->output_memory(0).argument;

    auto& filter_arg = this_conv->input_memory(1).argument; //convolution filter

    auto split = this_conv->argument.split;

    assert( output_size.feature[0] / split == filter_arg.size.feature[0] ); // memory::format oixy

    // todo remove
    if(filter_arg.format != memory::format::oiyx_f32) throw std::runtime_error("conv weights arent oiyx_f32 format");

    auto input  = this_conv->input_memory(0).pointer<float>();
    auto output = this_conv->output_memory(0).pointer<float>();
    std::vector<memory::ptr<float>> filters;
    std::vector<memory::ptr<float>> biases;
    for (size_t i = 0; i < split; i++)
    {
        filters.push_back(this_conv->input_memory(i * 2 + 1).pointer<float>());
        biases.push_back(this_conv->input_memory(i * 2 + 2).pointer<float>());
    }

    const int f_pos = 1; // neural::vector format is b,f,spatials. In input and output 'b' and 'f' fields are always scalars.
    namespace nd = ndimensional;
    nd::value<uint32_t> range (output_size);

    // weights neural::vector is: {b}, {ofm, ifm} {spatials}
    // ofm - output feature maps
    // ifm - input feature maps
    // b = 1 always
    // (weights can't have batch so it is equall to 1)
    // Ofm and batch is cropped, ofm will be hold manually
    // Batch is included in output size
    nd::value<uint32_t> window_range_truncated ({filter_arg.size.raw.cbegin()+2, filter_arg.size.raw.cend()});
    auto calc_in_idx  = nd::choose_calculate_idx(input_arg.format);
    auto calc_out_idx = nd::choose_calculate_idx(output_arg.format);
    auto calc_win_idx = nd::choose_calculate_idx(filter_arg.format);

    size_t batch_num = output_arg.size.batch[0];
    size_t output_feature_num = output_arg.size.feature[0];

    const size_t fm_offset_per_computation = input_arg.size.feature[0] / split;

    switch(padding){
        case padding::zero:
        {
            for(auto pos : range) {
                auto out_idx = calc_out_idx(output_arg.size.raw, pos + output_offset);
                auto feature_map_idx = (out_idx / batch_num) % output_feature_num;
                auto split_idx = feature_map_idx / (output_feature_num / split);
                output[out_idx] = biases[split_idx][pos[f_pos] / split];
            }

            // Ofm in weights and feature maps in output size is the same.
            // Iteration through ofm will be done in weights loop
            // so feature maps here are modified to not iterate 2 times through the same data.
            range[1] = 1;

            for(auto pos : range) { // {b}, {1}, {spatials}
                for(auto win_pos : window_range_truncated){ // {ifm}, {spatials}
                    for(uint32_t ofm = output_offset.feature[0]; ofm < output_offset.feature[0]+output_size.feature[0]; ++ofm){ // ofm
                        auto pos_with_modified_ofm(pos); // assign current ofm to output position
                        pos_with_modified_ofm[1] = ofm;

                        std::vector<uint32_t> arg_in_idx = pos*stride + input_offset + win_pos;

                        if( nd::is_out_of_range(input_arg.size, arg_in_idx) )
                            continue;

                        auto out_idx = calc_out_idx(output_arg.size.raw, pos_with_modified_ofm);

                        size_t split_idx = out_idx / batch_num % split;

                        auto in_idx  = calc_in_idx ( input_arg.size.raw, {arg_in_idx.begin(), arg_in_idx.end()} );
                        // We got 2x window for split=2, and we iterate over this window 2 times with different values for each of data sets,
                        // so we need to offset input position based on which part of split computations we are. If we don't do this, we will iterate
                        // two times over the same half of input values.
                        in_idx += split_idx * fm_offset_per_computation;

                        auto win_idx = calc_win_idx( filter_arg.size.raw,
                                                     [&](){
                                                        auto vec = std::vector<uint32_t>({0, (ofm-output_offset.feature[0]) / static_cast<uint32_t>(split)});
                                                        auto* win_pos_ptr = static_cast<std::vector<uint32_t>*>(&win_pos);
                                                        vec.insert(vec.end(), win_pos_ptr->begin(), win_pos_ptr->end());
                                                        return vec;
                                                     }()
                                                    );

                        output[out_idx] += input[in_idx] * filters[split_idx][win_idx];
                    }
                }
            }
            if (this_conv->argument.use_relu)
            {
                for (auto pos : range) {
                    auto out_idx = calc_out_idx(output_arg.size.raw, pos + output_offset);

                    output[out_idx] = std::max(output[out_idx], 0.0f) + this_conv->argument.negative_slope * std::min(output[out_idx], 0.0f);
                }
            }
        }
            break;
        default:
            throw std::runtime_error("Unknown padding mode in convolution.");
    }
}

namespace{
struct attach{
    attach(){
        auto key_fw = std::make_tuple(engine::reference, memory::format::yxfb_f32, memory::format::yxfb_f32);

        auto val_fw = convolution_cpu_reference::create;

        implementation_map<convolution>::add( key_fw, val_fw );
    }
    ~attach(){}
};

#ifdef __GNUC__
    __attribute__((visibility("default"))) //todo meybe dll_sym?
#elif _MSC_VER
#   pragma section(".nn_init$m", read, write)
#endif
attach attach_impl;

}
}
