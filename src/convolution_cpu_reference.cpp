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

#include "convolution_cpu_reference.h"
#include "multidimensional_counter.h"
#include "memory_utils.h"

namespace neural {

convolution_cpu_reference::convolution_cpu_reference(convolution &arg)
        : is_an_implementation(neural::type_id<convolution_cpu_reference>())
        , outer(arg) {};
convolution_cpu_reference::~convolution_cpu_reference() {};
void convolution_cpu_reference::implementation(const void *ptr) {
    auto this_conv = static_cast<const convolution *>(ptr);

    auto& input_offset  = this_conv->argument.input_offset;
    auto& output_offset = this_conv->argument.output_offset;
    auto& output_size   = this_conv->argument.output_size;
    auto& padding       = this_conv->argument.padding;
    auto& stride        = this_conv->argument.stride;

    auto& input_arg  = this_conv->input_memory(0).argument;
    auto& output_arg = this_conv->output_memory(0).argument;

    auto& filter_arg = this_conv->argument.weight.as<const memory&>().argument; //convolution filter

    auto input  = static_cast<float*>(this_conv->input_memory(0).pointer);
    auto output = static_cast<float*>(this_conv->output_memory(0).pointer);
    auto filter = static_cast<float*>(this_conv->argument.weight.as<const memory&>().pointer);
    auto bias   = static_cast<float*>(this_conv->argument.bias.as<const memory&>().pointer);

    //for(size_t i = 0; i < input_offset.raw.size(); ++i){
    //    // general formula: output size = (input size - filter size) / step + 1
    //    if(output_size[i] <
    //        std::abs(static_cast<int32_t>(input_arg.size[i] - input_offset[i] - filter_arg.size[i])) / stride[i] + 1) //todo is it safe?
    //        if(filter_arg.size[i] <= output_size[i])
    //            throw std::runtime_error("Output size of convolution is to small.");

    //    if(output_arg.size[i] < output_size[i] + output_offset[i])
    //        throw std::runtime_error("Convolution output buffer size is to small.");
    //}

    int f_pos = 1; // neural::vector format is b,f,spatials
    namespace nd = ndimensional;
    nd::value<uint32_t> range (output_size);
    nd::value<uint32_t> window_range (filter_arg.size);
    auto calc_in_idx  = nd::choose_calculate_idx(input_arg.format);
    auto calc_out_idx = nd::choose_calculate_idx(output_arg.format);
    auto calc_win_idx = nd::choose_calculate_idx(filter_arg.format);

    switch(padding){
        case padding::zero:
            for(auto pos : range) {
                float acc = 0;
                auto out_idx = calc_out_idx(output_arg.size.raw, pos + output_offset);

                for(auto win_pos : window_range){
                    const std::vector<int32_t> arg_in_idx = nd::value<int32_t>(input_offset) + pos*stride + win_pos;

                    if( nd::is_out_of_range(input_arg.size, arg_in_idx) )
                        continue;

                    auto in_idx  = calc_in_idx (input_arg.size.raw, {arg_in_idx.begin(), arg_in_idx.end()} );
                    auto win_idx = calc_win_idx(filter_arg.size.raw, win_pos );
                    acc += input[in_idx] * filter[win_idx];
                }
                output[out_idx] = acc + bias[ pos[f_pos] ];
            }
            break;
        default:
            throw std::runtime_error("Unknown padding mode in convolution.");
    }
}

convolution_backward_cpu_reference::convolution_backward_cpu_reference(convolution_backward &arg)
    : is_an_implementation(neural::type_id<convolution_backward_cpu_reference>())
    , outer(arg) {};
convolution_backward_cpu_reference::~convolution_backward_cpu_reference() {};
void convolution_backward_cpu_reference::implementation(const void *ptr) { //todo tests
    auto this_bw_conv = static_cast<const convolution_backward *>(ptr);

    auto& bw_input_size    = this_bw_conv->argument.input_size;  // todo output or input?
    auto& bw_input_offset  = this_bw_conv->argument.input_offset;
    auto& bw_output_offset = this_bw_conv->argument.output_offset;
    auto& stride           = this_bw_conv->argument.stride;
    auto& padding          = this_bw_conv->argument.padding;

    auto& bw_input_arg     = this_bw_conv->input_memory(0).argument;
    auto& filter_arg       = this_bw_conv->input_memory(2).argument;
    auto& bias_arg         = this_bw_conv->input_memory(3).argument; //todo bias isn't needed in bw conv. It is only used to compare its size with bias_diff. Remove?


    auto& bw_output_arg    = this_bw_conv->argument.output[0].as<const memory&>().argument;

    auto bw_input     = static_cast<float*>(this_bw_conv->argument.input[0].primitive.as<const memory&>().pointer);
    auto fw_input     = static_cast<float*>(this_bw_conv->argument.input[1].primitive.as<const memory&>().pointer);
    auto weights      = static_cast<float*>(this_bw_conv->argument.input[2].primitive.as<const memory&>().pointer);

    auto bw_output    = static_cast<float*>(this_bw_conv->argument.output[0].as<const memory&>().pointer);
    auto weights_diff = static_cast<float*>(this_bw_conv->argument.output[1].as<const memory&>().pointer);
    auto bias_diff    = static_cast<float*>(this_bw_conv->argument.output[2].as<const memory&>().pointer);

    // initializie gradients with 0
    fill(this_bw_conv->output_memory(0), 0.0f);
    fill(this_bw_conv->output_memory(1), 0.0f);
    fill(this_bw_conv->output_memory(2), 0.0f);

    const int F_POS = 1;
    namespace nd = ndimensional;
    nd::value<uint32_t> bias_range (bias_arg.size);
    nd::value<uint32_t> range (bw_input_size); //todo in/out size?
    nd::value<uint32_t> window_range (filter_arg.size);
    auto calc_in_idx   = nd::choose_calculate_idx(bw_input_arg.format);
    auto calc_out_idx  = nd::choose_calculate_idx(bw_output_arg.format);
    auto calc_win_idx  = nd::choose_calculate_idx(filter_arg.format);

    switch(padding){
        case padding::zero:
        {
            for(auto pos : range) {
                auto in_idx = calc_in_idx(bw_input_arg.size.raw , pos + bw_input_offset);

                for(auto win_pos : window_range){
                    const std::vector<uint32_t> arg_out_idx = nd::value<uint32_t>(bw_output_offset) + pos*stride + win_pos;

                    if( nd::is_out_of_range(bw_output_arg.size, arg_out_idx) )
                        continue;

                    auto out_idx = calc_out_idx(bw_output_arg.size.raw, arg_out_idx);
                    auto win_idx = calc_win_idx(filter_arg.size.raw, win_pos);

                    auto sensitivity = bw_input[in_idx] * weights[win_idx];

                    bw_output[out_idx] += sensitivity;
                    weights_diff[win_idx] += fw_input[out_idx] * sensitivity;
                }
                bias_diff[ pos[F_POS] ] += bw_input[in_idx];
            }
            break;
        }
        default:
            throw std::runtime_error("Unknown padding mode in backward convolution.");
    }
}

namespace{
struct attach{
    attach(){
        auto key_fw = std::make_tuple(engine::reference, memory::format::yxfb_f32, memory::format::yxfb_f32);
        auto key_bw = std::make_tuple(engine::reference, memory::format::yxfb_f32, memory::format::yxfb_f32);
        auto val_fw = convolution_cpu_reference::create;
        auto val_bw = convolution_backward_cpu_reference::create;

        conv_fw_implementation_map.insert( {key_fw, val_fw} ); //todo keys should be different
        conv_bw_implementation_map.insert( {key_bw, val_bw} );
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
