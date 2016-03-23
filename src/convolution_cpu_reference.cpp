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

namespace neural {

convolution_cpu_reference::convolution_cpu_reference(convolution &arg)
        : is_an_implementation(neural::type_id<convolution_cpu_reference>())
        , outer(arg) {};
convolution_cpu_reference::~convolution_cpu_reference() {};
/*static*/ void convolution_cpu_reference::implementation(const void *ptr) {
        auto this_conv = static_cast<const convolution *>(ptr);

        auto& input_offset  = this_conv->argument.input_offset;
        auto& output_offset = this_conv->argument.output_offset;
        auto& output_size   = this_conv->argument.output_size;
        auto& padding       = this_conv->argument.padding;
        auto& stride        = this_conv->argument.stride;

        auto& input_arg  = this_conv->input_memory(0).argument;
        auto& output_arg = this_conv->output_memory(0).argument;

        auto& filter_arg = this_conv->argument.weight.as<const memory&>().argument; //convolution filter
        auto& bias_arg   = this_conv->argument.bias.as<const memory&>().argument;

        int f_pos = 2; //todo need type traits

        if(input_arg.size.size()  != output_arg.size.size())   throw std::runtime_error("Convolution input/output number of dimension does not match.");
        if(stride.size()          != output_arg.size.size())   throw std::runtime_error("Convolution stride/output number of dimension does not match.");
        if(input_arg.format       != memory::format::yxfb_f32) throw std::runtime_error("Convolution reference uses yxfb_f32 format.");             // only yxfb_f32 format is supported
        if(input_arg.format       != output_arg.format)        throw std::runtime_error("Convolution input/output data format does not match.");    // only yxfb_f32 format is supported
        if(input_arg.format       != filter_arg.format)        throw std::runtime_error("Convolution input/weights data format does not match.");   // only yxfb_f32 format is supported
        if(filter_arg.size.size() != output_arg.size.size())   throw std::runtime_error("Convolution window_size/output number of dimension does not match.");
        if(bias_arg.size.size()   != 1)                        throw std::runtime_error("Convolution biases isn't 1D vector.");
        if(bias_arg.size[0]       != output_size[f_pos])       throw std::runtime_error("Convolution biases/output feature maps number does not match."); // todo need type traits for index of 'z' dimension
                                                                                                                                                          // than this implementation will be format independent
        auto input  = static_cast<float*>(this_conv->input_memory(0).pointer);
        auto output = static_cast<float*>(this_conv->output_memory(0).pointer);
        auto filter = static_cast<float*>(this_conv->argument.weight.as<const memory&>().pointer);
        auto bias   = static_cast<float*>(this_conv->argument.bias.as<const memory&>().pointer);

        for(size_t i = 0; i < input_offset.size(); ++i){
            // general formula: output size = (input size - filter size) / step + 1
            if(output_size[i] <
               std::abs(static_cast<int32_t>(input_arg.size[i] - input_offset[i] - filter_arg.size[i])) / stride[i] + 1) //todo is it safe?
                if(filter_arg.size[i] <= output_size[i])
                    throw std::runtime_error("Output size of convolution is to small.");

            if(output_arg.size[i] < output_size[i] + output_offset[i])
                throw std::runtime_error("Convolution output buffer size is to small.");
        }

        namespace nd = ndimensional;
        nd::value<uint32_t> range (output_size);
        nd::value<uint32_t> window_range (filter_arg.size);
        nd::calculate_idx<uint32_t> calc_in_idx  (input_arg.size);
        nd::calculate_idx<uint32_t> calc_out_idx (output_arg.size);
        nd::calculate_idx<uint32_t> calc_win_idx (filter_arg.size);
        switch(padding){
            case padding::zero:
                for(auto pos : range) {
                    float acc = 0;
                    auto out_idx = calc_out_idx(pos + output_offset);

                    for(auto win_pos : window_range){
                        const std::vector<int32_t> arg_in_idx = nd::value<int32_t>(input_offset) + pos*stride + win_pos;

                        if( calc_in_idx.is_out_of_range(arg_in_idx) )
                            continue;

                        auto in_idx  = calc_in_idx (arg_in_idx);
                        auto win_idx = calc_win_idx(win_pos);
                        acc += input[in_idx] * filter[win_idx];
                    }
                    output[out_idx] = acc + bias[ pos[f_pos] ]; // todo need type traits for index of 'f' dimension
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
/*static*/ void convolution_backward_cpu_reference::implementation(const void *ptr) { //todo tests
    auto this_bw_conv = static_cast<const convolution_backward *>(ptr);

    auto& bw_input_size    = this_bw_conv->argument.input_size;  // todo output or input?
    auto& bw_input_offset  = this_bw_conv->argument.input_offset;
    auto& bw_output_offset = this_bw_conv->argument.output_offset;
    auto& stride           = this_bw_conv->argument.stride;
    auto& padding          = this_bw_conv->argument.padding;

    auto& bw_input_arg     = this_bw_conv->input_memory(0).argument;
    auto& fw_input_arg     = this_bw_conv->input_memory(1).argument;
    auto& filter_arg       = this_bw_conv->input_memory(2).argument;
    auto& bias_arg         = this_bw_conv->input_memory(3).argument; //todo bias isn't needed in bw conv. It is only used to compare its size with bias_diff. Remove?

    auto& bw_output_arg    = this_bw_conv->output_memory(0).argument;
    auto& filter_diff_arg  = this_bw_conv->output_memory(1).argument;
    auto& bias_diff_arg    = this_bw_conv->output_memory(2).argument;

    int f_pos = 2; //todo need type traits

    if(bw_input_size.size()   != bw_output_arg.size.size())   throw std::runtime_error("Backward convolution bw_input/bw_output number of dimension does not match.");
    if(stride.size()          != bw_output_arg.size.size())   throw std::runtime_error("Backward convolution stride/bw_output number of dimension does not match.");
    if(bw_input_size.size()   != fw_input_arg.size.size())    throw std::runtime_error("Backward convolution bw_input/fw_output number of dimension does not match.");
    if(filter_arg.size.size() != bw_output_arg.size.size())   throw std::runtime_error("Backward convolution filter size/bw_output number of dimension does not match.");
    if(filter_arg.size.size() != filter_diff_arg.size.size()) throw std::runtime_error("Backward convolution weights/weights_diff number of dimension does not match.");
    if(bw_input_arg.format    != bw_output_arg.format)        throw std::runtime_error("Backward convolution bw_input/bw_output data format does not match.");
    if(bw_input_arg.format    != filter_arg.format)           throw std::runtime_error("Backward convolution bw_input/weights data format does not match.");
    if(bw_input_arg.format    != fw_input_arg.format)         throw std::runtime_error("Backward convolution bw_input/fw_output data format does not match.");
    if(bias_arg.size.size()   != 1)                           throw std::runtime_error("Backward convolution biases isn't 1D vector.");
    if(bias_arg.size.size()   != bias_diff_arg.size.size())   throw std::runtime_error("Backward convolution bias/bias_diff number dimensions doesn't match.");
    if(bias_arg.size[0]       != bw_output_arg.size[f_pos])   throw std::runtime_error("Backward convolution biases/bw_output feature maps number does not match."); // todo need type traits for index of 'z' dimension
    if(bias_arg.size[0]       != bias_diff_arg.size[0])       throw std::runtime_error("Backward convolution bias/bias_diff size doesn't match.");

    auto bw_input     = static_cast<float*>(this_bw_conv->input_memory(0).pointer);
    auto fw_input     = static_cast<float*>(this_bw_conv->input_memory(1).pointer);
    auto weights      = static_cast<float*>(this_bw_conv->input_memory(2).pointer);
    auto bias         = static_cast<float*>(this_bw_conv->input_memory(3).pointer);

    auto bw_output    = static_cast<float*>(this_bw_conv->output_memory(0).pointer);
    auto weights_diff = static_cast<float*>(this_bw_conv->output_memory(1).pointer);
    auto bias_diff    = static_cast<float*>(this_bw_conv->output_memory(2).pointer);

    for(size_t i = 0; i < bw_output_offset.size(); ++i){
        // general formula for forward: output size = (input size - filter size) / step + 1
        if(bw_input_size[i] <
            std::abs(static_cast<int32_t>(bw_output_arg.size[i] - bw_output_offset[i] - filter_arg.size[i])) / stride[i] + 1) //todo is it safe?
            if(filter_arg.size[i] <= bw_input_size[i])
                throw std::runtime_error("Output size of bw convolution is to small.");

        if(bw_input_arg.size[i] < bw_input_size[i] + bw_output_offset[i])
            throw std::runtime_error("Backward convolution bw_input buffer size is to small.");

        if(bw_output_arg.size[i] != fw_input_arg.size[i])
            throw std::runtime_error("Sizes of BW output and FW input buffers in convolution bw must be equal.");
    }
        // initializie gradients with 0
    this_bw_conv->output_memory(0).fill(0.0f);
    this_bw_conv->output_memory(1).fill(0.0f);
    this_bw_conv->output_memory(2).fill(0.0f);

    namespace nd = ndimensional;
    nd::value<uint32_t> bias_range (bias_arg.size);
    nd::value<uint32_t> range (bw_input_size); //todo in/out size?
    nd::value<uint32_t> window_range (filter_arg.size);
    nd::calculate_idx<uint32_t> calc_bias_idx(bias_arg.size);
    nd::calculate_idx<uint32_t> calc_in_idx  (bw_input_arg.size);
    nd::calculate_idx<uint32_t> calc_out_idx (bw_output_arg.size);
    nd::calculate_idx<uint32_t> calc_win_idx (filter_arg.size);

    switch(padding){
        case padding::zero:
        {
            for(auto pos : range) {
                auto in_idx = calc_in_idx(pos + bw_input_offset);

                for(auto win_pos : window_range){
                    const std::vector<uint32_t> arg_out_idx = nd::value<uint32_t>(bw_output_offset) + pos*stride + win_pos;

                    if( calc_out_idx.is_out_of_range(arg_out_idx) )
                        continue;

                    auto out_idx = calc_out_idx(arg_out_idx);
                    auto win_idx = calc_win_idx(win_pos);

                    auto sensitivity = bw_input[in_idx] * weights[win_idx];

                    bw_output[out_idx] += sensitivity;
                    weights_diff[win_idx] += fw_input[out_idx] * sensitivity;
                }
                bias_diff[ pos[f_pos] ] += bw_input[in_idx];
            }
            break;
        }
        default:
            throw std::runtime_error("Unknown padding mode in backward convolution.");
    }

}

attach_conv_cpu_ref::attach_conv_cpu_ref() {
    auto key = std::make_tuple(engine::reference, memory::format::yxfb_f32, memory::format::yxfb_f32);
    auto val_fw = convolution_cpu_reference::create;
    auto val_bw = convolution_backward_cpu_reference::create;

    conv_fw_implementation_map.insert( {key, val_fw} ); //todo keys should be different
    conv_bw_implementation_map.insert( {key, val_bw} );
}
}
