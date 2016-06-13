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

    auto& input_mem = this_conv->input_memory(0);
    auto& output_mem = this_conv->output_memory(0);
    auto& filter_mem = this_conv->input_memory(1);
    auto& bias_mem = this_conv->input_memory(2);

    auto& input_arg  = this_conv->input_memory(0).argument;

    auto& filter_arg = this_conv->argument.input[1].primitive.as<const memory&>().argument; //convolution filter

    assert( output_size.feature[0] == filter_arg.size.feature[0] ); // memory::format oixy

    // todo remove
    if(filter_arg.format != memory::format::oiyx_f32) throw std::runtime_error("conv weights arent oiyx_f32 format");

    const int f_pos = 1; // neural::vector format is b,f,spatials. In input and output 'b' and 'f' fields are always scalars.
    namespace nd = ndimensional;
    nd::value<uint32_t> range (output_size.raw);

    // weights neural::vector is: {b}, {ofm, ifm} {spatials}
    // ofm - output feature maps
    // ifm - input feature maps
    // b = 1 always
    // (weights can't have batch so it is equall to 1)
    // Ofm and batch is cropped, ofm will be hold manually
    // Batch is included in output size
    nd::value<uint32_t> window_range_truncated ({filter_arg.size.raw.cbegin()+2, filter_arg.size.raw.cend()});
    auto calc_in_ptr     = nd::choose_calculate_ptr(input_mem);
    auto calc_out_ptr    = nd::choose_calculate_ptr(output_mem);
    auto calc_filter_ptr = nd::choose_calculate_ptr(filter_mem);
    auto calc_bias_ptr   = nd::choose_calculate_ptr(bias_mem);

    switch(padding){
        case padding::zero:
        {
            for(auto pos : range) {
                auto output = static_cast<float*>(calc_out_ptr(output_mem, pos + output_offset));
                auto bias = static_cast<float*>(calc_bias_ptr(bias_mem, {0, 0, pos[f_pos]}));
                *(output) = *bias;
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

                        auto input  = static_cast<float*> (calc_in_ptr ( input_mem, {arg_in_idx.begin(), arg_in_idx.end()} ));
                        auto filter = static_cast<float*> (calc_filter_ptr( filter_mem,
                                                     [&](){
                                                        auto vec = std::vector<uint32_t>({0, ofm-output_offset.feature[0]});
                                                        auto* win_pos_ptr = dynamic_cast<std::vector<uint32_t>*>(&win_pos);
                                                        vec.insert(vec.end(), win_pos_ptr->begin(), win_pos_ptr->end());
                                                        return vec;
                                                     }()
                                                    ));

                        auto output = static_cast<float*> (calc_out_ptr(output_mem, pos_with_modified_ofm ));

                        *output += *input * *filter;
                    }
                }
            }
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
    auto& fw_input_arg     = this_bw_conv->input_memory(1).argument;
    auto& filter_arg       = this_bw_conv->input_memory(2).argument;
    auto& bias_arg         = this_bw_conv->input_memory(3).argument; //todo bias isn't needed in bw conv. It is only used to compare its size with bias_diff. Remove?

    auto& bw_output_arg    = this_bw_conv->output_memory(0).argument;
    auto& filter_diff_arg  = this_bw_conv->output_memory(1).argument;
    auto& bias_diff_arg    = this_bw_conv->output_memory(2).argument;

    assert( 1 == bw_input_size.feature.size() );
    assert( 1 == bw_input_size.batch.size()   );

    if(bw_input_size.raw.size()   != bw_output_arg.size.raw.size())   throw std::runtime_error("Backward convolution bw_input/bw_output number of dimension does not match.");
    if(stride.raw.size()          != bw_output_arg.size.raw.size())   throw std::runtime_error("Backward convolution stride/bw_output number of dimension does not match.");
    if(bw_input_size.raw.size()   != fw_input_arg.size.raw.size())    throw std::runtime_error("Backward convolution bw_input/fw_output number of dimension does not match.");
    if(filter_arg.size.raw.size() != bw_output_arg.size.raw.size())   throw std::runtime_error("Backward convolution filter size/bw_output number of dimension does not match.");
    if(filter_arg.size.raw.size() != filter_diff_arg.size.raw.size()) throw std::runtime_error("Backward convolution weights/weights_diff number of dimension does not match.");
    if(bw_input_arg.format        != bw_output_arg.format)            throw std::runtime_error("Backward convolution bw_input/bw_output data format does not match.");
    if(bw_input_arg.format        != filter_arg.format)               throw std::runtime_error("Backward convolution bw_input/weights data format does not match.");
    if(bw_input_arg.format        != fw_input_arg.format)             throw std::runtime_error("Backward convolution bw_input/fw_output data format does not match.");
    if(bias_arg.size.raw.size()   != 3 &&
       bias_arg.size.batch[0]     != 1 &&
       bias_arg.size.feature[0]   != 1)                               throw std::runtime_error("Backward convolution biases isn't 1D vector.");
    if(bias_arg.size.raw.size()   != bias_diff_arg.size.raw.size())   throw std::runtime_error("Backward convolution bias/bias_diff number dimensions doesn't match.");
    if(bias_arg.size.spatial[0]   != bw_input_arg.size.feature[0])    throw std::runtime_error("Backward convolution biases/bw_input dimensions does not match.");
    if(bias_arg.size              != bias_diff_arg.size)              throw std::runtime_error("Backward convolution bias/bias_diff size doesn't match.");

    auto& bw_input_mem = this_bw_conv->input_memory(0);
    auto& fw_input_mem = this_bw_conv->input_memory(1);
    auto& weights_mem  = this_bw_conv->input_memory(2);

    //todo fw bias is used only for size check, is it needed?

    auto& bw_output_mem     = this_bw_conv->output_memory(0);
    auto& weights_diff_mem  = this_bw_conv->output_memory(1);
    auto& bias_diff_mem     = this_bw_conv->output_memory(2);

    //todo review conditions below
    for(size_t i = 0; i < bw_output_offset.raw.size(); ++i){
        // general formula for forward: output size = (input size - filter size) / step + 1
        if(bw_input_size.raw[i] <
            std::abs(static_cast<int32_t>(bw_output_arg.size.raw[i] - bw_output_offset.raw[i] - filter_arg.size.raw[i])) / stride.raw[i] + 1) //todo is it safe?
            if(filter_arg.size.raw[i] <= bw_input_size.raw[i])
                throw std::runtime_error("Output size of bw convolution is to small.");

        if(bw_input_arg.size.raw[i] < bw_input_size.raw[i] + bw_output_offset.raw[i])
            throw std::runtime_error("Backward convolution bw_input buffer size is to small.");

        if(bw_output_arg.size.raw[i] != fw_input_arg.size.raw[i])
            throw std::runtime_error("Sizes of BW output and FW input buffers in convolution bw must be equal.");
    }

    // initializie gradients with 0
    fill(this_bw_conv->output_memory(0), 0.0f);
    fill(this_bw_conv->output_memory(1), 0.0f);
    fill(this_bw_conv->output_memory(2), 0.0f);

    const int F_POS = 1;
    namespace nd = ndimensional;
    nd::value<uint32_t> bias_range (bias_arg.size);
    nd::value<uint32_t> range (bw_input_size); //todo in/out size?
    nd::value<uint32_t> window_range (filter_arg.size);
    auto calc_in_ptr        = nd::choose_calculate_ptr(bw_input_mem);
    auto calc_out_ptr       = nd::choose_calculate_ptr(bw_output_mem);
    auto calc_win_ptr       = nd::choose_calculate_ptr(weights_mem);
    auto calc_diff_ptr      = nd::choose_calculate_ptr(weights_diff_mem);
    auto calc_fw_input_ptr  = nd::choose_calculate_ptr(fw_input_mem);
    auto calc_bias_diff_ptr = nd::choose_calculate_ptr(bias_diff_mem);

    switch(padding){
        case padding::zero:
        {
            for(auto pos : range) {
                auto in_pos = pos + bw_input_offset;
                auto bw_input = static_cast<float*>(calc_in_ptr(bw_input_mem, in_pos));
                auto bias_diff       = static_cast<float*>(calc_bias_diff_ptr(bias_diff_mem, {0, 0, pos[F_POS]}));

                for(auto win_pos : window_range){
                    const std::vector<uint32_t> arg_out_idx = nd::value<uint32_t>(bw_output_offset) + pos*stride + win_pos;

                    if( nd::is_out_of_range(bw_output_arg.size, arg_out_idx) )
                        continue;

                    auto bw_output       = static_cast<float*>(calc_out_ptr(bw_output_mem, arg_out_idx));
                    auto weights         = static_cast<float*>(calc_win_ptr(weights_mem, win_pos));
                    auto weights_diff    = static_cast<float*>(calc_diff_ptr(weights_diff_mem, win_pos));
                    auto fw_input        = static_cast<float*>(calc_fw_input_ptr(fw_input_mem, arg_out_idx));

                    auto sensitivity = *bw_input * *weights;

                    *bw_output += sensitivity;
                    *weights_diff += *fw_input * sensitivity;
                }
                *bias_diff += *bw_input;
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
