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
#include "convolution.h"
#include <sstream>

namespace neural {

convolution::arguments::arguments( neural::engine::type     eng,
                                   primitive                out,
                                   neural::vector<uint32_t> out_off,
                                   neural::vector<uint32_t> out_siz,
                                   primitive                in,
                                   neural::vector<int32_t>  in_off,
                                   neural::vector<uint32_t> strd,
                                   primitive                weights,
                                   primitive                biases,
                                   neural::padding::type    padd)
    : engine(eng)
    , output({out})
    , output_offset(out_off)
    , output_size(out_siz)
    , input({in})
    , input_offset(in_off)
    , stride(strd)
    , weight(weights)
    , bias(biases)
    , padding(padd) {};

convolution::arguments::arguments( neural::engine::type     eng,
                                   primitive                out,
                                   primitive                in,
                                   neural::vector<uint32_t> strd,
                                   primitive                weights,
                                   primitive                biases,
                                   neural::padding::type    padd)
    : engine(eng)
    , output({out})
    , output_offset(out.as<const memory&>().argument.size.batch.size(), out.as<const memory&>().argument.size.spatial.size(), out.as<const memory&>().argument.size.feature.size())
    , output_size(out.as<const memory&>().argument.size)
    , input({in})
    , input_offset(in.as<const memory&>().argument.size.batch.size(), in.as<const memory&>().argument.size.spatial.size(), in.as<const memory&>().argument.size.feature.size())
    , stride(strd)
    , weight(weights)
    , bias(biases)
    , padding(padd) {};

convolution::arguments::arguments( neural::engine::type     eng,
                                   primitive                out,
                                   primitive                in,
                                   primitive                weights,
                                   primitive                biases,
                                   neural::padding::type    padd)
    : engine(eng)
    , output({out})
    , output_offset(out.as<const memory&>().argument.size.batch.size(), out.as<const memory&>().argument.size.spatial.size(), out.as<const memory&>().argument.size.feature.size())
    , output_size(out.as<const memory&>().argument.size)
    , input({in})
    , input_offset(in.as<const memory&>().argument.size.batch.size(), in.as<const memory&>().argument.size.spatial.size(), in.as<const memory&>().argument.size.feature.size())
    , stride(1u, std::vector<uint32_t>(in.as<const memory&>().argument.size.spatial.size(), 1u), 1u)
    , weight(weights)
    , bias(biases)
    , padding(padd) {};

convolution_backward::arguments::arguments( neural::engine::type     eng,
                                            std::vector<primitive>   out,
                                            neural::vector<uint32_t> out_off,
                                            neural::vector<uint32_t> in_siz,
                                            std::vector<primitive>   in,
                                            neural::vector<int32_t>  in_off,
                                            neural::vector<uint32_t> strd,
                                            neural::padding::type    padd)
    : engine(eng)
    , output(out)
    , output_offset(out_off)
    , input_size(in_siz)
    , input(in.cbegin(), in.cend())
    , input_offset(in_off)
    , stride(strd)
    , padding(padd) {};

convolution_backward::arguments::arguments( neural::engine::type     eng,
                                            std::vector<primitive>   out,
                                            std::vector<primitive>   in,
                                            neural::vector<uint32_t> strd,
                                            neural::padding::type    padd)
    : engine(eng)
    , output(out)
    , output_offset(out[0].as<const memory&>().argument.size.batch.size(), out[0].as<const memory&>().argument.size.spatial.size(), out[0].as<const memory&>().argument.size.feature.size())
    , input_size(in[0].as<const memory&>().argument.size)
    , input(in.cbegin(), in.cend())
    , input_offset(in[0].as<const memory&>().argument.size.batch.size(), in[0].as<const memory&>().argument.size.spatial.size(), in[0].as<const memory&>().argument.size.feature.size())
    , stride(strd)
    , padding(padd) {};

// creates primitive with convolution implementation that supports provided arguments
primitive convolution::create(convolution::arguments arg) {
    auto& output_size = arg.output_size;
    auto& stride = arg.stride;

    auto& input_arg = arg.input[0].primitive.as<const memory&>().argument; //todo tmp solution
    auto& output_arg = arg.output[0].as<const memory&>().argument;

    auto& filter_arg = arg.weight.as<const memory&>().argument; //convolution filter
    auto& bias_arg = arg.bias.as<const memory&>().argument;

    auto& input_offset = arg.input_offset;
    auto& output_offset = arg.output_offset;

    if(input_arg.size.raw.size()  != output_arg.size.raw.size())  throw std::runtime_error("Convolution input/output number of dimension does not match.");
    if(stride.raw.size()          != output_arg.size.raw.size())  throw std::runtime_error("Convolution stride/output number of dimension does not match.");
    if(input_arg.format           != memory::format::yxfb_f32)    throw std::runtime_error("Convolution reference uses yxfb_f32 format.");             // only yxfb_f32 format is supported
    if(input_arg.format           != output_arg.format)           throw std::runtime_error("Convolution input/output data format does not match.");    // only yxfb_f32 format is supported
    if(filter_arg.size.raw.size() != output_arg.size.raw.size()+1)throw std::runtime_error("Convolution window_size != 5");
    if(bias_arg.size.raw.size()   != 3)                           throw std::runtime_error("Convolution biases isn't 1D vector."); // b=1, f=1
    if(bias_arg.size.spatial[0]   != output_size.feature[0])      throw std::runtime_error("Convolution biases/output feature maps number does not match.");
    if(arg.padding                != padding::zero)               throw std::runtime_error("Unknown padding mode in convolution.");
    if(input_offset.raw.size()    != input_arg.size.raw.size())   throw std::runtime_error("Convolution input offset/input number of dimension does not match.");
    if(output_offset.raw.size()   != input_arg.size.raw.size())   throw std::runtime_error("Convolution output offset/input number of dimension does not match.");

    for (uint32_t i = 0; i < output_arg.size.raw.size(); i++)
        if (output_arg.size.raw.at(i) < output_size.raw.at(i) + output_offset.raw.at(i))
            throw std::runtime_error("Convolution output buffer size is too small.");

    assert( 1 == output_size.feature.size() );
    assert( 1 == output_size.batch.size() );
    assert( 2 == filter_arg.size.feature.size());
    assert( 1 == filter_arg.size.batch.size() );
    assert( 1 == filter_arg.size.batch[0] );

    if(output_size.feature[0] + output_offset.feature[0] > output_arg.size.feature[0]
        || output_size.feature[0] > filter_arg.size.feature[0])
        throw std::runtime_error("Convolution weights/output feature maps number does not match.");
    if(input_arg.size.feature[0] - input_offset.feature[0] < filter_arg.size.feature[1])
        throw std::runtime_error("Convolution weights/input feature maps number does not match.");

    // wrap relu into RAII wrapper
    std::unique_ptr<convolution> result(new convolution(arg));

    // lookup in database; throw if not found
    conv_fw_key key = std::make_tuple(arg.engine, result-> input_memory(0).argument.format, result->output_memory(0).argument.format);
    auto it = conv_fw_implementation_map.find(key);
    if(it==std::end(conv_fw_implementation_map)) throw std::runtime_error("Not yet implemented.");

    // create implementation & attach it to result
    auto implementation = it->second(*result);
    result->_private.reset(implementation);
    result->_work = implementation->work();

    // release RAII wrapper, return naked pointer
    return result.release();
}
primitive convolution_backward::create(convolution_backward::arguments arg) {
    auto& bw_input_size = arg.input_size;  // todo output or input?
    auto& bw_input_offset = arg.input_offset;

    assert(1 == bw_input_size.feature.size());
    assert(1 == bw_input_size.batch.size());

    auto& bw_input_arg = arg.input[0].primitive.as<const memory&>().argument;
    auto& fw_input_arg = arg.input[1].primitive.as<const memory&>().argument;
    auto& filter_arg = arg.input[2].primitive.as<const memory&>().argument;
    auto& bias_arg = arg.input[3].primitive.as<const memory&>().argument;

    auto& bw_output_arg = arg.output[0].as<const memory&>().argument;
    auto& filter_diff_arg = arg.output[1].as<const memory&>().argument;
    auto& bias_diff_arg = arg.output[2].as<const memory&>().argument;

    auto& stride = arg.stride;

    if (bw_input_offset.raw.size() != bw_output_arg.size.raw.size())    throw std::runtime_error("Backward convolution bw_input_offset/bw_output number of dimension does not match.");
    if (bw_input_size.raw.size() != bw_output_arg.size.raw.size())      throw std::runtime_error("Backward convolution bw_input/bw_output number of dimension does not match.");
    if (stride.raw.size() != bw_output_arg.size.raw.size())             throw std::runtime_error("Backward convolution stride/bw_output number of dimension does not match.");
    if (bw_input_size.raw.size() != fw_input_arg.size.raw.size())       throw std::runtime_error("Backward convolution bw_input/fw_output number of dimension does not match.");
    if (filter_arg.size.raw.size() != bw_output_arg.size.raw.size())    throw std::runtime_error("Backward convolution filter size/bw_output number of dimension does not match.");
    if (filter_arg.size.raw.size() != filter_diff_arg.size.raw.size())  throw std::runtime_error("Backward convolution weights/weights_diff number of dimension does not match.");
    if (bw_input_arg.format != bw_output_arg.format)                    throw std::runtime_error("Backward convolution bw_input/bw_output data format does not match.");
    if (bw_input_arg.format != filter_arg.format)                       throw std::runtime_error("Backward convolution bw_input/weights data format does not match.");
    if (bw_input_arg.format != fw_input_arg.format)                     throw std::runtime_error("Backward convolution bw_input/fw_output data format does not match.");
    if (bias_arg.size.raw.size() != 3 &&
        bias_arg.size.batch[0] != 1 &&
        bias_arg.size.feature[0] != 1)                                  throw std::runtime_error("Backward convolution biases isn't 1D vector.");
    if (bias_arg.size.raw.size() != bias_diff_arg.size.raw.size())      throw std::runtime_error("Backward convolution bias/bias_diff number dimensions doesn't match.");
    if (bias_arg.size.spatial[0] != bw_input_arg.size.feature[0])       throw std::runtime_error("Backward convolution biases/bw_input dimensions does not match.");
    if (bias_arg.size != bias_diff_arg.size)                            throw std::runtime_error("Backward convolution bias/bias_diff size doesn't match.");

    auto& bw_output_offset = arg.output_offset;

    for (size_t i = 0; i < bw_output_offset.raw.size(); ++i) {
        if (bw_input_arg.size.raw[i] < bw_input_size.raw[i] + bw_output_offset.raw[i])
            throw std::runtime_error("Backward convolution bw_input buffer size is too small.");

        if (bw_output_arg.size.raw[i] != fw_input_arg.size.raw[i])
            throw std::runtime_error("Sizes of BW output and FW input buffers in convolution bw must be equal.");
    }

    // wrap relu into RAII wrapper
    std::unique_ptr<convolution_backward> result(new convolution_backward(arg));

    // lookup in database; throw if not found
    conv_bw_key key = std::make_tuple(arg.engine, result-> input_memory(0).argument.format, result->output_memory(0).argument.format);
    auto it = conv_bw_implementation_map.find(key);
    if(it==std::end(conv_bw_implementation_map)) throw std::runtime_error("Not yet implemented.");

    // create implementation & attach it to result
    auto implementation = it->second(*result);
    result->_private.reset(implementation);
    result->_work = implementation->work();

    // release RAII wrapper, return naked pointer
    return result.release();
}
}
