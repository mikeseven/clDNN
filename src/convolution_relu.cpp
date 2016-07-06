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
#include "convolution_relu.h"
#include <sstream>

namespace neural {

convolution_relu::arguments::arguments( neural::engine::type     eng,
                                   primitive                out,
                                   neural::vector<uint32_t> out_off,
                                   neural::vector<uint32_t> out_siz,
                                   std::vector<primitive_at>in,
                                   neural::vector<int32_t>  in_off,
                                   neural::vector<uint32_t> strd,
                                   neural::padding::type    padd)
    : engine(eng)
    , output({out})
    , output_offset(out_off)
    , output_size(out_siz)
    , input(in)
    , input_offset(in_off)
    , stride(strd)
    , padding(padd)
    , negative_slope(0) {};

convolution_relu::arguments::arguments( neural::engine::type     eng,
                                   primitive                out,
                                   std::vector<primitive_at>in,
                                   neural::vector<uint32_t> strd,
                                   neural::padding::type    padd,
                                   float nslp)
    : engine(eng)
    , output({out})
    , output_offset(out.as<const memory&>().argument.size.batch.size(), out.as<const memory&>().argument.size.spatial.size(), out.as<const memory&>().argument.size.feature.size())
    , output_size(out.as<const memory&>().argument.size)
    , input(in)
    , input_offset(in[0].primitive.as<const memory&>().argument.size.batch.size(), in[0].primitive.as<const memory&>().argument.size.spatial.size(), in[0].primitive.as<const memory&>().argument.size.feature.size())
    , stride(strd)
    , padding(padd)
    , negative_slope(nslp) {};

convolution_relu::arguments::arguments(neural::engine::type     eng,
    memory::format::type    out_fmt,
    std::vector<primitive_at>in,
    neural::vector<int32_t>  in_off,
    neural::vector<uint32_t> strd,
    neural::padding::type    padd,
    float nslp)
    : engine(eng)
    , input_offset(in_off)
    , stride(strd)
    , padding(padd)
    , negative_slope(nslp)
{
    // if input is previouse layer, not memory primitive need to set input to output memory of this primitive
    auto input_mem = in[0].primitive.id() == type_id<const memory>()->id ? in[0].primitive : in[0].primitive.output[0];
    if (in.size() != 3) throw std::runtime_error("input size mismatch");
    if (in[0].primitive.id() != type_id<const memory>()->id) {
        input = { in[0].primitive.output[0], in[1], in[2] };
    }
    else {
        input = { in[0], in[1], in[2] };;
    }
    // compute how many outputs in rows and columns will be generate by filter. 
    // outp <= (input_size - (2*input_offset) - kernel_size)/ stride 
    auto kernel_xy = in[1].primitive.as<const memory&>().argument.size.spatial;
    auto output_spatial_x = (input_mem.as<const memory&>().argument.size.spatial[0] - (2 * input_offset.spatial[0]) - kernel_xy[0]) / strd.spatial[0] + 1;
    auto output_spatial_y = (input_mem.as<const memory&>().argument.size.spatial[1] - (2 * input_offset.spatial[1]) - kernel_xy[1]) / strd.spatial[1] + 1;
    auto input_x = input_mem.as<const memory&>().argument;
    // get output feature map from weights. It should be the same as number of biases. Will be verifed in convolution::create()
    auto ofm = in[1].primitive.as<const memory&>().argument;
    auto number_of_batches = ofm.size.raw[1];
    output_size = {
        input_mem.as<const memory&>().argument.size.batch[0],
        { output_spatial_x, output_spatial_y },
        number_of_batches
    };
    output = { memory::allocate({ eng, out_fmt,output_size }) };
    output_offset = {
        output[0].as<const memory&>().argument.size.batch.size(),
        output[0].as<const memory&>().argument.size.spatial.size(),
        output[0].as<const memory&>().argument.size.feature.size()
    };
};

convolution_relu::arguments::arguments(neural::engine::type     eng,
    memory::format::type    out_fmt,
    std::vector<primitive_at>in,
    neural::vector<uint32_t> strd,
    neural::padding::type    padd,
    float nslp)
    : engine(eng)
    , input(in)
    , input_offset(in[0].primitive.as<const memory&>().argument.size.batch.size(), in[0].primitive.as<const memory&>().argument.size.spatial.size(), in[0].primitive.as<const memory&>().argument.size.feature.size())
    , stride(strd)
    , padding(padd)
    , negative_slope(nslp)
{
    // if input is previouse layer, not memory primitive need to set input to output memory of this primitive
    auto input_mem = in[0].primitive.id() == type_id<const memory>()->id ? in[0].primitive : in[0].primitive.output[0];
    if (in.size() != 3) throw std::runtime_error("input size mismatch");
    if (in[0].primitive.id() != type_id<const memory>()->id) {
        input = { in[0].primitive.output[0], in[1], in[2] };
    }
    else {
        input = { in[0], in[1], in[2] };;
    }
    // compute how many outputs in rows and columns will be generate by filter. 
    // outp <= (input_size - (2*input_offset) - kernel_size)/ stride 
    auto kernel_xy = in[1].primitive.as<const memory&>().argument.size.spatial;
    auto output_spatial_x = (input_mem.as<const memory&>().argument.size.spatial[0] - (2 * input_offset.spatial[0]) - kernel_xy[0]) / strd.spatial[0] + 1;
    auto output_spatial_y = (input_mem.as<const memory&>().argument.size.spatial[1] - (2 * input_offset.spatial[1]) - kernel_xy[1]) / strd.spatial[1] + 1;
    auto input_x = input_mem.as<const memory&>().argument;
    // get output feature map from weights. It should be the same as number of biases. Will be verifed in convolution::create()
    auto ofm = in[1].primitive.as<const memory&>().argument;
    auto number_of_batches = ofm.size.raw[1];
    output_size = {
        input_mem.as<const memory&>().argument.size.batch[0],
        { output_spatial_x, output_spatial_y },
        number_of_batches
    };
    output = { memory::allocate({ eng, out_fmt,output_size }) };
    output_offset = {
        output[0].as<const memory&>().argument.size.batch.size(),
        output[0].as<const memory&>().argument.size.spatial.size(),
        output[0].as<const memory&>().argument.size.feature.size()
    };
};

convolution_relu::arguments::arguments( neural::engine::type     eng,
                                   primitive                out,
                                   std::vector<primitive_at>in,
                                   neural::padding::type    padd,
                                   float nslp)
    : engine(eng)
    , output({out})
    , output_offset(out.as<const memory&>().argument.size.batch.size(), out.as<const memory&>().argument.size.spatial.size(), out.as<const memory&>().argument.size.feature.size())
    , output_size(out.as<const memory&>().argument.size)
    , input(in)
    , input_offset(in[0].primitive.as<const memory&>().argument.size.batch.size(), in[0].primitive.as<const memory&>().argument.size.spatial.size(), in[0].primitive.as<const memory&>().argument.size.feature.size())
    , stride(1u, std::vector<uint32_t>(in[0].primitive.as<const memory&>().argument.size.spatial.size(), 1u), 1u)
    , padding(padd)
    , negative_slope(nslp) {};

// creates primitive with convolution_relu implementation that supports provided arguments
primitive convolution_relu::create(convolution_relu::arguments arg) {
    auto& output_size = arg.output_size;
    auto& stride = arg.stride;

    auto& input_arg = arg.input[0].primitive.as<const memory&>().argument;
    auto& output_arg = arg.output[0].as<const memory&>().argument;

    auto& filter_arg = arg.input[1].primitive.as<const memory&>().argument; //convolution filter
    auto& bias_arg = arg.input[2].primitive.as<const memory&>().argument;

    auto& input_offset = arg.input_offset;
    auto& output_offset = arg.output_offset;

    if(input_arg.size.raw.size()  != output_arg.size.raw.size())  throw std::runtime_error("Convolution input/output number of dimension does not match.");
    if(stride.raw.size()          != output_arg.size.raw.size())  throw std::runtime_error("Convolution stride/output number of dimension does not match.");
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
    std::unique_ptr<convolution_relu> result(new convolution_relu(arg));

    // create implementation for non-lazy evaluation
    if(0 == (arg.engine & engine::lazy)) {
        // lookup in database; throw if not found
        conv_relu_fw_key key = std::make_tuple(arg.engine, result-> input_memory(0).argument.format, result->output_memory(0).argument.format);
        auto it = conv_relu_fw_implementation_map.find(key);
        if(it==std::end(conv_relu_fw_implementation_map)) throw std::runtime_error("Not yet implemented.");

        // create implementation & attach it to result
        auto implementation = it->second(*result);
        result->_private.reset(implementation);
        result->_work = implementation->work();
    }

    // release RAII wrapper, return naked pointer
    return result.release();
}
}
