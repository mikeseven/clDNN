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
#include "implementation_map.h"
#include <sstream>

namespace neural {

convolution::arguments::arguments(neural::engine::type     eng,
    primitive                out,
    neural::vector<uint32_t> out_off,
    neural::vector<uint32_t> out_siz,
    std::vector<primitive_at>in,
    neural::vector<int32_t>  in_off,
    neural::vector<uint32_t> strd,
    neural::padding::type    padd,
    size_t                   splt,
    bool                     use_relu,
    float                    negative_slope)
    : engine(eng)
    , output({ out })
    , output_offset(out_off)
    , output_size(out_siz)
    , input(in)
    , input_offset(in_off)
    , stride(strd)
    , padding(padd)
    , split(splt)
    , use_relu(use_relu)
    , negative_slope(negative_slope){}

convolution::arguments::arguments(neural::engine::type     eng,
    primitive                out,
    std::vector<primitive_at>in,
    neural::vector<uint32_t> strd,
    neural::padding::type    padd,
    size_t                   splt,
    bool                     use_relu,
    float                    negative_slope)
    : engine(eng)
    , output({ out })
    , output_offset(out.as<const memory&>().argument.size.batch.size(),
        out.as<const memory&>().argument.size.spatial.size(),
        out.as<const memory&>().argument.size.feature.size())
    , output_size(out.as<const memory&>().argument.size)
    , input(in)
    , input_offset(in[0].primitive().as<const memory&>().argument.size.batch.size(),
        in[0].primitive().as<const memory&>().argument.size.spatial.size(),
        in[0].primitive().as<const memory&>().argument.size.feature.size())
    , stride(strd)
    , padding(padd)
    , split(splt)
    , use_relu(use_relu)
    , negative_slope(negative_slope){}

convolution::arguments::arguments(neural::engine::type     eng,
    memory::format::type     out_fmt,
    std::vector<primitive_at>in,
    neural::vector<int32_t>  in_off,
    neural::vector<uint32_t> strd,
    neural::padding::type    padd,
    size_t                   splt,
    bool                     use_relu,
    float                    negative_slope)
    : engine(eng)
    , input_offset(in_off)
    , stride(strd)
    , padding(padd)
    , split(splt)
    , use_relu(use_relu)
    , negative_slope(negative_slope)
{
    const size_t input_expected_size = split * 2 + 1;

    // if input is previouse layer, not memory primitive need to set input to output memory of this primitive
    if (in.size() != input_expected_size) throw std::runtime_error("input size mismatch");
    input.reserve(input_expected_size);
    input.push_back(
        in[0].primitive().id() != type_id<const memory>()->id ? in[0].primitive().output[0] : in[0]
    );
    for (size_t i = 1; i < in.size(); i++)
        input.push_back(in[i]);

    auto &input_mem = input[0].primitive().as<const memory&>();

    // compute how many outputs in rows and columns will be generate by filter. 
    // outp <= (input_size - (2*input_offset) - kernel_size)/ stride 
    auto kernel_xy = in[1].primitive().as<const memory&>().argument.size.spatial;
    auto output_spatial_x = (input_mem.argument.size.spatial[0] - (2 * input_offset.spatial[0]) - kernel_xy[0]) / strd.spatial[0] + 1;
    auto output_spatial_y = (input_mem.argument.size.spatial[1] - (2 * input_offset.spatial[1]) - kernel_xy[1]) / strd.spatial[1] + 1;
    auto input_x = input_mem.argument;
    // get output feature map from weights. It should be the same as number of biases. Will be verifed in convolution::create()
    auto ofm = in[1].primitive().as<const memory&>().argument;
    auto number_of_batches = ofm.size.raw[1] * static_cast<uint32_t>(split);
    output_size = {
        input_mem.argument.size.batch[0],
        { output_spatial_x, output_spatial_y },
        number_of_batches
    };
    output = { memory::allocate({ eng, out_fmt,output_size }) };
    output_offset = {
        output[0].as<const memory&>().argument.size.batch.size(),
        output[0].as<const memory&>().argument.size.spatial.size(),
        output[0].as<const memory&>().argument.size.feature.size()
    };
}

convolution::arguments::arguments(neural::engine::type     eng,
    memory::format::type     out_fmt,
    std::vector<primitive_at>in,
    neural::vector<uint32_t> strd,
    neural::padding::type    padd,
    size_t                   splt,
    bool                     use_relu,
    float                    negative_slope)
    : arguments(eng,
        out_fmt,
        in,
        {
            in[0].primitive().as<const memory&>().argument.size.batch.size(),
            in[0].primitive().as<const memory&>().argument.size.spatial.size(),
            in[0].primitive().as<const memory&>().argument.size.feature.size()
        },
        strd,
        padd,
        splt,
        use_relu,
        negative_slope) {}

convolution::arguments::arguments(neural::engine::type     eng,
    primitive                out,
    std::vector<primitive_at>in,
    neural::padding::type    padd,
    size_t                   splt,
    bool                     use_relu,
    float                    negative_slope)
    : engine(eng)
    , output({ out })
    , output_offset(out.as<const memory&>().argument.size.batch.size(),
        out.as<const memory&>().argument.size.spatial.size(),
        out.as<const memory&>().argument.size.feature.size())
    , output_size(out.as<const memory&>().argument.size)
    , input(in)
    , input_offset(in[0].primitive().as<const memory&>().argument.size.batch.size(),
        in[0].primitive().as<const memory&>().argument.size.spatial.size(),
        in[0].primitive().as<const memory&>().argument.size.feature.size())
    , stride(1u, std::vector<uint32_t>(in[0].primitive().as<const memory&>().argument.size.spatial.size(), 1u), 1u)
    , padding(padd)
    , split(splt)
    , use_relu(use_relu)
    , negative_slope(negative_slope) {}

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
    , padding(padd) {}

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
    , padding(padd) {}

// creates primitive with convolution implementation that supports provided arguments
primitive convolution::create(convolution::arguments arg) {
    auto& output_size = arg.output_size;
    auto& stride = arg.stride;

    auto& input_arg = arg.input[0].primitive().as<const memory&>().argument;
    auto& output_arg = arg.output[0].as<const memory&>().argument;

    if (input_arg.size.raw.size() != output_arg.size.raw.size())  throw std::runtime_error("input/output number of dimension does not match.");
    if (stride.raw.size() != output_arg.size.raw.size())          throw std::runtime_error("stride/output number of dimension does not match.");

    const size_t split = arg.split;
    for (size_t j = 0; j < split; j++)
    {
        auto& filter_arg = arg.input[j * 2 + 1].primitive().as<const memory&>().argument; //convolution filter
        auto& bias_arg = arg.input[j * 2 + 2].primitive().as<const memory&>().argument;

        auto& input_offset = arg.input_offset;
        auto& output_offset = arg.output_offset;

        if (filter_arg.size.raw.size() != output_arg.size.raw.size() + 1)   throw std::runtime_error("window_size != 5");
        if (bias_arg.size.raw.size() != 3)                                  throw std::runtime_error("biases isn't 1D vector."); // b=1, f=1
        if (bias_arg.size.spatial[0] != output_size.feature[0] / split)     throw std::runtime_error("biases/output feature maps number does not match.");
        if (arg.padding != padding::zero)                                   throw std::runtime_error("unknown padding mode.");
        if (input_offset.raw.size() != input_arg.size.raw.size())           throw std::runtime_error("input offset/input number of dimension does not match.");
        if (output_offset.raw.size() != input_arg.size.raw.size())          throw std::runtime_error("output offset/input number of dimension does not match.");

        for (uint32_t i = 0; i < output_arg.size.raw.size(); i++)
            if (output_arg.size.raw.at(i) < output_size.raw.at(i) + output_offset.raw.at(i))
                throw std::runtime_error("output buffer size is too small.");

        assert(1 == output_size.feature.size());
        assert(1 == output_size.batch.size());
        assert(2 == filter_arg.size.feature.size());
        assert(1 == filter_arg.size.batch.size());
        assert(1 == filter_arg.size.batch[0]);

        if (output_size.feature[0] + output_offset.feature[0] > output_arg.size.feature[0]
            || (output_size.feature[0] / split) > filter_arg.size.feature[0])
            throw std::runtime_error("weights/output feature maps number does not match.");
        if ((input_arg.size.feature[0] - input_offset.feature[0]) / split < filter_arg.size.feature[1])
            throw std::runtime_error("weights/input feature maps number does not match.");
    }

    return is_a_primitive::create<convolution>(arg);
}

primitive convolution_backward::create(convolution_backward::arguments arg) {
    auto& bw_input_size = arg.input_size;  // todo output or input?
    auto& bw_input_offset = arg.input_offset;

    assert(1 == bw_input_size.feature.size());
    assert(1 == bw_input_size.batch.size());

    auto& bw_input_arg = arg.input[0].primitive().as<const memory&>().argument;
    auto& fw_input_arg = arg.input[1].primitive().as<const memory&>().argument;
    auto& filter_arg = arg.input[2].primitive().as<const memory&>().argument;
    auto& bias_arg = arg.input[3].primitive().as<const memory&>().argument;

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
    return is_a_primitive::create<convolution_backward>(arg);
}
}
