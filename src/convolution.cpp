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

///////////////////////////////////////////////////////////////////////////////////////////////////
#include "convolution_arg.h"
#include "network_impl.h"
#include "primitive_type_base.h"
#include <memory>

namespace cldnn
{
primitive_type_id convolution_type_id()
{
    static primitive_type_base<convolution, convolution_arg> instance;
    return &instance;
}

layout convolution_arg::calc_output_layout(const topology_map& topology_map, std::shared_ptr<const convolution> desc)
{
    auto input_desc = topology_map.at(desc->input()[0])->primitive_desc;
    auto input_layout = input_desc->type()->calc_output_layout(topology_map, input_desc);
    auto weight0_desc = topology_map.at(desc->weights[0])->primitive_desc;
    auto weights_layout = weight0_desc->type()->calc_output_layout(topology_map, weight0_desc);
    auto input_offset = desc->input_offset().transform(input_layout.size.format, 0);
    auto strd = desc->stride.transform(format::yx, 0);
    auto split = desc->weights.size();

    // compute how many outputs in rows and columns will be generate by filter. 
    // outp <= (input_size - (2*input_offset) - kernel_size)/ stride 
    auto kernel_xy = weights_layout.size.spatial;
    assert(kernel_xy.size() == 2);

    // TODO: Consider moving general parameter verification to arguments constructor.
    if (strd.spatial[0] <= 0 || strd.spatial[1] <= 0)
        throw std::invalid_argument("Stride must be positive (>= 1)");
    if (2 * input_offset.spatial[0] >= input_layout.size.spatial[0] || 2 * input_offset.spatial[1] >= input_layout.size.spatial[1])
        throw std::invalid_argument("Input offset is greater than input data range. There is no input data to process");

    // NOTE: Using most common calculation.
    //       For example - consider convolution with input=224x224, filter=7x7, offset=-3 (top/left/right/bottom), stride=2x2:
    //       Input index range is: 0-223, including offset: (-3)-226 (values outside of input index range are assumed to be 0).
    //       Output size should be: 113, becuase output index range would be: 0-112 which uses windows in input:
    //       (-3)-3; (-1)-5; 1-7; ...; 219-225; 221-227
    //       Common algorithm returns: 112, output index range: 0-111 which uses windows in input:
    //       (-3)-3; (-1)-5; 1-7; ...; 217-223; 219-225
    //
    //       The situation is even worse when we have e.g. input=230x230, filter=7x7, offset=0 (top/left/right/bottom), stride=2x2:
    //       Input index range is: 0-229, including offset: 0-229 (values outside of input index range are assumed to be 0).
    //       Output size should be: 113, becuase output index range would be: 0-112 which uses windows in input:
    //       0-6; 2-8; 4-10; ...; 222-228; 224-230
    //       Common algorithm returns: 112, output index range: 0-111 which uses windows in input:
    //       0-6; 2-8; 4-10; ...; 220-226; 222-228
    //
    //       In common calculation we dropped entire valid column and row of input data. Please, take into consideration this
    //       behavior (it can omit from calculation up to stride-1 last rows and columns).
    auto output_spatial_x = static_cast<cldnn::tensor::value_type>(
        2 * input_offset.spatial[0] < input_layout.size.spatial[0]
            ? std::max(input_layout.size.spatial[0] - 2 * input_offset.spatial[0] - kernel_xy[0], 0) / strd.spatial[0] + 1
            // ? ceil_div(std::max(input_layout.size.spatial[0] - 2 * input_offset.spatial[0] - kernel_xy[0], 0), strd.spatial[0]) + 1
            : 0);
    auto output_spatial_y = static_cast<cldnn::tensor::value_type>(
        2 * input_offset.spatial[1] < input_layout.size.spatial[1]
            ? std::max(input_layout.size.spatial[1] - 2 * input_offset.spatial[1] - kernel_xy[1], 0) / strd.spatial[1] + 1
            // ? ceil_div(std::max(input_layout.size.spatial[1] - 2 * input_offset.spatial[1] - kernel_xy[1], 0), strd.spatial[1]) + 1
            : 0);
    // get output feature map from weights. It should be the same as number of biases. Will be verifed in convolution::create()
    auto number_of_features = weights_layout.size.feature[0] * static_cast<int32_t>(split);

    tensor output_size(format::yxfb, {
                           output_spatial_y, output_spatial_x, number_of_features, input_layout.size.batch[0] }
                      );

    auto result = layout({ input_layout.data_type, output_size.transform(input_layout.size.format, 1) });
    return result;
}

convolution_arg::convolution_arg(network_impl& network, std::shared_ptr<const convolution> desc)
    : primitive_arg_base(network, desc, calc_output_layout(network.get_topology()->get_primitives(), desc))
    , _weights(network.get_primitives(desc->weights))
    , _biases(network.get_primitives(desc->bias))
{
    auto stride = desc->stride;
    auto output_size = output_memory().argument().size;

    auto input_arg = input_memory(0).get_layout();
    auto output_arg = output_memory().get_layout();

    if (input_arg.size.raw.size() != output_arg.size.raw.size()) throw std::runtime_error("input/output number of dimension does not match.");
    if (stride.raw.size() != output_arg.size.raw.size()) throw std::runtime_error("stride/output number of dimension does not match.");

    auto split = desc->split;
    for (decltype(split) j = 0; j < split; j++)
    {
        auto& filter_mem = weights_memory(j);
        auto& filter_arg = filter_mem.get_layout(); //convolution filter
        auto& bias_arg = bias_memory(j).get_layout();

        auto input_offset = desc->input_offset().transform(input_arg.size.format, 0);
        auto output_offset = desc->output_offset().transform(output_arg.size.format, 0);

        if (filter_arg.size.raw.size() != output_arg.size.raw.size() + 1) throw std::runtime_error("window_size != 5");
        if (bias_arg.size.raw.size() != 3) throw std::runtime_error("biases isn't 1D vector."); // b=1, f=1
        if (bias_arg.size.spatial[0] != output_size.feature[0] / split) throw std::runtime_error("biases/output feature maps number does not match.");
        if (desc->padding_filling_value() != 0.0f) throw std::runtime_error("unknown padding mode.");
        if (input_offset.raw.size() != input_arg.size.raw.size()) throw std::runtime_error("input offset/input number of dimension does not match.");
        //not relevant anymore
        //if (output_offset.raw.size() != input_arg.size.raw.size()) throw std::runtime_error("output offset/input number of dimension does not match.");

        //Is not relevant anymore
        //for (uint32_t i = 0; i < output_arg.size.raw.size(); i++)
        //    if (output_arg.size.raw.at(i) < output_size.raw.at(i) + output_offset.raw.at(i))
        //        throw std::runtime_error("output buffer size is too small.");

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
}

const memory& convolution_arg::weights_memory(size_t index) const
{
    return _weights[index]->output_memory();
}

const memory& convolution_arg::bias_memory(size_t index) const
{
    return _biases[index]->output_memory();
}
}
