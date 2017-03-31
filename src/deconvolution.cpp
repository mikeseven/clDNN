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
#include "deconvolution_inst.h"
#include "primitive_type_base.h"

namespace cldnn
{
primitive_type_id deconvolution_type_id()
{
    static primitive_type_base<deconvolution, deconvolution_inst> instance;
    return &instance;
}

layout deconvolution_inst::calc_output_layout(deconvolution_node const& node)
{
    auto desc = node.get_primitive();

    auto input_layout = node.input().get_output_layout();
    auto weights_layout = node.weights(0).get_output_layout(); //weights are stored after inputs
    auto input_offset = desc->input_offset().transform(input_layout.size.format, 0);
    auto strd = desc->stride.transform(format::yx, 0);
    auto split = desc->weights.size();

    //compute output_dim <= stride * (input_size - 1) + kernel_size + 2 * input_offset;
    auto kernel_xy = weights_layout.size.spatial;
    assert(kernel_xy.size() == 2);

    auto output_spatial_x = strd.spatial[0] * (input_layout.size.spatial[0] - 1) + kernel_xy[0] + 2 * input_offset.spatial[0];
    auto output_spatial_y = strd.spatial[1] * (input_layout.size.spatial[1] - 1) + kernel_xy[1] + 2 * input_offset.spatial[1];
    auto number_of_features = weights_layout.size.feature[0] * static_cast<int32_t>(split);

    tensor output_size(format::yxfb, {
        output_spatial_y, output_spatial_x, number_of_features, input_layout.size.batch[0] }
    );

    auto result = layout({ input_layout.data_type, output_size.transform(input_layout.size.format, 1) });
    return result;
}

deconvolution_inst::typed_primitive_inst(network_impl& network, deconvolution_node const& node)
    : parent(network, node)
{
    auto stride = argument.stride;
    auto output_size = output_memory().get_layout().size;

    auto input_inst = input_memory().get_layout();
    auto output_inst = output_memory().get_layout();

    if (input_inst.size.raw.size() != output_inst.size.raw.size())
        throw std::runtime_error("input/output number of dimension does not match.");
    if (stride.raw.size() != output_inst.size.raw.size())
        throw std::runtime_error("stride/output number of dimension does not match.");

    auto split = argument.split();
    for (decltype(split) j = 0; j < split; j++)
    {
        auto& filter_mem = weights_memory(j);
        auto& filter_inst = filter_mem.get_layout(); //deconvolution filter
        auto& bias_inst = bias_memory(j).get_layout();

        auto input_offset = argument.input_offset().transform(input_inst.size.format, 0);
        auto output_offset = argument.output_offset().transform(output_inst.size.format, 0);

        if (bias_inst.size.raw.size() != 3)
            throw std::runtime_error("biases isn't 1D vector."); // b=1, f=1

        if (bias_inst.size.spatial[0] != output_size.feature[0] / split)
            throw std::runtime_error("biases/output feature maps number does not match.");

        if (argument.padding_filling_value() != 0.0f)
            throw std::runtime_error("unknown padding mode.");

        if (input_offset.raw.size() != input_inst.size.raw.size())
            throw std::runtime_error("input offset/input number of dimension does not match.");

        assert(1 == output_size.feature.size());
        assert(1 == output_size.batch.size());
        assert(2 == filter_inst.size.feature.size());
        assert(1 == filter_inst.size.batch.size());
        assert(1 == filter_inst.size.batch[0]);

        if (output_size.feature[0] + output_offset.feature[0] > output_inst.size.feature[0]
            || (output_size.feature[0] / split) > filter_inst.size.feature[0])
            throw std::runtime_error("weights/output feature maps number does not match.");
        if ((input_inst.size.feature[0] - input_offset.feature[0]) / split < filter_inst.size.feature[1])
            throw std::runtime_error("weights/input feature maps number does not match.");
    }
}
}
