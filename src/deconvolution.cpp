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
#include "network_impl.h"
#include "primitive_type_base.h"
#include <memory>

namespace cldnn
{
primitive_type_id deconvolution_type_id()
{
    static primitive_type_base<deconvolution, deconvolution_inst> instance;
    return &instance;
}

layout deconvolution_inst::calc_output_layout(const topology_map& topology_map, std::shared_ptr<const deconvolution> desc)
{
    auto input_desc = topology_map.at(desc->input[0])->primitive_desc;
    auto input_layout = input_desc->type->calc_output_layout(topology_map, input_desc);
    auto weight0_desc = topology_map.at(desc->weights[0])->primitive_desc;
    auto weights_layout = weight0_desc->type->calc_output_layout(topology_map, weight0_desc);
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

deconvolution_inst::typed_primitive_inst(network_impl& network, std::shared_ptr<const deconvolution> desc)
    : parent(network, desc, calc_output_layout(network.get_topology()->get_primitives(), desc))
{
    auto stride = desc->stride;
    auto output_size = output_memory().get_layout().size;

    auto input_arg = input_memory().get_layout();
    auto output_arg = output_memory().get_layout();

    if (input_arg.size.raw.size() != output_arg.size.raw.size()) throw std::runtime_error("input/output number of dimension does not match.");
    if (stride.raw.size() != output_arg.size.raw.size()) throw std::runtime_error("stride/output number of dimension does not match.");

    auto split = desc->split();
    for (decltype(split) j = 0; j < split; j++)
    {
        auto& filter_mem = weights_memory(j);
        auto& filter_arg = filter_mem.get_layout(); //deconvolution filter
        auto& bias_arg = bias_memory(j).get_layout();

        auto input_offset = desc->input_offset().transform(input_arg.size.format, 0);
        auto output_offset = desc->output_offset().transform(output_arg.size.format, 0);

        if (filter_arg.size.raw.size() != output_arg.size.raw.size() + 1) throw std::runtime_error("window_size != 5");
        if (bias_arg.size.raw.size() != 3) throw std::runtime_error("biases isn't 1D vector."); // b=1, f=1
        if (bias_arg.size.spatial[0] != output_size.feature[0] / split) throw std::runtime_error("biases/output feature maps number does not match.");
        if (desc->padding_filling_value() != 0.0f) throw std::runtime_error("unknown padding mode.");
        if (input_offset.raw.size() != input_arg.size.raw.size()) throw std::runtime_error("input offset/input number of dimension does not match.");

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
}
