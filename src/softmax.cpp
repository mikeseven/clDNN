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

#include "softmax_inst.h"
#include "primitive_type_base.h"
#include "network_impl.h"

namespace cldnn
{
primitive_type_id softmax_type_id()
{
    static primitive_type_base<softmax, softmax_inst> instance;
    return &instance;
}

layout softmax_inst::calc_output_layout(softmax_node const& node)
{
    auto input_layout = node.input().get_output_layout();

    cldnn::layout layoutTemp = input_layout;
    if (input_layout.size.raw.size() == 4 && input_layout.size.spatial[0] == 1 && input_layout.size.spatial[1] == 1) //squeezenet spatials are 1x1
    {
        if (input_layout.size.format == format::bfyx)
            layoutTemp = cldnn::layout(input_layout.data_type, tensor(format::bx, { input_layout.size.batch[0], input_layout.size.feature[0] }));
        else
            layoutTemp = cldnn::layout(input_layout.data_type, tensor(format::xb, { input_layout.size.feature[0], input_layout.size.batch[0] }));
    }
    return layoutTemp;
}

softmax_inst::typed_primitive_inst(network_impl& network, softmax_node const& node)
    : parent(network, node)
{
    //    auto& input_offset  = arg.input_offset;
    //    auto& output_offset = arg.output_offset;
    //    auto& output_size   = arg.output_size;
    //
    //    auto& input_arg  = arg.input[0].primitive().as<const memory&>().argument;
    //    auto& output_arg = arg.output[0].as<const memory&>().argument;
    //    for (auto &x : input_offset.raw) if (x < 0) throw std::runtime_error("Softmax negative input offset.");
    //
    //    for(size_t i = 0; i < input_arg.size.raw.size(); ++i) {
    //        if( input_arg.size.raw[i] < output_size.raw[i] +  input_offset.raw[i]) throw std::runtime_error("Softmax input/output size does not match.");
    //        if(output_arg.size.raw[i] < output_size.raw[i] + output_offset.raw[i]) throw std::runtime_error("Softmax sizes too small.");
    //    }

    //auto& input_arg = network.get_topology()->get_primitives().at(desc->input()[0]);
    //if (input_arg->output_layout->size.format == cldnn::format::bfyx)
    //    if (input_arg->output_layout->size.spatial[0] != 1 || input_arg->output_layout->size.spatial[1] != 1)
    //        throw std::runtime_error("Softmax input has more than one dimension per batch");
}
}
