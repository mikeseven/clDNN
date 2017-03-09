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

#include "scale_arg.h"
#include "primitive_type_base.h"
#include "network_impl.h"

namespace cldnn
{
    primitive_type_id scale_type_id()
    {
        static primitive_type_base<scale, scale_arg> instance;
        return &instance;
    }

    layout scale_arg::calc_output_layout(const topology_map& topology_map, std::shared_ptr<const scale> desc)
    {
        auto input_desc = topology_map.at(desc->input()[0])->primitive_desc;
        auto result = input_desc->type()->calc_output_layout(topology_map, input_desc);
        return result;
    }

    scale_arg::scale_arg(network_impl& network, std::shared_ptr<const scale> desc)
        :primitive_arg_base(network, desc, calc_output_layout(network.get_topology()->get_primitives(), desc))
    {
        auto scale_format = scale_memory().get_layout().size.format;

        auto scale_batch_size = scale_memory().get_layout().size.batch[0];
        auto scale_feature_size = scale_memory().get_layout().size.feature[0];
        auto scale_x_size = scale_memory().get_layout().size.spatial[0];
        auto scale_y_size = scale_memory().get_layout().size.spatial[1];

        auto input_batch_size = input_memory(0).get_layout().size.batch[0];
        auto input_feature_size = input_memory(0).get_layout().size.feature[0];
        auto input_x_size = input_memory(0).get_layout().size.spatial[0];
        auto input_y_size = input_memory(0).get_layout().size.spatial[1];

        if((scale_batch_size != input_batch_size) && (scale_batch_size != 1))
            throw std::runtime_error("Batch dimension mismatch between input and scale input!");
        if ((scale_feature_size != input_feature_size) && (scale_feature_size != 1))
            throw std::runtime_error("Feature dimension mismatch between input and scale input!");
        if ((scale_x_size != input_x_size) && (scale_x_size != 1))
            throw std::runtime_error("X dimension mismatch between input and scale input!");
        if ((scale_y_size != input_y_size) && (scale_y_size != 1))
            throw std::runtime_error("Y dimension mismatch between input and scale input!");

        if (bias_term())
        {
            auto bias_format = bias_memory().get_layout().size.format;
            auto bias_raw_sizes = bias_memory().get_layout().size.raw;

            if (scale_format != bias_format) throw std::runtime_error("Scale input format do not match bias format!");

            for (size_t i = 0; i < bias_memory().get_layout().size.raw.size(); ++i)
            {
                if (scale_memory().get_layout().size.raw[i] != bias_raw_sizes[i]) throw std::runtime_error("Scale input size do not match bias size!");
            }
        }
    }

    const memory& scale_arg::scale_memory() const
    {
        return _network.get_primitive(argument.scale_input)->output_memory();
    }

    const bool& scale_arg::bias_term() const
    {
        return argument.bias_term;
    }

    const memory& scale_arg::bias_memory() const
    {
        return _network.get_primitive(argument.bias)->output_memory();
    }
}