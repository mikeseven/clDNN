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

#include "pooling_arg.h"
#include "primitive_type_base.h"
#include "network_impl.h"

#include <cmath>

namespace cldnn
{
primitive_type_id pooling::type_id()
{
    static primitive_type_base<pooling, pooling_arg> instance;
    return &instance;
}

layout pooling_arg::calc_output_layout(const topology_map& topology_map, std::shared_ptr<const pooling> desc)
{
    auto input_desc = topology_map.at(desc->input()[0])->primitive_desc;
    auto input_layout = input_desc->type()->calc_output_layout(topology_map, input_desc);
    assert(input_layout.size.spatial.size() == 2);
    auto input_offset = desc->input_offset().transform(input_layout.size.format, 0).sizes();
    auto siz = desc->size.transform(input_layout.size.format, 1).sizes();
    auto strd = desc->stride.transform(input_layout.size.format, 1).sizes();
    //TODO !!!implement correct output size calculation!!!
    auto output_sizes = input_layout.size.sizes();
    auto format_order = input_layout.size.format.order();
    assert(output_sizes.size() == format_order.size());
    for (decltype(output_sizes.size()) i = 0; i < output_sizes.size(); i++)
    {
        if (format_traits::is_spatial_char(format_order[i]))
        {
            if (strd[i] < 1) throw std::invalid_argument("stride should be >= 1");
            if (strd[i] > 1 || 0 != input_offset[i])
            {
                output_sizes[i] = static_cast<int32_t>(ceil(static_cast<float>(output_sizes[i] - (2 * input_offset[i]) - siz[i]) / static_cast<float>(strd[i]))) + 1;
            }
            else
            {
                output_sizes[i] = (output_sizes[i] - (2 * input_offset[i]) - siz[i]) / strd[i] + 1;
            }
        }
    }

    return{ input_layout.data_type, {input_layout.size.format, output_sizes} };
}

pooling_arg::pooling_arg(network_impl& network, std::shared_ptr<const pooling> desc)
    :primitive_arg_base(network, desc, calc_output_layout(network.get_topology()->get_primitives(), desc))
{}
}
