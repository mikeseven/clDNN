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

layout pooling_arg::calc_output_layout(network_impl& network, std::shared_ptr<const pooling> desc)
{
    auto& input_mem = network.get_primitive(desc->input()[0])->output_memory();
    auto output_layout = input_mem.get_layout();
    auto input_offset = desc->input_offset().transform(output_layout.size.format, 0);
    auto siz = desc->size.transform(output_layout.size.format, 1);
    auto strd = desc->stride.transform(output_layout.size.format, 1);
    //TODO !!!implement correct output size calculation!!!
    for (size_t i = 0; i < output_layout.size.spatial.size(); i++)
    {
        if (strd.spatial[i] < 1) throw std::invalid_argument("stride should be >= 1");
        if (strd.spatial[i] > 1 || 0 != input_offset.spatial[0])
        {
            output_layout.size.spatial[i] = static_cast<int32_t>(ceil(static_cast<float>(output_layout.size.spatial[i] - (2 * input_offset.spatial[i]) - siz.spatial[i]) / static_cast<float>(strd.spatial[i]))) + 1;
        }
        else
        {
            output_layout.size.spatial[i] = (output_layout.size.spatial[i] - (2 * input_offset.spatial[i]) - siz.spatial[i]) / strd.spatial[i] + 1;
        }
    }
    return output_layout;
}

pooling_arg::pooling_arg(network_impl& network, std::shared_ptr<const pooling> desc)
    :primitive_arg_base(network, desc, calc_output_layout(network, desc))
{}
}
