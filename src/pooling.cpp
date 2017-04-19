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

#include "pooling_inst.h"
#include "primitive_type_base.h"


namespace cldnn
{
primitive_type_id pooling_type_id()
{
    static primitive_type_base<pooling, pooling_inst> instance;
    return &instance;
}

layout pooling_inst::calc_output_layout(parent::typed_node const& node)
{
    auto desc = node.get_primitive();

    auto input_layout = node.input().get_output_layout();

    if (input_layout.size.format != format::bfyx && input_layout.size.format != format::yxfb)
        throw std::invalid_argument("Pooling supports yxfb and bfyx formats only");
    if (input_layout.size.spatial.size() != 2)
        throw std::invalid_argument("One dimensional spatials aren't supported by pooling");

    auto input_offsets = desc->input_offset().transform(input_layout.size.format, 0).sizes();
    auto strides = desc->stride.transform(input_layout.size.format, 1).sizes();
    auto window_sizes = desc->size.transform(input_layout.size.format, 1).sizes();
    //TODO !!!implement correct output size calculation!!!
    auto output_sizes = input_layout.size.sizes();
    auto format_order = input_layout.size.format.order();
    if (output_sizes.size() != format_order.size())
        throw std::invalid_argument("output and input format size mismatch");

    for (decltype(output_sizes.size()) i = 0; i < output_sizes.size(); i++)
    {
        if (format_traits::is_spatial_char(format_order[i]))
        {
            // TODO: Consider moving general parameter verification to arguments constructor.
            if (strides[i] <= 0)
                throw std::invalid_argument("Stride must be positive (>= 1)");
            if (2 * input_offsets[i] >= output_sizes[i])
                throw std::invalid_argument("Input offset is greater than input data range. There is no input data to process");

            output_sizes[i] = static_cast<cldnn::tensor::value_type>(
                2 * input_offsets[i] < output_sizes[i]
                // ? std::max(output_sizes[i] - 2 * input_offsets[i] - window_sizes[i], 0) / strides[i] + 1
                ? ceil_div(std::max(output_sizes[i] - 2 * input_offsets[i] - window_sizes[i], 0), strides[i]) + 1
                : 0);
        }
    }

    return{ input_layout.data_type,{ input_layout.size.format, output_sizes } };
}
}
