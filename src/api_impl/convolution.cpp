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
#include "api/primitives/convolution.hpp"
#include "primitive_type.h"
#include "primitive_arg.h"
#include "network_builder.h"

namespace cldnn
{
class convolution_arg : public primitive_arg
{
public:
    static layout calc_output_layout(network_builder& builder, std::shared_ptr<const convolution> desc)
    {
        auto input = builder.get_primitive(desc->input()[0]);
        auto input_layout = input->output_memory().get_layout();
        auto weight0 = builder.get_primitive(desc->weights()[0]);
        auto weights_layout = weight0->output_memory().get_layout();
        auto input_offset = desc->input_offset().transform(format::yx, 0);
        auto strd = desc->stride().transform(format::yx, 0);
        auto split = desc->weights().size();

        // compute how many outputs in rows and columns will be generate by filter. 
        // outp <= (input_size - (2*input_offset) - kernel_size)/ stride 
        auto kernel_xy = weights_layout.size.spatial;
        assert(kernel_xy.size() == 2);
        auto output_spatial_x = (input_layout.size.spatial[0] - (2 * input_offset.spatial[0]) - kernel_xy[0]) / strd.spatial[0] + 1;
        auto output_spatial_y = (input_layout.size.spatial[1] - (2 * input_offset.spatial[1]) - kernel_xy[1]) / strd.spatial[1] + 1;
        // get output feature map from weights. It should be the same as number of biases. Will be verifed in convolution::create()
        auto number_of_features = weights_layout.size.feature[0] * static_cast<int32_t>(split);

        tensor output_size(format::yxfb, {
            input_layout.size.batch[0], number_of_features, output_spatial_x, output_spatial_y }
        );

        return{ input_layout.data_type, output_size.transform(input_layout.size.format, 1) };
    }

    convolution_arg(network_builder& builder, std::shared_ptr<const convolution> desc)
        : primitive_arg(builder, desc, builder.get_engine().allocate_memory(calc_output_layout(builder, desc)))
    {}
};

struct convolution_type : public primitive_type
{
    std::shared_ptr<const primitive> from_dto(const primitive_dto* dto) const override
    {
        if(dto->type != this) throw std::invalid_argument("dto: primitive type mismatch");
        return std::make_shared<convolution>(dto->as<convolution>());
    }
    std::shared_ptr<const primitive_arg> create_arg(network_builder& builder, std::shared_ptr<const primitive> desc) const override
    {
        if (desc->type() != this) throw std::invalid_argument("desc: primitive type mismatch");
        return std::make_shared<convolution_arg>(builder, std::static_pointer_cast<const convolution>(desc));
    }
};

primitive_type_id convolution::type_id()
{
    static convolution_type instance;
    return &instance;
}
}
