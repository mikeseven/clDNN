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
    primitive_type_id scale::type_id()
    {
        static primitive_type_base<scale, scale_arg> instance;
        return &instance;
    }

    layout scale_arg::calc_output_layout(network_impl& network, std::shared_ptr<const scale> desc)
    {
        auto& input_mem = network.get_primitive(desc->input()[0])->output_memory();
        return input_mem.get_layout();
    }

    scale_arg::scale_arg(network_impl& network, std::shared_ptr<const scale> desc)
        :primitive_arg_base(network, desc, calc_output_layout(network, desc))
    {
        auto input_format = input_memory(0).get_layout().size.format;
        auto output_format = output_memory().get_layout().size.format;
        auto scale_format = scale_memory().get_layout().size.format;

        if (bias_term())
        {
            auto bias_format = bias_memory().get_layout().size.format;
            if (scale_format != bias_format)
            {
                throw std::runtime_error("Scale input format do not match bias format!");
            }
        }
    }

    const memory& scale_arg::scale_memory() const
    {
        return _network.get_primitive(argument.scale_input)->output_memory();
    }

    const int& scale_arg::axis() const
    {
        return argument.axis;
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