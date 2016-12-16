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
#include "mean_substract_arg.h"
#include "primitive_type_base.h"
#include "network_impl.h"

namespace cldnn
{
primitive_type_id mean_substract::type_id()
{
    static primitive_type_base<mean_substract, mean_substract_arg> instance;
    return &instance;
}

layout mean_substract_arg::calc_output_layout(network_impl& network, std::shared_ptr<const mean_substract> desc)
{
    auto& input_mem = network.get_primitive(desc->input()[0])->output_memory();
    return input_mem.get_layout();
}

mean_substract_arg::mean_substract_arg(network_impl& network, std::shared_ptr<const mean_substract> desc)
    :primitive_arg_base(network, desc, calc_output_layout(network, desc))
{
    auto input_format = input_memory(0).get_layout().size.format;
    auto output_format = output_memory().get_layout().size.format;
    auto mean_format = mean_memory().get_layout().size.format;

    if (input_format != format::yxfb)
    {
        throw std::runtime_error("Mean subtract input is not in yxfb format!");
    }
    if (output_format != format::yxfb)
    {
        throw std::runtime_error("Mean subtract output is not in yxfb format!");
    }
    if (mean_format != format::yxfb && 
        mean_format != format::bfyx)
    {
        throw std::runtime_error("Mean subtract mean is not in yxfb or bfyx format!");
    }
}

const memory& mean_substract_arg::mean_memory() const
{
    return _network.get_primitive(argument.mean)->output_memory();
}
}
