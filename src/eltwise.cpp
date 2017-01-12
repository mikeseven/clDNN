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
#include "eltwise_arg.h"
#include "primitive_type_base.h"
#include "network_impl.h"

namespace cldnn
{
primitive_type_id eltwise::type_id()
{
    static primitive_type_base<eltwise, eltwise_arg> instance;
    return &instance;
}

layout eltwise_arg::calc_output_layout(network_impl& network, std::shared_ptr<const eltwise> desc)
{
    auto& input_mem = network.get_primitive(desc->input()[0])->output_memory();
    return input_mem.get_layout();
}

eltwise_arg::eltwise_arg(network_impl& network, std::shared_ptr<const eltwise> desc)
    :primitive_arg_base(network, desc, calc_output_layout(network, desc))
{
    auto input_format = input_memory(0).get_layout().size.format;
    auto output_format = output_memory().get_layout().size.format;
    auto input2_format = input2_memory().get_layout().size.format;

    if (input_format != input2_format)
    {
        throw std::runtime_error("Different formats of eltwise input layers");
    }
}

const memory& eltwise_arg::input2_memory() const
{
    return _network.get_primitive(argument.input2)->output_memory();
}
}
