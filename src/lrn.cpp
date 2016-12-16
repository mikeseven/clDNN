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

#include "normalization_arg.h"
#include "primitive_type_base.h"
#include "network_impl.h"

namespace cldnn
{
primitive_type_id normalization::type_id()
{
    static primitive_type_base<normalization, normalization_arg> instance;
    return &instance;
}

layout normalization_arg::calc_output_layout(network_impl& network, std::shared_ptr<const normalization> desc)
{
    auto& input_mem = network.get_primitive(desc->input()[0])->output_memory();
    return input_mem.get_layout();
}

normalization_arg::normalization_arg(network_impl& network, std::shared_ptr<const normalization> desc)
    :primitive_arg_base(network, desc, calc_output_layout(network, desc))
{}

}
