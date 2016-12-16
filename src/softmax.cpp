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

#include "softmax_arg.h"
#include "primitive_type_base.h"
#include "network_impl.h"

namespace cldnn
{
primitive_type_id softmax::type_id()
{
    static primitive_type_base<softmax, softmax_arg> instance;
    return &instance;
}

layout softmax_arg::calc_output_layout(network_impl& network, std::shared_ptr<const softmax> desc)
{
    auto& input_mem = network.get_primitive(desc->input()[0])->output_memory();
    return input_mem.get_layout();
}

softmax_arg::softmax_arg(network_impl& network, std::shared_ptr<const softmax> desc)
    : primitive_arg_base(network, desc, calc_output_layout(network, desc))
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
}

}
