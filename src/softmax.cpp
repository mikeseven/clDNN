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

namespace
{
    bool is_batch_after_spatial(const std::string order)
    {
        bool spatial_found = false;
        bool batch_found = false;
        for (auto c : order)
        {
            switch (c)
            {
            case 'b':
            case 'n':
                batch_found = true;
                if (spatial_found)
                    return true;
            case 'x':
            case 'y':
            case 'z':
            case 'w':
            case 's':
                spatial_found = true;
                if (batch_found)
                    return false;
            default: break;
            }
        }
        return false;
    }
}

layout softmax_arg::calc_output_layout(network_impl& network, std::shared_ptr<const softmax> desc)
{
    auto& input_mem = network.get_primitive(desc->input()[0])->output_memory();
    auto input_layout = input_mem.get_layout();


    cldnn::layout layoutTemp = (is_batch_after_spatial(input_layout.size.format.order())) ?
        cldnn::layout(input_layout.data_type, tensor(format::xb, { input_layout.size.spatial[0], input_layout.size.batch[0] })) :
        cldnn::layout(input_layout.data_type, tensor(format::bx, { input_layout.size.batch[0], input_layout.size.spatial[0] }));
    if (input_layout.size.raw.size() == 4) layoutTemp = cldnn::layout(input_layout.data_type, tensor(format::xb, { input_layout.size.feature[0], input_layout.size.batch[0] }));
    return layoutTemp;
    //return input_mem.get_layout();
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
