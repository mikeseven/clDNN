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
#include "fully_connected_arg.h"
#include "primitive_type_base.h"
namespace cldnn
{
primitive_type_id fully_connected::type_id()
{
    static primitive_type_base<fully_connected, fully_connected_arg> instance;
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

layout fully_connected_arg::calc_output_layout(network_impl& network, std::shared_ptr<const fully_connected> desc)
{
    auto input = network.get_primitive(desc->input()[0]);
    auto input_layout = input->output_memory().get_layout();

    auto bias = network.get_primitive(desc->bias());
    auto bias_layout = bias->output_memory().get_layout();

    if(is_batch_after_spatial(input_layout.size.format.order()))
    {
        return layout(input_layout.data_type, tensor(format::xb, { bias_layout.size.spatial[0], input_layout.size.batch[0] }));
    }
    else
    {
        return layout(input_layout.data_type, tensor(format::bx, { input_layout.size.batch[0], bias_layout.size.spatial[0] }));
    }
}

fully_connected_arg::fully_connected_arg(network_impl& network, std::shared_ptr<const fully_connected> desc)
    :primitive_arg_base(network, desc, calc_output_layout(network, desc))
{
    auto input_size = input_memory(0).get_layout().size;
    auto output_size = output_memory().get_layout().size;

    if(input_size.format != format::yxfb && (input_size.raw.size() != output_size.raw.size()) )
    {
        throw std::invalid_argument("Fully connected input/output number of dimension does not match.");
    }
}

const memory& fully_connected_arg::weights_memory() const
{
    return _network.get_primitive(argument.weights)->output_memory();
}

const memory& fully_connected_arg::bias_memory() const
{
    return _network.get_primitive(argument.bias)->output_memory();
}

}

//namespace neural {
//
//fully_connected::arguments::arguments(
//    primitive            out,
//    primitive            in,
//    primitive            weights,
//    primitive            bias,
//    bool                 use_relu,
//    float                negative_slope)
//    : output({ out })
//    , input({ in, weights, bias })
//    , use_relu(use_relu)
//    , negative_slope(negative_slope) {}
//
//fully_connected::arguments::arguments(
//    neural::memory::format::type out_fmt,
//    primitive                    in,
//    primitive                    weights,
//    primitive                    bias,
//    bool                         use_relu,
//    float                        negative_slope)
//    : use_relu(use_relu)
//    , negative_slope(negative_slope)
//{
//    // if input is previouse layer, not memory primitive need to set input to output memory of this primitive
//    const auto& input_mem = get_memory_primitive(in);
//    if (in.id() != type_id<const memory>()->id) {
//        input = { in.output[0], weights, bias };
//    }
//    else {
//        input = { in, weights, bias };
//    }
//
//    neural::vector<uint32_t> output_size = {
//        input_mem.argument.size.batch[0],
//        { { get_memory_primitive(bias).argument.size.spatial[0] } },
//        1
//    };
//
//    output = { memory::allocate({ out_fmt, output_size }) };
//}
//
//// creates primitive with fully_connected implementation that supports provided arguments
//primitive fully_connected::create(fully_connected::arguments arg) {
//    auto& input_arg = arg.input[0].primitive().as<const memory&>().argument;
//    auto& output_arg = arg.output[0].as<const memory&>().argument;
//    
//    if (input_arg.format == memory::format::yxfb_f32 ||
//        input_arg.format == memory::format::yxfb_f16)
//    {
//        // NOTE: Testing for supported weights format is now inside each device implementation of the primitve (e.g. fully_connected_gpu).
//    }
//    else
//    {
//        if (input_arg.size.raw.size() != output_arg.size.raw.size())
//            throw std::runtime_error("Fully connected input/output number of dimension does not match.");
//    }
//
//    return is_a_primitive::create<fully_connected>(arg);
//}
//
//}
