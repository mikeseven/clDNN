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
#include "primitive_arg.h"
#include "network_impl.h"
#include "engine_impl.h"

namespace cldnn
{
refcounted_obj_ptr<event_impl> primitive_arg::execute(const std::vector<refcounted_obj_ptr<event_impl>>& events) const
{
    std::vector<refcounted_obj_ptr<event_impl>> dependencies(_inputs.size());

    std::transform(
        std::begin(_inputs),
        std::end(_inputs),
        std::begin(dependencies),
        [&](decltype(_inputs.front()) input)
        {
            return get_network().execute_primitive(input, events);
        });

    return _impl->execute(dependencies.size() > 0 ? dependencies : events);
}

primitive_arg::primitive_arg(network_impl& network, std::shared_ptr<const primitive> desc, const memory& output_memory)
    : _network(network)
    , _desc(desc)
    , _inputs(network.get_primitives(desc->dependecies()))
    , _output(output_memory)
{}

primitive_arg::primitive_arg(network_impl& network, std::shared_ptr<const primitive> desc, const layout& output_layout)
    : primitive_arg(network, desc, allocate_output(network, desc, output_layout))
{}

memory primitive_arg::allocate_output(network_impl& network, std::shared_ptr<const primitive> desc, const layout& output_layout)
{
    auto output_size = output_layout.size;
    auto padding_size = desc->output_padding().size();
    auto padding_size2 = padding_size.mul(2);
    auto allocation_size = output_size.add(padding_size2);
    //auto allocation_size = output_layout.size.add(desc()->output_padding().size().mul(2));
    return network.get_engine()->allocate_buffer({ output_layout.data_type, allocation_size });
}
}
