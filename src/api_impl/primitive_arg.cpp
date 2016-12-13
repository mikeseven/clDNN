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
primitive_arg::primitive_arg(network_impl& network, std::shared_ptr<const primitive> desc, const memory& output_memory)
    : _network(network)
    , _desc(desc)
    , _inputs(network.get_primitives(desc->input()))
    , _output(output_memory)
{}

primitive_arg::primitive_arg(network_impl& network, std::shared_ptr<const primitive> desc, const layout& output_layout)
    : primitive_arg(network, desc, network.get_engine()->allocate_buffer(output_layout))
{}

}
