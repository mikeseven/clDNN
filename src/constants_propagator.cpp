/*
// Copyright (c) 2017 Intel Corporation
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

#include "constants_propagator.h"
#include "engine_impl.h"
#include "program_impl.h"
#include "network_impl.h"
#include "memory_impl.h"

#include "api/CPP/input_layout.hpp"

using namespace cldnn;

constants_propagator::constants_propagator(program_impl::ptr program) : prog(program)
{
}

void constants_propagator::visit_node(program_node& node)
{
    if (node.is_constant())
        handle_constant(node);
}

std::list<std::pair<primitive_id, memory>> cldnn::constants_propagator::calculate()
{
    if (!has_non_trivial_constants)
        return{};

    build_options bo;
    bo.set_option(build_option::optimize_data(false));
    bo.set_option(build_option::outputs(const_outputs));
    network_impl::ptr net = prog->get_engine()->build_network(tpl, bo);
    for (auto& cin : const_inputs)
        net->set_input_data(cin->id(), api_cast(cin->get_primitive()->mem.get()));

    net->execute({});
    net->reset_execution(true); //wait for computations to complete
    auto outputs = net->get_outputs();

    std::list<std::pair<primitive_id, memory>> ret;
    for (auto& out : outputs)
        ret.push_back({ out->id(), out->output_memory() });

    return ret;
}

void constants_propagator::handle_constant(program_node& node)
{
    if (!node.is_type<data>())
    {
        add_constant(node);
        if (node.has_non_const_user())
            const_outputs.push_back(node.id());
    }
}

void constants_propagator::add_constant(program_node& node)
{
    if (node.is_type<data>())
        return;

    tpl.add(node.desc);
    has_non_trivial_constants = true;

    //if a node is an endpoint, always add it as an output
    if (node.is_endpoint())
        const_outputs.push_back(node.id());

    //if a non-tirivial constant has a trivial input, add this input as an input for our network
    for (auto& dep : node.get_dependencies())
    {
        if (dep->is_type<data>())
        {
            tpl.add(std::make_shared<input_layout>(dep->id(), dep->as<data>().get_primitive()->mem.get_layout()));
            const_inputs.push_back(&dep->as<data>());
        }
    }
}

