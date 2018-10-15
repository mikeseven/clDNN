// Copyright (c) 2018 Intel Corporation
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

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <api/CPP/condition.hpp>

#include "network_impl.h"
#include "primitive_inst.h"

namespace cldnn
{
namespace details
{

}

template <>
struct typed_program_node<condition> : public typed_program_node_base<condition>
{
private:
    using parent = typed_program_node_base<condition>;

    class branch
    {
    public:
        branch(topology_impl& tpl) : _topology(tpl) {}

        void set(const program_node& node)
        {
            add_or_change_input_layout(node);
            _program = node.get_program().get_engine().build_program(_topology, node.get_program().get_options(), true); //rebuild program 
        }
        decltype(auto) get() const { return _program; }

    private:
        topology_impl & _topology;
        program_impl::ptr _program = nullptr;

        void add_or_change_input_layout(const program_node& node)
        {
            auto layout = node.get_dependency(0).get_output_layout();
            auto input_id = node.as<condition>().result_id();
            if (_program == nullptr) //if first run, create input_layout
            {
                _topology.add(std::make_shared<input_layout>(input_id, layout));
                for (auto& prim : _topology.get_primitives())
                {
                    for (auto& inp : prim.second->input)
                    {
                        if (inp == node.id())
                            inp = input_id;
                    }
                }
            }
            else
            {
                _topology.change_input_layout(input_id, layout);
            }
        }
    };

public:
    using parent::parent;

    typed_program_node(std::shared_ptr<primitive> prim, program_impl& prog)
        : parent(prim, prog)
        , _branch_true(*api_cast(this->get_primitive()->topology_true.get()))
        , _branch_false(*api_cast(this->get_primitive()->topology_false.get()))
    {
    }

    decltype(auto) input() const { return get_dependency(0); }
    decltype(auto) compare() const { return get_dependency(1); }
    decltype(auto) func() const { return get_primitive()->function; }
    decltype(auto) offset() const { return get_primitive()->offset; }
    void set_branches() const
    {
        _branch_true.set(*this);
        _branch_false.set(*this);
    }
    decltype(auto) get_branch_true() const { return _branch_true.get(); }
    decltype(auto) get_branch_false() const{ return _branch_false.get(); }
    primitive_id result_id() const { return id() + ":result"; }

private:
    mutable branch _branch_true;
    mutable branch _branch_false;
};

using condition_node = typed_program_node<condition>;


template <>
class typed_primitive_inst<condition> : public typed_primitive_inst_base<condition>
{
    using parent = typed_primitive_inst_base<condition>;

public:
    static layout calc_output_layout(condition_node const& node);
    static std::string to_string(condition_node const& node);
    typed_primitive_inst(network_impl& network, condition_node const& node);

    decltype(auto) input_memory() const { return dep_memory(0); }
    decltype(auto) compare_memory() const { return dep_memory(1); }
    decltype(auto) get_net_true() const { return _net_true; }
    decltype(auto) get_net_false() const { return _net_false; }
    primitive_id result_id() const { return node.result_id(); }
private:
    network_impl::ptr _net_true;
    network_impl::ptr _net_false;
};

using condition_inst = typed_primitive_inst<condition>;
}
