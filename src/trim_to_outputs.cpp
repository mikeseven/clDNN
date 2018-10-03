/*
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
*/

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "pass_manager.h"

using namespace cldnn;

//ToDo remove friendship relation from  program_node and program_impl
void trim_to_outputs::run(program_impl &p) 
{
        size_t actual_nodes = p.processing_order.size();
        if (!actual_nodes) //degenerated case but can happen
            return;

        if (p.outputs.size() == actual_nodes)
            return;

        //do backward bfs starting from all outputs
        std::list<const std::vector<program_node*>*> stack = { &(p.outputs) };
        while (!stack.empty())
        {
            auto nodes_list = stack.front();
            stack.pop_front();

            for (auto node : *nodes_list)
            {
                if (!node->is_marked())
                {
                    node->mark();
                    if (!node->get_dependencies().empty())
                        stack.push_back(&node->get_dependencies());
                }
            }
        }

        //all not-marked nodes should be removed
        std::list<program_node*> to_rem;
        for (auto node : p.processing_order)
        {
            if (node->is_type<input_layout>()) //input layout may become disconnected during prior boxes calculations so it may have not been marked at this place but we don't want to remove it
                node->mark();
            else if (!node->is_marked())
                to_rem.push_back(node);
        }

        for (auto const& node : to_rem)
        {
            //ToDo: replace by remove_node method in p
            if (node->is_input())
                p.inputs.remove(node);
            else
            {
                for (auto dep : node->dependencies)
                    if (dep->is_marked())
                        dep->users.remove(node);
            }

            for (auto user : node->users)
                if (user->is_marked())
                    user->dependencies.erase(std::remove(user->dependencies.begin(), user->dependencies.end(), node), user->dependencies.end());

            p.optimized_out.push_back(node->id());
            p.nodes_map.erase(node->id());
        }

        //unmark all nodes
        //ToDo: mark()/unmark() methods might cause hidden dependencies in between optimization passed. They shoud be encapsulated within the opt pass itself.
        for (auto& node : p.processing_order)
        {
            node->unmark();
        }
}