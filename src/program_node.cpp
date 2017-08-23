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

#include "program_node.h"
#include "program_impl.h"
#include "primitive_inst.h"

using namespace cldnn;

program_node::program_node(std::shared_ptr<primitive> prim, program_impl & prog) : desc(prim), myprog(prog)
{
    if (prim)
        output_layout.data_padding = prim->output_padding;

    processing_itr = prog.processing_order.end();
}

void program_node::replace_dependency(size_t idx, program_node& new_dep, bool detach_whole_branch)
{
    if (idx >= dependencies.size())
        return;
    if (dependencies[idx] == &new_dep)
        return;

    dependencies[idx]->users.remove(this);
    myprog.remove_if_dangling(*dependencies[idx], detach_whole_branch);

    dependencies[idx] = &new_dep;
    if (!is_type<internal_primitive>())
        desc->dependecies()[idx].get() = new_dep.id();
    new_dep.users.push_back(this);
}

void program_node::replace_dependency(program_node const& old_dep, program_node& new_dep, bool detach_whole_branch)
{
    for (size_t i = 0; i < dependencies.size(); ++i)
        if (dependencies[i] == &old_dep)
            return replace_dependency(i, new_dep, detach_whole_branch);
}

bool program_node::has_next() const
{
    auto itr = processing_itr;
    return (++itr == myprog.processing_order.end());
}

std::vector<primitive_id> program_node::get_dependencies_ids() const
{
    std::vector<primitive_id> dep_ids;
    for (auto& dependency : dependencies)
        dep_ids.push_back(dependency->get_primitive()->id);
    return dep_ids;
}

void program_node::remove_dependency(size_t idx)
{
    if (idx >= dependencies.size())
        return;

    dependencies[idx]->users.remove(this);
    myprog.remove_if_dangling(*dependencies[idx]);
    dependencies.erase(dependencies.begin() + idx);
}

void program_node::remove_dependency(program_node & node)
{
    for (size_t i = 0; i < dependencies.size(); ++i)
        if (dependencies[i] == &node)
            remove_dependency(i);
}

bool program_node::is_detached(bool whole_branch)
{
    if (!users.empty())
        return false;
    if (!whole_branch && !dependencies.empty())
        return false;
    if (joint != nullptr || dominator != nullptr)
        return false;

    return true;
}

layout program_node::calc_output_layout() const
{
    return type()->calc_output_layout(*this);
}

layout program_node::get_output_layout(bool invalidate_users_if_changed)
{
    if (valid_output_layout)
        return output_layout;

    auto new_layout = calc_output_layout();
    set_output_layout(new_layout, invalidate_users_if_changed);
    return new_layout;
}

layout program_node::get_output_layout() const
{
    if (!valid_output_layout)
        throw std::runtime_error("Output layout not calculated");

    return output_layout;
}

bool program_node::set_output_layout(layout new_layout, bool invalidate_users_if_changed)
{
    //TODO: after merging padding into layout, calc_output_layout can now return padding as well
    // for now just ignore it and preserve already set padding value - in future we should probably take care of this
    // situation however.
    new_layout.data_padding = output_layout.data_padding;
    bool changed = (new_layout != output_layout);
    if (changed && invalidate_users_if_changed) //output_layout has changed! invalidate users
        invalidate_users();

    output_layout = new_layout;
    valid_output_layout = true;
    return changed;
}

bool program_node::recalc_output_layout(bool invalidate_users_if_changed)
{
    return set_output_layout(calc_output_layout(), invalidate_users_if_changed);
}

bool program_node::has_padded_dependency()
{
    return std::any_of(get_dependencies().begin(), get_dependencies().end(), [](program_node* node) { return node->is_padded(); });
}

bool program_node::has_padded_dependency() const
{
    return std::any_of(get_dependencies().begin(), get_dependencies().end(), [](const program_node* node) { return node->is_padded(); });
}

void program_node::invalidate_users() const
{
    for (auto& user : users)
    {
        if (user->valid_output_layout)
        {
            user->valid_output_layout = false;
            user->invalidate_users();
        }
    }
}

primitive_id details::internal_program_node_base::get_next_internal_id()
{
    static std::atomic<uint64_t> counter{ 0 };
    auto idx = counter++;
    return primitive_id("_cldnn_internal_") + std::to_string(idx);
}

details::internal_program_node_base::internal_program_node_base(program_impl & prog) : program_node(nullptr, prog), internal_id(get_next_internal_id())
{
}

void details::internal_program_node_base::set_implementation(std::unique_ptr<primitive_impl>&& impl)
{
    selected_impl = std::move(impl);
}
