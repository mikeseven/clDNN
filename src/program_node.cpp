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

using namespace cldnn;

primitive_id cldnn::details::empty_primitive_id{};

void program_node::replace_dependency(size_t idx, program_node& new_dep)
{
    if (idx >= dependencies.size())
        return;
    if (dependencies[idx] == &new_dep)
        return;

    dependencies[idx]->users.remove(this);
    myprog.remove_if_dangling(*dependencies[idx]);

    dependencies[idx] = &new_dep;
    desc->dependecies()[idx].get() = new_dep.id();
    new_dep.users.push_back(this);
}

void program_node::replace_dependency(program_node const& old_dep, program_node& new_dep)
{
    for (size_t i = 0; i < dependencies.size(); ++i)
        if (dependencies[i] == &old_dep)
            return replace_dependency(i, new_dep);
}

void program_node::remove_dependency(size_t idx)
{
    if (idx >= dependencies.size())
        return;

    dependencies[idx]->users.remove(this);
    myprog.remove_if_dangling(*dependencies[idx]);
    dependencies.erase(dependencies.begin() + idx);
}

layout program_node::get_output_layout()
{
    if (valid_output_layout)
        return output_layout;

    for (auto dep : dependencies)
        dep->get_output_layout();

    auto new_layout = desc->type->calc_output_layout(*this);
    //TODO: after merging padding into layout, calc_output_layout can now return padding as well
    // for now just ignore it and preserve already set padding value - in future we should probably take care of this
    // situation however.
    new_layout.data_padding = output_layout.data_padding;
    if (new_layout != output_layout) //output_layout has changed! invalidate users
        invalidate_users();

    output_layout = new_layout;
    valid_output_layout = true;
    return std::move(new_layout);
}