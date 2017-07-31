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
#pragma once

#include "api/CPP/primitive.hpp"
#include "internal_primitive.h"

#include "meta_utils.h"

namespace cldnn
{

struct program_impl;

template <class T>
struct typed_program_node;

template <class PType>
struct internal_primitive_type_base;

/*
    Base class for all primitives which wraps API class and extends it to be used
    in graph context.

    Besides primitive description provided by user, this class includes functionality to
    ask for direct predecessors and succesors as well as takes care of changes to primitive
    which would affect other graph's nodes (the most commont case is probably calculating output layout).

    At graph level, all connections between nodes are directly stored inside program_nodes - in oposite
    to API level where all primitives store only ids of related ones.
*/
struct program_node
{
    friend struct program_impl;

    program_node(std::shared_ptr<primitive> prim, program_impl& prog) : desc(prim), myprog(prog)
    {
        if (prim)
            output_layout.data_padding = prim->output_padding;
    }

    program_node(program_node const&) = delete;

public:
    virtual const primitive_id& id() const { return desc->id; }
    virtual primitive_type_id type() const { return desc->type; }

    template <class PType>
    bool is_type() const
    {
        static_assert(meta::is_primitive_v<PType>, "Type argument for program_node::is_type should be a non-const, non-volatile type derived from primitive");
        return type() == PType::type_id();
    }

    auto& get_program() { return myprog; }
    auto const& get_program() const { return myprog; }

    auto get_selected_impl() const { return selected_impl; }

    auto const& get_dependencies() const { return dependencies; }
    auto& get_dependency(size_t idx) const { return *dependencies.at(idx); }

    void replace_dependency(size_t idx, program_node& new_dep);
    void replace_dependency(program_node const& old_dep, program_node& new_dep);

    auto const& get_users() { return users; }
    // for const method, add const to stored successors/predecessors
    auto const& get_users() const { return reinterpret_cast<const std::list<const program_node*>&>(users); }

    //do not modify primitive directly to keep synchronisation wit graph
    std::shared_ptr<const primitive> get_primitive() const { return desc; }

    //primitive modification functions
    void set_output_padding(padding const& padd)
    {
        //changing output padding shouldn't cause any changes to other primitives
        //so just change it
        output_layout.data_padding = padd;
    }

    void merge_output_padding(padding const& padd)
    {
        set_output_padding(padding::max(padd, output_layout.data_padding));
    }

    layout get_output_layout();

    layout get_output_layout() const
    {
        if (!valid_output_layout)
            throw std::runtime_error("Output layout not calculated");

        return output_layout;
    }

    void set_output_layout(layout layout)
    {
        layout.data_padding = output_layout.data_padding;
        if (layout != output_layout) //output_layout has changed! invalidate users
            invalidate_users();

        output_layout = layout;
        valid_output_layout = true;
    }

    void recalc_output_layout()
    {
        valid_output_layout = false;
        get_output_layout();
    }

    bool is_padded() { return static_cast<bool>(get_output_layout().data_padding); }
    bool is_padded() const { return static_cast<bool>(get_output_layout().data_padding); }

    bool has_padded_dependency()
    {
        return std::any_of(get_dependencies().begin(), get_dependencies().end(), [](program_node* node) { return node->is_padded(); });
    }

    bool has_padded_dependency() const
    {
        return std::any_of(get_dependencies().begin(), get_dependencies().end(), [](const program_node* node) { return node->is_padded(); });
    }

    auto is_input() const { return dependencies.empty(); }
    auto is_endpoint() const { return users.empty(); }
    auto set_output(bool out) { output = out; }
    auto is_output() const { return output; }

    auto mark() { user_mark = true; }
    auto unmark() { user_mark = false; }
    auto is_marked() const { return user_mark; }

    // returns immidiate dominator of this node if it's not its direct predecessor, otherwise returns nullptr
    program_node* get_dominator() { return dominator; }
    const program_node* get_dominator() const { return dominator; }

    //returns joint point associated with this node,
    //if the node is not a split point (i.e. it has exactly one user) this function returns nullptr,
    //otherwise returns pointer to a node which immidiately post-dominates this
    program_node* get_joint() { return joint; }
    const program_node* get_joint() const { return joint; }

    bool is_joint_point() const { return dominator != nullptr; }
    bool is_split_point() const { return joint != nullptr; }

    bool is_constant() const { return constant; }

    //returns true if all paths from network's source to sink must come through this node
    //(i.e. if this is a dominator of all the outputs)
    //a source, in this context, is defined as an input which lies within a data flow (see is_in_data_flow)
    bool is_in_main_branch() const { return main_branch; }

    //returns true if this node is within main data flow of the network (i.e. it does not describe helper data like convolution's weights etc.)
    bool is_in_data_flow() const { return data_flow; }

    //conversion from generic to specific
    template <class To, class..., class = std::enable_if_t<!std::is_same<To, primitive>::value>>
    typed_program_node<To>& as()
    {
        if (type() != To::type_id())
            throw std::invalid_argument("program_node: mismatching primitive's type");

        return reinterpret_cast<typed_program_node<To>&>(*this);
    }

    template <class To, class..., class = std::enable_if_t<!std::is_same<To, primitive>::value>>
    typed_program_node<To> const& as() const
    {
        if (type() != To::type_id())
            throw std::invalid_argument("program_node: mismatching primitive's type");

        return reinterpret_cast<typed_program_node<To> const&>(*this);
    }

    template <class To>
    operator typed_program_node<To>& ()
    {
        return as<To>();
    }

    template <class To>
    operator typed_program_node<To> const& () const
    {
        return as<To>();
    }

protected:
    std::shared_ptr<primitive> desc;
    program_impl& myprog;

    std::shared_ptr<primitive_impl> selected_impl;

    bool valid_output_layout = false;
    layout output_layout = layout(data_types::f32, format::bfyx, tensor());

    std::vector<program_node*> dependencies;
    std::list<program_node*> users;

    std::list<program_node*>::const_iterator processing_itr;
    uint32_t processing_num = 0;

    program_node* dominator = nullptr;
    program_node* joint = nullptr;
    bool constant = false;
    bool main_branch = true;
    bool data_flow = false;

    bool output = false;
    bool user_mark = false;

    void invalidate_users() const
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
};

namespace details
{
    template <class PType>
    struct api_typed_program_node_base : public program_node
    {
        static_assert(meta::is_api_primitive_v<PType>, "PType should name a non-const, non-volatile type derived from cldnn::primitive but not from cldnn::internal_primitive");
        friend struct cldnn::program_impl;

    public:
        using program_node::program_node;

        std::shared_ptr<const PType> get_primitive() const { return std::static_pointer_cast<const PType>(program_node::get_primitive()); }

    protected:
        std::shared_ptr<PType> typed_desc() const { return std::static_pointer_cast<PType>(desc); }
    };

    extern primitive_id empty_primitive_id;

    template <class PType>
    struct internal_typed_program_node_base : public program_node
    {
        static_assert(meta::is_internal_primitive_v<PType>, "PType should name a non-const, non-volatile type derived from cldnn::internal_primitive");
        friend struct cldnn::program_impl;

    public:
        internal_typed_program_node_base(program_impl& prog) : program_node(nullptr, prog)
        {}

        const primitive_id& id() const override { return empty_primitive_id; }
        primitive_type_id type() const override { return PType::type_id(); }

        template <class... Guard>
        [[noreturn]]
        void get_primitive(Guard&&...)
        {
            static_assert(meta::always_false_v<meta::pack<Guard...>>, "Trying to get primitive from internal node");
        }

    protected:
        template <class... Guard>
        [[noreturn]]
        void typed_desc(Guard&&...)
        {
            static_assert(meta::always_false_v<meta::pack<Guard...>>, "Trying to get primitive from internal node");
        }
    };
}

/*
Template class used to indicate that usage context requires 'program_node' to wrap primitive
of type 'PType'. Successful conversion from 'program_node' to 'typed_program_node<PType>' means
that this restriction in fact holds and functions/method/etc. may saftly use uderlaying primitive.

This class shadows 'get_primitive' method from base class which now returns pointer to more specific
type.
*/
template <class PType>
using typed_program_node_base = std::conditional_t<meta::is_api_primitive_v<PType>, details::api_typed_program_node_base<PType>, details::internal_typed_program_node_base<PType>>;

/*
    Actual template class used in context which requires 'program_node' to wrap
    primitive of type 'PType'. This class is introduced to provide possibility of explicit specialization.
    In most cases such specializations would add accessors to make access to PType-specific fields easier.

    It's not required to specialize this class for new primitives types.
*/
template <class PType>
struct typed_program_node : public typed_program_node_base<PType>
{
    using typed_program_node_base<PType>::typed_program_node_base;

    auto& input() const { return program_node::get_dependency(0); }
};

}