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

#pragma once


#include "network_impl.h"
#include "memory_impl.h"
#include "meta_utils.h"

#include "convolution_arg.h"
#include "fully_connected_arg.h"

#include <boost/optional.hpp>

#include <vector>

namespace cldnn
{

//this class is used for both static and dynamic reordering of data withing network.
//static reordering is done for cldnn::data (i.e. immutable) primitives via internal network 
//  - its done once before network build by running reorder in separate network and fetching its result.
//dynamic reordering is done for cldnn::input_layout (i.e. unknown data during network building)
//  - its done by inserting extra reorder into target topology.
//
//this class does not choose whether there's a need for static or dynamic optimization.
//it's programmers responsiblity to choose between 'get_reorder', which creates reorder to best format
//for given primitive (or nullptr if it's already optimal) and user shall insert it into it's own topology.
//  (note: layout_optimizer has internal caching mechanism, so if there's already reorder added for given (mem,format)
//   pair during 'get_reorder' call, it will be reused);
//or 'add_weights_for_optimization' which, beside creating the reorder, adds both primitives (data and reorder) to its
//internal network which allows later to call 'optimize' and get already reordered data to be exchanged in target topology.
class layout_optimizer
{
public:
    enum class data_type
    {
        weights,
        bias,
        input
    };

private:
    bool _enabled;
    refcounted_obj_ptr<topology_impl> _topology;
    refcounted_obj_ptr<engine_impl> _engine;
    std::vector<primitive_id> _outputs;

    layout get_expected_layout(layout const& current_layout, data_type type, std::shared_ptr<const convolution> prim, layout const& output_layout);
    layout get_expected_layout(layout const& current_layout, data_type type, std::shared_ptr<const fully_connected> prim, layout const& output_layout);

    std::shared_ptr<const cldnn::reorder> add_reorder_if_needed(const cldnn::memory& mem, const cldnn::primitive_id& memid, layout const& expected_layout);

public:
    explicit layout_optimizer(refcounted_obj_ptr<engine_impl> eng, bool enabled = true);

    //this method creates reorder for data, which is currently in 'data_layout' format, to best format in context of 'user' primitive.
    //data is used by 'user' in a way described by 'type' (i.e. weights/bias/input).
    //id shall be primitive_id of data's source (used as reorder's input and for cache checks).
    //user_layout is optional parameter (required for weights and bias, optional for input) which tells what kind of output 'user'
    //  is supposed to compute - it's used for example to decide if weights shall be converted to fp16.
    //
    //if 'data_layout' is already optimal, nullptr is returned
    //currently optimizations are supported only for convolution and fully-connected.
    template <class T>
    auto get_reorder(layout const& data_layout,
                     primitive_id const& id,
                     data_type type,
                     std::shared_ptr<const T> user,
                     boost::optional<layout> user_layout = boost::optional<layout>())
        -> std::enable_if_t<
            meta::is_any_of_v<T, convolution, fully_connected>,
            meta::deduce_ret_type_t<decltype(&layout_optimizer::add_reorder_if_needed)>
        >
    {
        if (!_enabled)
            return meta::deduce_ret_type_t<decltype(&layout_optimizer::add_reorder_if_needed)>();

        auto expected_layout = get_expected_layout(data_layout, type, prim, output_layout);
        return add_reorder_if_needed(data_layout, id, expected_layout);
    }

    //case for unsupported 'user' primitives
    template <class T>
    auto get_reorder(layout const& data_layout,
                     data_type type,
                     std::shared_ptr<const T> user,
                     boost::optional<layout> user_layout = boost::optional<layout>())
        -> std::enable_if_t<
            meta::is_any_of_v<T, convolution, fully_connected>,
            meta::deduce_ret_type_t<decltype(&layout_optimizer::add_reorder_if_needed)>
        >
    {
        static_assert(meta::always_false_v<T>, "Layout optimization for given primitive type is currently unsupported!");
        return meta::deduce_ret_type_t<decltype(&layout_optimizer::add_reorder_if_needed)>();
    }

    template <class T>
    void add_weights_for_optimization(const std::shared_ptr<const data> data_prim, weights_type type, std::shared_ptr<const T> prim, layout const& output_layout)
    {
        auto reorder = get_reorder(data_prim->mem.get_layout(), data_prim->id(), type, prim, output_layout);
        if (!reorder)
            return;

        _topology->add(data_prim, reorder);
        _outputs.push_back(reorder);
    }

    auto optimize() const -> meta::deduce_ret_type_t<decltype(&network_impl::get_primitives)>;
    auto get_engine() { return _engine; }
};
}
