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

// TODO!!! - optimize weights based on HW
class weights_optimizer
{
public:
    enum class weights_type
    {
        weights,
        bias
    };

    bool _enabled;
    refcounted_obj_ptr<topology_impl> _topology;
    refcounted_obj_ptr<engine_impl> _engine;
    std::vector<primitive_id> _outputs;

    //this functions return either mem_id directly, if mem does not need optimization, or id of a reorder which tooks mem as input and returns it in optimizied format
    layout get_expected_layout(const cldnn::memory& mem, weights_type type, std::shared_ptr<const convolution> prim, layout const& output_layout);
    layout get_expected_layout(const cldnn::memory& mem, weights_type type, std::shared_ptr<const fully_connected> prim, layout const& output_layout);

    //returns true if reorder is actually needed
    bool add_reorder_if_needed(const cldnn::memory& mem, const cldnn::primitive_id& memid, layout const& expected_layout);

public:
    explicit weights_optimizer(refcounted_obj_ptr<engine_impl> eng, bool enabled = true);

    template <class T>
    auto add_weights(const std::shared_ptr<const data> data_prim, weights_type type, std::shared_ptr<const T> prim, layout const& output_layout)
        -> std::enable_if_t<meta::is_any_of_v<T, convolution, fully_connected>>
    {
        if (!_enabled)
            return;

        auto expected_layout = get_expected_layout(data_prim->mem, type, prim, output_layout);
        if (!add_reorder_if_needed(data_prim->mem, data_prim->id(), expected_layout))
            return;

        //reoreder was added so we need to add 'data_prim' to topology so it can be used as input
        _topology->add(data_prim);
    }

    template <class T>
    auto add_weights(const std::shared_ptr<const data> data_prim, weights_type type, std::shared_ptr<const T> prim, layout const& output_layout)
        -> std::enable_if_t<!meta::is_any_of_v<T, convolution, fully_connected>>
    {
        static_assert(meta::always_false<T>::value, "Weights optimization for given primitive type is not currently supported!");
        return primitive_id();
    }

    auto optimize() const -> meta::deduce_ret_type_t<decltype(&network_impl::get_primitives)>;
    auto get_engine() { return _engine; }
};
}
