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

#include "memory_impl.h"
#include "engine_impl.h"
#include "topology_impl.h"
#include "meta_utils.h"

#include "api/CPP/data.hpp"
#include "api/CPP/reorder.hpp"
#include "api/CPP/convolution.hpp"
#include "api/CPP/deconvolution.hpp"
#include "api/CPP/fully_connected.hpp"

#include "generic_layer.hpp"

#include "kernel_selector_common.h"
#include "kernel_selector_helper.h"
#include <boost/optional.hpp>

#include <vector>

namespace cldnn
{

class primitive_inst;

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
    enum class optimization_attributes_type
    {
        splitted_convolution
    };
    struct optimization_attributes
    {
        int32_t splitted_convolution = 0;
    };

private:
    bool _enabled;
    topology_impl _topology;
    engine_impl::ptr _engine;
    optimization_attributes _optimization_attributes;

    struct cache_key
    {
        primitive_id data_source;
        layout expected_layout;

        friend bool operator ==(cache_key const& lhs, cache_key const& rhs)
        {
            return lhs.data_source == rhs.data_source && lhs.expected_layout == rhs.expected_layout;
        }

        friend bool operator !=(cache_key const& lhs, cache_key const& rhs)
        {
            return !(lhs == rhs);
        }

        friend bool operator <(cache_key const& lhs, cache_key const& rhs)
        {
            if (lhs.data_source != rhs.data_source)
                return (lhs.data_source < rhs.data_source);
            return lhs.expected_layout < rhs.expected_layout;
        }
    };

    std::map<cache_key, std::shared_ptr<reorder>> _cached_reorders;
    std::map<cache_key, std::shared_ptr<generic_layer>> _cached_generic_layers;

    layout get_expected_layout(layout const& current_layout, data_type type, std::shared_ptr<const convolution> prim, boost::optional<layout> const& output_layout);
    layout get_expected_layout(layout const& current_layout, data_type type, std::shared_ptr<const fully_connected> prim, boost::optional<layout> const& output_layout);
    layout get_expected_layout(layout const& current_layout, data_type type, std::shared_ptr<const deconvolution> prim, boost::optional<layout> const& output_layout);

    //pair.first is reorder (may be nullptr if reorder is not needed), pair.second tells if returned reorder was cached (no need to add it to 'ouputs' etc.)
    //for pair.first == nullptr, pair.second == true
    std::pair<std::shared_ptr<cldnn::reorder>, bool>
    create_reorder_if_needed(const layout& current_layout, const cldnn::primitive_id& memid, layout const& expected_layout);

    std::pair<std::shared_ptr<cldnn::generic_layer>, bool>
    create_ks_reorder_if_needed(const cldnn::primitive_id& memid, layout const& expected_layout, const KernelSelector::WeightsReorderParams& reorder_params);

public:
    explicit layout_optimizer(engine_impl::ptr eng, bool enabled = true);

    //this method creates reorder for data, which is currently in 'data_layout' format, to best format in context of 'user' primitive.
    //data is used by 'user' in a way described by 'type' (i.e. weights/bias/input).
    //id shall be primitive_id of data's source (used as reorder's input and for cache checks).
    //user_layout is optional parameter (required for weights and bias, optional for input) which tells what kind of output 'user'
    //  is supposed to compute - it's used for example to decide if weights shall be converted to fp16.
    //
    //if 'data_layout' is already optimal, nullptr is returned
    //currently optimizations are supported only for convolution and fully-connected.
    //
    //returns a pair<reorder,bool> - where pair.first is a pointer to the reorder primitive and pair.second tells if it's been reused
    //from cache, pair.second == false means this is a newly created primitive and probably needs to be added to topology etc.
    template <class T>
    auto get_reorder(layout const& data_layout,
                     primitive_id const& id,
                     data_type type,
                     std::shared_ptr<const T> user,
                     boost::optional<layout> user_layout = boost::optional<layout>())
        -> std::enable_if_t<
            meta::is_any_of_v<T, convolution, fully_connected, deconvolution>,
            meta::deduce_ret_type_t<decltype(&layout_optimizer::create_reorder_if_needed)>
        >
    {
        if (!_enabled)
            return meta::deduce_ret_type_t<decltype(&layout_optimizer::create_reorder_if_needed)>();

        auto expected_layout = get_expected_layout(data_layout, type, user, user_layout);
        return create_reorder_if_needed(data_layout, id, expected_layout);
    }

    //case for unsupported 'user' primitives
    template <class T>
    auto get_reorder(layout const& data_layout,
                     primitive_id const& id,
                     data_type type,
                     std::shared_ptr<const T> user,
                     boost::optional<layout> user_layout = boost::optional<layout>())
        -> std::enable_if_t<
            !meta::is_any_of_v<T, convolution, fully_connected, deconvolution>,
            meta::deduce_ret_type_t<decltype(&layout_optimizer::create_reorder_if_needed)>
        >
    {
        static_assert(meta::always_false_v<T>, "Layout optimization for given primitive type is currently unsupported!");
        return meta::deduce_ret_type_t<decltype(&layout_optimizer::create_reorder_if_needed)>();
    }

    auto get_generic_layer(
        const KernelSelector::WeightsReorderParams& reorder_params,
        primitive_id input_id,
        const layout& old_layout,
        data_type type)
    {
        std::vector<std::pair<std::shared_ptr<primitive>, bool>> ret;

        if (reorder_params.engine != KernelSelector::WeightsReorderParams::Engine::NONE &&
            type == data_type::weights &&
            _enabled)
        {
            if (reorder_params.engine == KernelSelector::WeightsReorderParams::Engine::CPU &&
                reorder_params.cpu_kernel != nullptr)
            {
                const auto intermediate_format = weight_layput_2_tensor_format(reorder_params.cpu_kernel->GetInputLayout());
                if (intermediate_format != old_layout.format)
                {
                    const layout intermediate_layout = { old_layout.data_type, intermediate_format, old_layout.size.transform(intermediate_format, 1) };

                    auto reorder = create_reorder_if_needed(old_layout, input_id, intermediate_layout);
                    if (reorder.first)
                    {
                        ret.push_back(reorder);
                        input_id = reorder.first->id;
                    }
                }
            }

            const auto bpp = data_type_traits::size_of(old_layout.data_type);
            layout expected_layout = {
                old_layout.data_type, format::bfyx, // simple linear format (flatten to x channel)
                { 1,1,1,(tensor::value_type)(reorder_params.new_buffer_size / bpp) }
            };
            auto reorder = create_ks_reorder_if_needed(input_id, expected_layout, reorder_params);
            if (reorder.first)
            {
                ret.push_back(reorder);
            }
        }

        return std::move(ret);
    }

    template <typename T>
    void add_reoder_to_topology(T& reorder, const std::shared_ptr<data> src_data)
    {
        if (reorder.first)
        {
            _topology.add(src_data);
            if (!reorder.second) //returned reorder is a new primitive (i.e. not cached), add it to topology and as an output
            {
                _topology.add(reorder.first);
            }
        }
    }

    template <class T>
    auto add_weights_for_optimization(const std::shared_ptr<data> data_prim,
                                      data_type type,
                                      std::shared_ptr<const T> user,
                                      boost::optional<layout> user_layout = boost::optional<layout>())
    {
        auto reorder = get_reorder(data_prim->mem.get_layout(), data_prim->id, type, user, user_layout);
        add_reoder_to_topology(reorder, data_prim);
        return reorder;
    }

    void add_ks_weights_for_optimization(
        const KernelSelector::WeightsReorderParams& reorder_params,
        const std::shared_ptr<data> data_prim,
        data_type type)
    {
        auto reorders = get_generic_layer(reorder_params, data_prim->id, data_prim->mem.get_layout(), type);
        for (auto& reorder : reorders)
        {
            add_reoder_to_topology(reorder, data_prim);
        }
    }

    void set_optimization_attribute(optimization_attributes_type attribute, int32_t val)
    {
        switch (attribute)
        {
        case optimization_attributes_type::splitted_convolution:
            _optimization_attributes.splitted_convolution = val;
            break;
        default: throw std::out_of_range("unsupported layout optimization attribute");
        }
    }

    void optimize() const;
    auto get_engine() { return _engine; }
};
}
