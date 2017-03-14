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
#include "network_builder.h"
#include "primitive_type.h"
#include "primitive_arg.h"
#include "network_impl.h"
#include "convolution_arg.h"
#include "fully_connected_arg.h"
#include "api/primitives/convolution.hpp"
#include "api/primitives/mean_substract.hpp"
#include <set>
#include <functional>

using namespace cldnn;

namespace {
    
    //helper function for selecting function basing on the type of the given primitive
    //this is the termination case for parameter pack recurrence, see overload below for logic
    template <class... T>
    void do_for_types(std::shared_ptr<const primitive>)
    {
        return;
    }

    //helper function for selecting function basing on the type of the given primitive
    //this function should be explicitly given set of types and implicitly set of functions.
    //both sets should have equal size. First function will be called if type of the given primitive
    //will match first explicitly given type, second will be called if it matches second explicitly given
    //type etc.
    //Functions given as arguments should themselves take std::shared_ptr<T> as argument
    //where T is the type that should be match if this function should be called
    //
    //example:
    // do_for_types<
    //      convolution,
    //      pooling
    //  >(primitive,
    //      [](std::shared_ptr<convolution>){ do something if 'primitive' is a convolution },
    //      [](std::shared_ptr<pooling>){ do something if 'primitive' as a pooling }
    //  );
    template <class T, class... RestOfT, class Func, class... RestOfFuncs>
    decltype(static_cast<void>(std::declval<Func>()(std::declval<std::shared_ptr<T>>()))) do_for_types(
        std::shared_ptr<primitive> prim,
        Func const& func,
        RestOfFuncs const&... rest)
    {
        if (prim->type() == T::type_id())
            func(std::static_pointer_cast<T>(prim));
        else
            do_for_types<RestOfT...>(prim, rest...);
    }

    template <class T>
    struct single_element_container
    {
        single_element_container(T& t) : elem(&t)
        {}

        auto begin() const { return single_element_container(elem); }
        auto end() const { return single_element_container(nullptr); }
        auto& operator ++() { elem = nullptr; return *this; }
        bool operator !=(single_element_container const& sec) { return elem != sec.elem; }

        decltype(auto) operator *() { return *elem; }

    private:
        single_element_container(T* t) : elem(t)
        {}

        T* elem;
    };

    //helper function which creates single-element array if it's given anything
    //other than std::vector.
    //It should be used in generic code when there's a need to force vector usage
    //in foreach loop over variable which can in one context be a vector or a scalar
    //in another.
    //example:
    // T t;
    // for (auto& string : wrap_if_single(t.dump()))
    //depending on type T, t.dump() may return either std::string or std::vector<std::string>,
    //to ensure compatibility between these cases, wrap_if_single will create single-element
    //container in case t.dump() would return plain std::string.
    //
    // T& case -> returns container which holds T&
    template <class T>
    auto wrap_if_single(T& t)
    {
        return single_element_container<T>(t);
    }

    //helper function which creates single-element array if it's given anything
    //other than std::vector.
    // T const& case -> returns container which holds T const&
    template <class T>
    auto wrap_if_single(T const& t)
    {
        return single_element_container<T const>(t);
    }

    //helper function which creates single-element array if it's given anything
    //other than std::vector.
    // T&& case -> returns container which holds new instance of T created by moving given param
    template <class T>
    auto wrap_if_single(T&& t)
    {
        static_assert(meta::always_false_v<T>, "Wrapping temporary object into single_element_container is an error (requires valid reference)");
        return single_element_container<T>(t);
    }

    //helper function which creates single-element array if it's given anything
    //other than std::vector.
    // std::vector case -> does not wrap, returns t as-is
    template <class T>
    decltype(auto) wrap_if_single(std::vector<T>& t)
    {
        return t;
    }
}

network_builder::network_builder(refcounted_obj_ptr<engine_impl> eng, const build_options& options)
    : _engine(eng), _options(options), _lo(_engine, _options.get<build_option::optimize_data>() != nullptr)
{
}


network_impl* network_builder::build_network(refcounted_obj_ptr<topology_impl> tpl)
{
    assert(tpl);
    _topology_map = tpl->get_primitives();

    if (_options.get<build_option::optimize_data>())
    {
        reorder_inputs();
        optimize_weights();
    }
        
    prepare_padding();

    optimize_topology();
    auto network_topology = refcounted_obj_ptr<topology_impl>(new topology_impl(_topology_map), false);

    auto outputs_option = _options.get<build_option_type::outputs>();
    assert(outputs_option && !outputs_option->outputs.empty());

    return new network_impl(get_engine(), network_topology, outputs_option->outputs);
}

void network_builder::optimize_topology()
{
    // TODO some optimizations aka weights reordering, fusing, etc.
    auto outputs_option = _options.get<build_option_type::outputs>();

    // in debug mode select all primitives as output
    if (_options.get<build_option::debug>())
    {
        std::vector<primitive_id> outputs;
        if (outputs_option != nullptr)
        {
            outputs = outputs_option->outputs;
        }
        for (auto& p : _topology_map)
        {
            primitive_id id = p.second->primitive_desc->id();
            //do not add 'data' primitives to the outputs list
            if (p.second->primitive_desc->type() != data::type_id())
            {
                auto it = std::find(std::begin(outputs), std::end(outputs), id);
                if (it == std::end(outputs))
                {
                    outputs.push_back(id);
                }
            }
        }

        _options.set_option(build_option::outputs(outputs));
        return;
    }

    if (outputs_option == nullptr || outputs_option->outputs.empty())
    {
        std::vector<primitive_id> outputs;
        // by default, outputs are primitives which are not inputs for others
        std::set<primitive_id> unreferenced_ids;
        for (auto& pair : _topology_map)
        {
            unreferenced_ids.insert(pair.second->primitive_desc->id());
        }
        for (auto& pair : _topology_map)
        {
            for (auto& in : pair.second->primitive_desc->dependecies())
            {
                unreferenced_ids.erase(in);
            }
        }
        std::copy(std::begin(unreferenced_ids), std::end(unreferenced_ids), std::back_inserter(outputs));
        _options.set_option(build_option::outputs(outputs));
    }
}

// Prepares output padding for primitives
// TODO: case when input primitive is used by multiple primitives
void network_builder::prepare_padding()
{
    for (auto& pair : _topology_map)
    {
        // right now we optimize only for convolution
        if (pair.second->primitive_desc->type() == cldnn::convolution::type_id())
        {
            // if dependencies are not empty, that means this is not the leaf in the graph
            if (!pair.second->primitive_desc->dependecies().empty())
            {
                auto conv = std::static_pointer_cast<const cldnn::convolution>(pair.second->primitive_desc);
                auto conv_input_id = conv->input().at(0);
                auto input_desc = _topology_map.at(conv_input_id)->primitive_desc;
                auto conv_layout = conv->type()->calc_output_layout(_topology_map, conv);

                // right now output padding optimization is only available for bfyx format and data type = float32
                if (conv_layout.size.format != cldnn::format::bfyx)
                {
                    continue;
                }

                // Calculating input padding needed for convolution
                auto filter_id = conv->weights.at(0);
                auto filter_desc = _topology_map.at(filter_id)->primitive_desc;
                layout filter_layout(data_types::f32, { format::x,{ 0 } });
                if (filter_desc->type() == data::type_id())
                {
                    filter_layout = std::static_pointer_cast<const cldnn::data>(filter_desc)->mem.get_layout();
                }
                else if (filter_desc->type() == input_layout::type_id())
                {
                    filter_layout = std::static_pointer_cast<const cldnn::input_layout>(filter_desc)->layout;
                }
                else if (filter_desc->type() == reorder::type_id())
                {
                    filter_layout = std::static_pointer_cast<const cldnn::reorder>(filter_desc)->output_layout;
                }

                // convolution have only one input primitive
                primitive_id prev_id = pair.second->primitive_desc->input().at(0);
                auto prim = _topology_map.at(prev_id)->primitive_desc;

                auto prev_prim_output_layout = prim->type()->calc_output_layout(_topology_map, prim);

                // Compute initial required paddings for primitive used as input for convolution.
                auto input_offset = conv->input_offset().transform(conv_layout.size.format, 0);
                auto stride = conv->stride.transform(cldnn::format::yx, 0);
                auto input_limit_x = input_offset.spatial[0] + (conv_layout.size.spatial[0] - 1) * stride.spatial[0] + filter_layout.size.spatial[0];
                auto input_limit_y = input_offset.spatial[1] + (conv_layout.size.spatial[1] - 1) * stride.spatial[1] + filter_layout.size.spatial[1];

                auto left_padding = std::max(-input_offset.spatial[0], 0);
                auto top_padding = std::max(-input_offset.spatial[1], 0);
                auto right_padding = std::max(input_limit_x - prev_prim_output_layout.size.spatial[0], 0);
                auto bottom_padding = std::max(input_limit_y - prev_prim_output_layout.size.spatial[1], 0);

                // Adjust right padding, so entire buffer size in X dimension is properly aligned.
                // TODO: NOTE: Will be reenabled with next check-in once heuristic for line-aligned algorithm will be added.
                //auto needed_buffer_size_x = static_cast<cldnn::tensor::value_type>(
                //    round_up_to(left_padding + prev_prim_output_layout.size.spatial[0] + right_padding, 16));
                //right_padding = needed_buffer_size_x - left_padding - prev_prim_output_layout.size.spatial[0];

                cldnn::padding needed_padding(cldnn::format::xy, { left_padding, top_padding }, { right_padding, bottom_padding });

                prim->output_padding() = padding::max(needed_padding, prim->output_padding());
            }
        }
    }
}

void cldnn::network_builder::reorder_inputs()
{
    const auto reorder_input = [this](std::shared_ptr<convolution> conv)
    {
        std::shared_ptr<const convolution> const_conv = std::static_pointer_cast<const convolution>(conv);

        for (auto& in_id : conv->input())
        {
            std::shared_ptr<primitive> in = _topology_map[in_id]->primitive_desc;
            std::pair<std::shared_ptr<reorder>, bool> new_input = { nullptr, false };

            if (in->type() == data::type_id())
            {
                new_input = _lo.add_weights_for_optimization(std::static_pointer_cast<data>(in),
                    layout_optimizer::data_type::input,
                    const_conv);
            }
            else if (in->type() == input_layout::type_id())
            {
                new_input = _lo.get_reorder(
                    std::static_pointer_cast<const input_layout>(in)->layout,
                    in->id(),
                    layout_optimizer::data_type::input,
                    const_conv);
            }
            else if (in->type() == reorder::type_id()) //convolution's input is a reorder
            {
                auto current = std::static_pointer_cast<reorder>(in);
                auto current_layout = current->output_layout;
                new_input = _lo.get_reorder(
                    current_layout,
                    current->id(),
                    layout_optimizer::data_type::input,
                    const_conv);

                if (new_input.first) //output format is not optimal
                {
                    auto opt_layout = new_input.first->output_layout;
                    auto current_input = _topology_map[current->input().at(0)]->primitive_desc;
                    auto input_layout = current_input->type()->calc_output_layout(_topology_map, current_input);
                    if (input_layout == opt_layout) //current reorder 'breaks' optimal format
                    {
                        if (current->substract_per_feature.empty() &&
                            current->mean.empty() &&
                            !current->input_padding() &&
                            !current->output_padding()) //just plain reorder
                        {
                            in_id = current_input->id();
                            new_input.first = nullptr;
                        }
                        else //change reorder's output layout
                        {
                            current->output_layout = opt_layout;
                            new_input.first = nullptr;
                        }
                    }
                    else //current reorder gives bad output, simply change it
                    {
                        current->output_layout = opt_layout;
                        new_input.first = nullptr;
                    }
                }
            }
            else if (in->type() == mean_substract::type_id())
            {
                auto current = std::static_pointer_cast<const mean_substract>(in);
                auto current_layout = current->type()->calc_output_layout(_topology_map, current);
                new_input = _lo.get_reorder(
                    current_layout,
                    current->id(),
                    layout_optimizer::data_type::input,
                    const_conv);

                if (new_input.first) //not optimal, fuse mean_substract with reorder
                {
                    new_input.first->mean = current->id();
                }
            }

            if (new_input.first)
            {
                add_if_new(new_input);
                in_id = new_input.first->id();
            }
        }
    };

    for (auto& p : _topology_map)
    {
        auto& prim = p.second;

        //there's an assumption that only convolution will take data/input_layout as input
        //exception to that rule would be a convolution which takes a reorder as input - see reoder_input above
        do_for_types<convolution>(prim->primitive_desc,
            reorder_input       //case for convolution
            );
    }
}

void network_builder::optimize_weights()
{
    //lambda function which finds weights primitive with given pimitive_id and adds it to weights_optimizer
    //this function is reused in all cases (convolution weights, convolution bias, fc weights and fc bias) and does
    //some basic sanity checks about existence of the primitive and it's type. throws std::logic_error
    const auto add_weights = [this](primitive_id const& weights_id, layout_optimizer::data_type weights_type, auto prim, layout const& output_layout)
    {
        auto const_prim = std::static_pointer_cast<std::add_const_t<std::remove_reference_t<decltype(*prim.get())>>>(prim);

        auto itr = _topology_map.find(weights_id);
        if (itr == _topology_map.end())
            throw std::logic_error("Weights primitive with id " + weights_id + " does not exist in topology map");

        auto weights_prim = itr->second->primitive_desc;
        if (weights_prim->type() == data::type_id())
        {
            return _lo.add_weights_for_optimization(std::static_pointer_cast<data>(weights_prim), weights_type,
                const_prim, output_layout);
        }
        else if (weights_prim->type() == input_layout::type_id())
        {
            auto reorder = _lo.get_reorder(
                std::static_pointer_cast<const input_layout>(weights_prim)->layout,
                weights_id,
                weights_type,
                const_prim,
                output_layout);

            return reorder;
        }
        else
            throw std::logic_error("Optimization of weights which are neither of type cldnn::data nor cldnn::input_layout!");
    };

    //generic lambda function which prepares given primitive for weights optimization
    //it deduces the type of weights from the type of the argument and calls 'add_weights' for all
    //weights and biases used by given primitive.
    //argument should match few requirements:
    // - it should be of a form 'std::shared_ptr<const T>'
    // - both 'T.weights' and 'T.bias' should be either of type 'primitive_id' or 'std::vector<primitive_id>'
    const auto prep_opt = [this, &add_weights](auto prim) -> void
    {
        auto output_layout = prim->type()->calc_output_layout(_topology_map, prim);

        for (auto& w_id : wrap_if_single(prim->weights))
        {
            auto reorder = add_weights(w_id, layout_optimizer::data_type::weights, prim, output_layout);
            if (reorder.first)
            {
                this->add_if_new(reorder);
                w_id = reorder.first->id();
            }
        }

        for (auto& b_id : wrap_if_single(prim->bias))
        {
            auto reorder = add_weights(b_id, layout_optimizer::data_type::bias, prim, output_layout);
            if (reorder.first)
            {
                this->add_if_new(reorder);
                b_id = reorder.first->id();
            }
        }
    };

    for (auto& p : _topology_map)
    {
        auto& prim = p.second;

        do_for_types<convolution, fully_connected>(prim->primitive_desc,
            prep_opt,   //case for convolution
            prep_opt    //case for fully_connected
        );
    }

    //all optimizing primitives has been added and inputs for all primitives has been updated.
    //run reorders now
    auto outputs = _lo.optimize();

    //replace weights primitives with optimized one, if required
    for (auto const& output : outputs)
    {
        _topology_map[output->id()]->primitive_desc = std::make_shared<cldnn::data>(
            output->id(),
            output->output_memory()
        );
    }
}

void network_builder::add_if_new(std::pair<std::shared_ptr<reorder>, bool> const& reorder_from_optimizer)
{
    if (!reorder_from_optimizer.first)
        return;

    auto id = reorder_from_optimizer.first->id();
    auto itr = _topology_map.find(id);
    if (itr != _topology_map.end())
        return;

    _topology_map[id] = std::make_shared<topology_node>(reorder_from_optimizer.first);
}
