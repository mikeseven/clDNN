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
#include "api/primitives/pooling.hpp"
#include "api/primitives/depth_concatenate.hpp"
#include "api/primitives/normalization.hpp"
#include "api/primitives/mean_substract.hpp"
#include "api/primitives/eltwise.hpp"
#include <set>
#include <functional>

using namespace cldnn;

namespace {
    
    //helper function for selecting function basing on the type of the given primitive
    //this is the termination case for parameter pack recurrence, see overload below for logic
    template <class... T>
    void do_for_types(std::shared_ptr<const primitive> prim)
    {
        return;
    }

    //helper function for selecting function basing on the type of the given primitive
    //this function should be explicitly given set of types and implicitly set of functions.
    //both sets should have equal size. First function will be called if type of the given primitive
    //will match first explicitly given type, second will be called if it matches second explicitly given
    //type etc.
    //Functions given as arguments should themselves take std::shared_ptr<const T> as argument
    //where T is the type that should be match if this function should be called
    //
    //example:
    // do_for_types<
    //      convolution,
    //      pooling
    //  >(primitive,
    //      [](std::shared_ptr<const convolution>){ do something if 'primitive' is a convolution },
    //      [](std::shared_ptr<const pooling>){ do something if 'primitive' as a pooling }
    //  );
    template <class T, class... RestOfT, class Func, class... RestOfFuncs>
    decltype(static_cast<void>(std::declval<Func>()(std::declval<std::shared_ptr<const T>>()))) do_for_types(
        std::shared_ptr<const primitive> prim,
        Func const& func,
        RestOfFuncs const&... rest)
    {
        if (prim->type() == T::type_id())
            func(std::static_pointer_cast<const T>(prim));
        else
            do_for_types<RestOfT...>(prim, rest...);
    }

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
        return std::array<std::reference_wrapper<T>, 1>{ std::ref(t) };
    }

    //helper function which creates single-element array if it's given anything
    //other than std::vector.
    // T const& case -> returns container which holds T const&
    template <class T>
    auto wrap_if_single(T const& t)
    {
        return std::array<std::reference_wrapper<const T>, 1>{ std::cref(t) };
    }

    //helper function which creates single-element array if it's given anything
    //other than std::vector.
    // T&& case -> returns container which holds new instance of T created by moving given param
    template <class T>
    auto wrap_if_single(T&& t)
    {
        return std::array<T, 1>{ std::move(t) };
    }

    //helper function which creates single-element array if it's given anything
    //other than std::vector.
    // std::vector case -> does not wrap, returns t as-is
    template <class T>
    auto wrap_if_single(std::vector<T> const& t)
    {
        return t;
    }

    auto replace_weights(std::shared_ptr<const convolution> prim, std::vector<primitive_id> const& weights, std::vector<primitive_id> const& bias)
    {
        return std::make_shared<const convolution>(
            prim->id(),
            prim->input().at(0),
            weights,
            bias,
            prim->input_padding(),
            prim->stride,
            prim->with_activation,
            prim->activation_negative_slope,
            prim->output_padding()
            );
    }

    auto replace_weights(std::shared_ptr<const fully_connected> prim, std::vector<primitive_id> const& weights, std::vector<primitive_id> const& bias)
    {
        if (weights.size() != 1 || bias.size() != 1)
            throw std::runtime_error("Fully connected primitive accepts exactly one weights/bias");

        return std::make_shared<const fully_connected>(
            prim->id(),
            prim->input().at(0),
            weights.at(0),
            bias.at(0),
            prim->with_activation,
            prim->activation_negative_slope,
            prim->input_padding(),
            prim->output_padding()
            );
    }

    auto replace_inputs(std::shared_ptr<const convolution> prim, std::vector<primitive_id> const& inputs)
    {
        if (inputs.size() != 1)
            throw std::runtime_error("Convolution primitive accepts exactly one input");

        return std::make_shared<const convolution>(
            prim->id(),
            inputs.at(0),
            prim->weights,
            prim->bias,
            prim->input_padding(),
            prim->stride,
            prim->with_activation,
            prim->activation_negative_slope,
            prim->output_padding()
            );
    }

    auto replace_reorder_layout(std::shared_ptr<const reorder> prim, const layout& output_layout)
    {
        if (prim->output_layout == output_layout)
            return prim;

        if (!prim->substract_per_feature.empty())
        {
            return std::make_shared<const reorder>(
                prim->id(),
                prim->input().at(0),
                output_layout,
                prim->substract_per_feature,
                prim->input_padding(),
                prim->output_padding()
                );
        }
        else
        {
            return std::make_shared<const reorder>(
                prim->id(),
                prim->input().at(0),
                output_layout,
                prim->mean,
                prim->input_padding(),
                prim->output_padding()
                );
        }
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
                cldnn::padding needed_padding(cldnn::format::yx, { filter_layout.size.spatial[0] - 1, filter_layout.size.spatial[1] - 1 });

                // convolution have only one input primitive
                primitive_id prev_id = pair.second->primitive_desc->input().at(0);
                auto prim = _topology_map.at(prev_id)->primitive_desc;

                const auto add_padding = [](cldnn::padding newpadd, auto const& current, auto&&... params)
                {
                    auto currpadd = current->output_padding().size().transform(newpadd.size().format, 0);

                    bool needs_update = false;
                    auto newsizes = newpadd.size().sizes();
                    auto const& currsizes = currpadd.sizes();
                    for (uint32_t i = 0; i < newsizes.size(); ++i)
                    {
                        if (newsizes[i] > currsizes[i])
                            needs_update = true;
                        else
                            newsizes[i] = currsizes[i]; //select max(old, new) for each component
                    }

                    if (!needs_update)
                        return current;
                        
                    return std::make_shared<std::remove_reference_t<decltype(*current.operator->())>>(
                        std::forward<std::remove_reference_t<decltype(params)>>(params)...,
                        cldnn::padding(newpadd.size().format, newsizes, newpadd.type())
                    );
                };

                // set output padding for previous primitive
                if (prim->type() == cldnn::pooling::type_id())
                {
                    const auto _pool = std::static_pointer_cast<const cldnn::pooling>(prim);
                    auto new_pool = add_padding(
                        needed_padding,
                        _pool,
                        _pool->id(),
                        _pool->input().at(0),
                        _pool->mode,
                        _pool->stride,
                        _pool->size,
                        _pool->input_padding()
                    );
                    _topology_map[_pool->id()]->replace(new_pool);
                }
                else if (prim->type() == cldnn::normalization::type_id())
                {
                    const auto _lrn = std::static_pointer_cast<const cldnn::normalization>(prim);
                    auto new_lrn = add_padding(
                        needed_padding,
                        _lrn,
                        _lrn->id(),
                        _lrn->input().at(0),
                        _lrn->size,
                        _lrn->k,
                        _lrn->alpha,
                        _lrn->beta,
                        _lrn->norm_region,
                        _lrn->input_padding()
                    );
                    _topology_map[_lrn->id()]->replace(new_lrn);
                }
                else if (prim->type() == cldnn::convolution::type_id())
                {
                    const auto _conv = std::static_pointer_cast<const cldnn::convolution>(prim);
                    auto new_conv = add_padding(
                        needed_padding,
                        _conv,
                        _conv->id(),
                        _conv->input().at(0),
                        _conv->weights,
                        _conv->bias,
                        _conv->input_padding(),
                        _conv->stride,
                        _conv->with_activation,
                        _conv->activation_negative_slope
                    );
                    _topology_map[_conv->id()]->replace(new_conv);
                }
                else if (prim->type() == cldnn::depth_concatenate::type_id())
                {
                    const auto _depth_concat = std::static_pointer_cast<const cldnn::depth_concatenate>(prim);
                    auto new_depth_concat = std::make_shared<depth_concatenate>(
                        _depth_concat->id(),
                        _depth_concat->input()
                    );
                    _topology_map[_depth_concat->id()]->replace(new_depth_concat);
                }
                else if (prim->type() == cldnn::reorder::type_id())
                {
                    const auto _reorder = std::static_pointer_cast<const cldnn::reorder>(prim);
                    if (!_reorder->substract_per_feature.empty())
                    {
                        auto new_reorder = add_padding(
                            needed_padding,
                            _reorder,
                            _reorder->id(),
                            _reorder->input().at(0),
                            _reorder->output_layout,
                            _reorder->substract_per_feature,
                            _reorder->input_padding()
                            );
                        _topology_map[_reorder->id()]->replace(new_reorder);
                    }
                    else
                    {
                        auto new_reorder = add_padding(
                            needed_padding,
                            _reorder,
                            _reorder->id(),
                            _reorder->input().at(0),
                            _reorder->output_layout,
                            _reorder->mean,
                            _reorder->input_padding()
                            );
                        _topology_map[_reorder->id()]->replace(new_reorder);
                    }
                }
                else if (prim->type() == cldnn::eltwise::type_id())
                {
                    const auto _eltwise = std::static_pointer_cast<const cldnn::eltwise>(prim);
                    auto new_eltwise = add_padding(
                        needed_padding,
                        _eltwise,
                        _eltwise->id(),
                        _eltwise->input().at(0),
                        _eltwise->input2,
                        _eltwise->mode,
                        _eltwise->input_padding()
                        );
                    _topology_map[_eltwise->id()]->primitive_desc = new_eltwise;
                }
                else
                {
                    throw std::runtime_error("want to add output padding to primitive type that does not support it yet!");
                }
            }
        }
    }
}

void cldnn::network_builder::reorder_inputs()
{
    const auto reorder_input = [this](std::shared_ptr<const convolution> conv)
    {
        std::vector<primitive_id> new_inputs;
        for (auto const& in_id : conv->input())
        {
            auto in = _topology_map[in_id]->primitive_desc;
            if (in->type() == data::type_id())
            {
                auto reorder = _lo.add_weights_for_optimization(std::static_pointer_cast<const data>(in),
                    layout_optimizer::data_type::input,
                    conv);
                add_if_new(reorder);
                new_inputs.push_back(reorder.first ? reorder.first->id() : in->id());
            }
            else if (in->type() == input_layout::type_id())
            {
                auto reorder = _lo.get_reorder(
                    std::static_pointer_cast<const input_layout>(in)->layout,
                    in->id(),
                    layout_optimizer::data_type::input,
                    conv);
                add_if_new(reorder);
                new_inputs.push_back(reorder.first ? reorder.first->id() : in->id());
            }
            else if (in->type() == reorder::type_id()) //convolution's input is a reorder
            {
                auto current = std::static_pointer_cast<const reorder>(in);
                auto current_layout = current->output_layout;
                auto opt_reorder = _lo.get_reorder(
                    current_layout,
                    current->id(),
                    layout_optimizer::data_type::input,
                    conv);

                if (opt_reorder.first) //output format is not optimal
                {
                    auto opt_layout = opt_reorder.first->output_layout;
                    auto current_input = _topology_map[current->input().at(0)]->primitive_desc;
                    auto input_layout = current_input->type()->calc_output_layout(_topology_map, current_input);
                    if (input_layout == opt_layout) //current reorder 'breaks' optimal format
                    {
                        if (current->substract_per_feature.empty() &&
                            current->mean.empty() &&
                            !current->input_padding() &&
                            !current->output_padding()) //just plain reorder
                        {
                            new_inputs.push_back(current_input->id());
                        }
                        else //change reorder's output layout
                        {
                            current = replace_reorder_layout(current, opt_layout);
                            _topology_map[current->id()]->replace(current);
                            new_inputs.push_back(current->id());
                        }
                    }
                    else //current reorder gives bad output, simply change it
                    {
                        current = replace_reorder_layout(current, opt_layout);
                        _topology_map[current->id()]->replace(current);
                        new_inputs.push_back(current->id());
                    }
                }
                else
                    new_inputs.push_back(in->id());
            }
            else if (in->type() == mean_substract::type_id())
            {
                auto current = std::static_pointer_cast<const mean_substract>(in);
                auto current_layout = current->type()->calc_output_layout(_topology_map, current);
                auto opt_reorder = _lo.get_reorder(
                    current_layout,
                    current->id(),
                    layout_optimizer::data_type::input,
                    conv);

                if (!opt_reorder.first)
                    new_inputs.push_back(current->id());
                else //not optimal, fuse mean_substract with reorder
                {
                    auto fused_reorder = std::make_shared<const reorder>(
                        current->id(),
                        current->input().at(0),
                        opt_reorder.first->output_layout,
                        current->mean
                    );

                    _topology_map[current->id()]->replace(fused_reorder);
                    new_inputs.push_back(fused_reorder->id());
                }
            }
            else
                new_inputs.push_back(in->id());
        }

        _topology_map[conv->id()]->replace(replace_inputs(conv, new_inputs));
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
        auto itr = _topology_map.find(weights_id);
        if (itr == _topology_map.end())
            throw std::logic_error("Weights primitive with id " + weights_id + " does not exist in topology map");

        auto weights_prim = itr->second->primitive_desc;
        if (weights_prim->type() == data::type_id())
        {
            return _lo.add_weights_for_optimization(std::static_pointer_cast<const data>(weights_prim), weights_type, prim, output_layout);
        }
        else if (weights_prim->type() == input_layout::type_id())
        {
            auto reorder = _lo.get_reorder(
                std::static_pointer_cast<const input_layout>(weights_prim)->layout,
                weights_id,
                weights_type,
                prim,
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

        std::vector<primitive_id> new_weights;
        std::vector<primitive_id> new_bias;

        for (auto const& w_id : wrap_if_single(prim->weights))
        {
            auto reorder = add_weights(w_id, layout_optimizer::data_type::weights, prim, output_layout);
            this->add_if_new(reorder); //gcc bug? in generic lambda we must explicitly provide 'this'
            new_weights.push_back(reorder.first ? reorder.first->id() : static_cast<primitive_id>(w_id));
        }

        for (auto const& w_id : wrap_if_single(prim->bias))
        {
            auto reorder = add_weights(w_id, layout_optimizer::data_type::bias, prim, output_layout);
            this->add_if_new(reorder);
            new_bias.push_back(reorder.first ? reorder.first->id() : static_cast<primitive_id>(w_id));
        }

        _topology_map[prim->id()]->primitive_desc = replace_weights(prim, new_weights, new_bias);
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

void network_builder::add_if_new(std::pair<std::shared_ptr<const reorder>, bool> const& reorder_from_optimizer)
{
    if (!reorder_from_optimizer.first)
        return;

    auto id = reorder_from_optimizer.first->id();
    auto itr = _topology_map.find(id);
    if (itr != _topology_map.end())
        return;

    _topology_map[id] = std::make_shared<topology_node>(reorder_from_optimizer.first);
}
