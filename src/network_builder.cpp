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
#include "weights_optimizer.h"
#include "api/primitives/convolution.hpp"
#include "api/primitives/pooling.hpp"
#include "api/primitives/depth_concatenate.hpp"
#include "api/primitives/normalization.hpp"

#include <set>

using namespace cldnn;

network_builder::network_builder(refcounted_obj_ptr<engine_impl> eng, const build_options& options)
    : _engine(eng), _options(options)
{
}


network_impl* network_builder::build_network(refcounted_obj_ptr<topology_impl> tpl)
{
    assert(tpl);
    _topology_map = tpl->get_primitives();

    if (_options.get<build_option::optimize_data>())
        _optimize_weights();
        
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
                if (conv_layout.size.format != cldnn::format::bfyx || conv_layout.data_type != data_types::f32)
                {
                    continue;
                }
                if (input_desc->type() == cldnn::reorder::type_id())
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
                cldnn::padding input_padding(cldnn::format::yx, { filter_layout.size.spatial[0] - 1, filter_layout.size.spatial[1] - 1 });

                // convolution have only one input primitive
                primitive_id prev_id = pair.second->primitive_desc->get_dto()->input.at(0);
                auto prim = _topology_map.at(prev_id)->primitive_desc;

                // set output padding for previous primitive
                if (prim->type() == cldnn::pooling::type_id())
                {
                    const auto new_pool = std::static_pointer_cast<const cldnn::pooling>(prim);
                    auto _pool = std::make_shared<pooling>(
                        new_pool->id(),
                        new_pool->input().at(0),
                        new_pool->mode,
                        new_pool->stride,
                        new_pool->size,
                        new_pool->input_padding(),
                        input_padding
                        );
                    _topology_map[new_pool->id()]->primitive_desc = _pool;
                }
                else if (prim->type() == cldnn::normalization::type_id())
                {
                    const auto new_lrn = std::static_pointer_cast<const cldnn::normalization>(prim);
                    auto _lrn = std::make_shared<normalization>(
                        new_lrn->id(),
                        new_lrn->input().at(0),
                        new_lrn->size,
                        new_lrn->k,
                        new_lrn->alpha,
                        new_lrn->beta,
                        new_lrn->input_padding(),
                        input_padding
                        );
                    _topology_map[new_lrn->id()]->primitive_desc = _lrn;
                }
                else if (prim->type() == cldnn::convolution::type_id())
                {
                    const auto new_conv = std::static_pointer_cast<const cldnn::convolution>(prim);
                    auto _conv = std::make_shared<convolution>(
                        new_conv->id(),
                        new_conv->input().at(0),
                        new_conv->weights,
                        new_conv->bias,
                        new_conv->input_padding(),
                        new_conv->stride,
                        new_conv->with_activation ? true : false,
                        new_conv->negative_slope,
                        input_padding
                        );
                    _topology_map[new_conv->id()]->primitive_desc = _conv;
                }
                else if (prim->type() == cldnn::depth_concatenate::type_id())
                {
                    const auto new_depth_concat = std::static_pointer_cast<const cldnn::depth_concatenate>(prim);
                    auto _depth_concat = std::make_shared<depth_concatenate>(
                        new_depth_concat->id(),
                        new_depth_concat->input()
                        );
                    _topology_map[new_depth_concat->id()]->primitive_desc = _depth_concat;
                }
                else
                {
                    throw std::runtime_error("want to add output padding to primitive type that does not support it yet!");
                }
            }
        }
    }
}

namespace {
    /* Internal function which fills new vector with primitives, determining for each primitve in 'old' whether it needs optimization:
       - if primive is already in optimal format. it's inserted into 'optimized' as is
       - otherwise new primitive which optimize old's format is inserted
    */
    void replace_prim_to_opt(std::vector<primitive_id> const& old, std::vector<primitive_id>& optimized, weights_optimizer& wo,
                                            weights_optimizer::weights_type type, unsigned int batch_size, topology_map const& topo)
    {
        for (auto const& prim_id : old)
        {
            auto prim = topo.find(prim_id)->second;
            assert(prim->type() == data::type_id() && "Optimization of type different than cldnn::data");
            optimized.push_back(wo.add_weights(std::static_pointer_cast<const data>(prim), type, batch_size));
        }
    }
}

void network_builder::_optimize_weights()
{
    weights_optimizer wo{ _engine, true };

    for (auto& p : _topology_map)
    {
        auto& prim = p.second;

        if (prim->primitive_desc->type() == convolution::type_id())
            _prepare_for_optimization(wo, std::static_pointer_cast<const convolution>(prim));
        else if (prim->primitive_desc->type() == fully_connected::type_id())
            _prepare_for_optimization(wo, std::static_pointer_cast<const fully_connected>(prim));
    }

    //all optimizing primitives has beed added and inputs for all primitives has been updated.
    //run reorders now
    wo.optimize();
}

void cldnn::network_builder::_prepare_for_optimization(weights_optimizer& wo, std::shared_ptr<const convolution> prim)
{
    std::vector<primitive_id> new_weights;
    std::vector<primitive_id> new_bias;

    unsigned int batch_size = prim->type()->calc_output_layout(_topology_map, prim).size.batch[0];

    replace_prim_to_opt(prim->weights, new_weights, wo, weights_optimizer::weights_type::convolution, batch_size, _topology_map);
    replace_prim_to_opt(prim->bias, new_bias, wo, weights_optimizer::weights_type::bias, batch_size, _topology_map);

    _topology_map[prim->id()] = std::make_shared<cldnn::convolution>(
        prim->id(),
        prim->input().at(0),
        new_weights,
        new_bias,
        prim->input_padding(),
        prim->stride,
        prim->with_activation != 0,
        prim->negative_slope,
        prim->output_padding()
    );
}

void cldnn::network_builder::_prepare_for_optimization(weights_optimizer& wo, std::shared_ptr<const fully_connected> prim)
{
    std::vector<primitive_id> new_weights;
    std::vector<primitive_id> new_bias;

    unsigned int batch_size = prim->type()->calc_output_layout(_topology_map, prim).size.batch[0];

    replace_prim_to_opt({ prim->weights }, new_weights, wo, weights_optimizer::weights_type::fully_connected, batch_size, _topology_map);
    replace_prim_to_opt({ prim->bias }, new_bias, wo, weights_optimizer::weights_type::bias, batch_size, _topology_map);

    _topology_map[prim->id()] = std::make_shared<cldnn::fully_connected>(
        prim->id(),
        prim->input().at(0),
        new_weights.at(0),
        new_bias.at(0),
        prim->with_activation != 0,
        prim->negative_slope,
        prim->input_padding(),
        prim->output_padding()
    );
}
