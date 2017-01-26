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
#include "api/primitives/convolution.hpp"
#include "api/primitives/pooling.hpp"
#include "api/primitives/depth_concatenate.hpp"
#include "api/primitives/normalization.hpp"

namespace cldnn
{

    network_impl* network_builder::build_network(refcounted_obj_ptr<topology_impl> tpl)
    {
        assert(tpl);
        _topology_map = tpl->get_primitives();

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

                    // set output padding for previous primitive
                    if (prim->type() == cldnn::pooling::type_id())
                    {
                        const auto _pool = std::static_pointer_cast<const cldnn::pooling>(prim);
                        auto new_pool = std::make_shared<pooling>(
                            _pool->id(),
                            _pool->input().at(0),
                            _pool->mode,
                            _pool->stride,
                            _pool->size,
                            _pool->input_padding(),
                            needed_padding
                            );
                        _topology_map[_pool->id()]->primitive_desc = new_pool;
                    }
                    else if (prim->type() == cldnn::normalization::type_id())
                    {
                        const auto _lrn = std::static_pointer_cast<const cldnn::normalization>(prim);
                        auto new_lrn = std::make_shared<normalization>(
                            _lrn->id(),
                            _lrn->input().at(0),
                            _lrn->size,
                            _lrn->k,
                            _lrn->alpha,
                            _lrn->beta,
                            _lrn->input_padding(),
                            needed_padding
                            );
                        _topology_map[_lrn->id()]->primitive_desc = new_lrn;
                    }
                    else if (prim->type() == cldnn::convolution::type_id())
                    {
                        const auto _conv = std::static_pointer_cast<const cldnn::convolution>(prim);
                        auto new_conv = std::make_shared<convolution>(
                            _conv->id(),
                            _conv->input().at(0),
                            _conv->weights,
                            _conv->bias,
                            _conv->input_padding(),
                            _conv->stride,
                            _conv->with_activation ? true : false,
                            _conv->activation_negative_slope,
                            needed_padding
                            );
                        _topology_map[_conv->id()]->primitive_desc = new_conv;
                    }
                    else if (prim->type() == cldnn::depth_concatenate::type_id())
                    {
                        const auto _depth_concat = std::static_pointer_cast<const cldnn::depth_concatenate>(prim);
                        auto new_depth_concat = std::make_shared<depth_concatenate>(
                            _depth_concat->id(),
                            _depth_concat->input()
                            );
                        _topology_map[_depth_concat->id()]->primitive_desc = new_depth_concat;
                    }
                    else if (prim->type() == cldnn::reorder::type_id())
                    {
                        const auto _reorder = std::static_pointer_cast<const cldnn::reorder>(prim);
                        if (!_reorder->substract_per_feature.empty())
                        {
                            auto new_reorder = std::make_shared<reorder>(
                                _reorder->id(),
                                _reorder->input().at(0),
                                _reorder->output_layout,
                                _reorder->substract_per_feature,
                                _reorder->input_padding(),
                                needed_padding
                                );
                            _topology_map[_reorder->id()]->primitive_desc = new_reorder;
                        }
                        else
                        {
                            auto new_reorder = std::make_shared<reorder>(
                                _reorder->id(),
                                _reorder->input().at(0),
                                _reorder->output_layout,
                                _reorder->mean,
                                _reorder->input_padding(),
                                needed_padding
                                );
                            _topology_map[_reorder->id()]->primitive_desc = new_reorder;
                        }
                    }
                    else
                    {
                        throw std::runtime_error("want to add output padding to primitive type that does not support it yet!");
                    }
                }
            }
        }
    }
}