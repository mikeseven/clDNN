/*
// Copyright (c) 2018 Intel Corporation
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


#include "pass_manager.h"
#include "program_helpers.h"

//ToDo remove friendship relation from program_node and program_impl

post_optimize_weights::post_optimize_weights(layout_optimizer& lo_ref) : _lo(lo_ref) {}

void post_optimize_weights::run(program_impl &p) {
    run(p, _lo);
}

void post_optimize_weights::run(program_impl &p, layout_optimizer& lo)
{
    //lambda function which finds weights primitive with given pimitive_id and adds it to weights_optimizer
    //this function is reused in all cases (convolution weights, convolution bias, fc weights and fc bias) and does
    //some basic sanity checks about existence of the primitive and it's type. throws std::logic_error
    const auto add_weights = [&p, &lo](program_node const& weights, auto& node, size_t dep_idx)
    {
        auto* impl = node.get_selected_impl().get();
        auto output_layout = node.get_output_layout();
        auto& weights_node = node.get_dependency(1);
        auto weights_layout = weights_node.get_output_layout();
        const auto weights_type = layout_optimizer::data_type::weights;

        auto reorders = lo.get_generic_layer(
            impl->_weights_reorder_params,
            weights.id(),
            weights_layout,
            weights_type);

        for (auto& reorder : reorders)
        {
            //insert new generic_layer node to topology
            p.add_intermediate(reorder.first, node, dep_idx, !reorder.second);
            //set generic_layer's node output layout and implementation
            auto& g_node = node.get_dependency(dep_idx);
            g_node.get_output_layout(false);
            g_node.selected_impl = g_node.type()->choose_impl(*(p.engine), g_node);
        }
        //set the old output layout and do not invalidate users as change of weights will not affect output layout
        node.set_output_layout(output_layout, false);
    };

    //generic lambda function which prepares given primitive for weights optimization
    //it deduces the type of weights from the type of the argument and calls 'add_weights' for all
    //weights used by given primitive.
    //argument should match requirements:
    // - it should be of a form 'typed_program_node<T>&'
    // - 'T.weights' should be either of type 'primitive_id' or 'std::vector<primitive_id>'
    const auto prep_opt = [&p, &add_weights](auto& node) -> void
    {
        auto weights_offset = node.get_primitive()->input.size();
        auto bias_offset = weights_offset + program_helpers::wrap_if_single(node.get_primitive()->weights).size();
        for (auto i = weights_offset; i < bias_offset; i++)
        {
            add_weights(node.get_dependency(i), node, i);
        }
    };

    for (auto& nm : p.nodes_map)
    {
        auto& prim = *nm.second;
        if (prim.type() == convolution::type_id())
        {
            prep_opt(prim.as<convolution>());
        }
        else if (prim.type() == deconvolution::type_id())
        {
            prep_opt(prim.as<deconvolution>());
        }
        else if (prim.type() == fully_connected::type_id())
        {
            prep_opt(prim.as<fully_connected>());
        }
        //else if (prim.type() == lstm_gemm::type_id())//TODO: Enable postoptimize weights for lstm
        //{
        //    prep_opt(prim.as<lstm_gemm>()); //we should take care of weights and reccurent
        //}
    }
}