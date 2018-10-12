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
#include "program_node.h"
#include "layout_optimizer.h"
#include "program_impl.h"
#include "program_helpers.h"
#include "fully_connected_inst.h"

using namespace cldnn;

//ToDo remove friendship relation from  program_node and program_impl

pre_optimize_bias::pre_optimize_bias(layout_optimizer& lo_ref) : _lo(lo_ref) {}

void pre_optimize_bias::run(program_impl &p) {
    run(p, _lo);
}

void pre_optimize_bias::run(program_impl &p, layout_optimizer& lo)
{
    //lambda function which finds weights primitive with given pimitive_id and adds it to weights_optimizer
    //this function is reused in all cases (convolution weights, convolution bias, fc weights and fc bias) and does
    //some basic sanity checks about existence of the primitive and it's type. throws std::logic_error
    const auto add_bias = [&p, &lo](program_node& bias, auto& node, layout const& output_layout, size_t dep_idx)
    {
        const auto bias_type = layout_optimizer::data_type::bias;
        auto reorder = lo.get_reorder(
            bias.get_output_layout(),
            bias.id(),
            bias_type,
            node,
            output_layout);

        if (reorder.first)
            p.add_intermediate(reorder.first, node, dep_idx, !reorder.second);
    };

    //generic lambda function which prepares given primitive for weights optimization
    //it deduces the type of weights from the type of the argument and calls 'add_weights' for all
    //weights and biases used by given primitive.
    //argument should match few requirements:
    // - it should be of a form 'typed_program_node<T>&'
    // - both 'T.weights' and 'T.bias' should be either of type 'primitive_id' or 'std::vector<primitive_id>'
    const auto prep_opt = [&p, &add_bias](auto& node) -> void
    {
        auto output_layout = node.get_output_layout();

        auto weights_offset = node.get_primitive()->input.size();
        auto bias_offset = weights_offset + program_helpers::wrap_if_single(node.get_primitive()->weights).size();
        for (auto i = bias_offset; i < node.get_dependencies().size(); ++i)
        {
            add_bias(node.get_dependency(i), node, output_layout, i);
        }
    };

    for (auto& nm : p.nodes_map)
    {
        auto& prim = *nm.second;
        if (prim.type() == convolution::type_id())
        {
            if (!prim.as<convolution>().weights_quantization_term())
                prep_opt(prim.as<convolution>());
        }
        else if (prim.type() == deconvolution::type_id())
        {
            prep_opt(prim.as<deconvolution>());
        }
        else if (prim.type() == fully_connected::type_id())
        {
            if (!prim.as<fully_connected>().weights_quantization_term())
                prep_opt(prim.as<fully_connected>());
        }
        else if (prim.type() == embed::type_id())
        {
            prep_opt(prim.as<embed>());
        }
    }
}