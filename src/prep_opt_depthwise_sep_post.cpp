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


void prep_opt_depthwise_sep_post::run(program_impl &p)
{
    const auto prep_opt_depthwise_sep = [&p](auto& node) -> void
    {
        if (!node.get_depthwise_sep_opt())
            return;

        const auto& split = node.get_primitive()->split();

        auto dependency_offset = node.get_primitive()->input.size();
        //concatenate weights
        {
            //if weights were optimized it is needed to use the sizes after optimization
            auto target_layout = program_helpers::get_weights_layout(node.get_dependency(dependency_offset), split);
            program_helpers::merge_buffers(p.engine, node, target_layout, dependency_offset, dependency_offset + split);
            dependency_offset++;
        }

        //concatenate biases
        if (node.get_primitive()->bias.size() != 0)
        {
            const auto& bias_layout = node.get_dependency(dependency_offset).get_output_layout();
            auto target_layout = layout(bias_layout.data_type, cldnn::format::bfyx, { 1, 1, bias_layout.size.spatial[0] * split, 1 });
            program_helpers::merge_buffers(p.engine, node, target_layout, dependency_offset, dependency_offset + split);
            dependency_offset++;
        }

        if (node.template is_type<convolution>())
        {
            auto& prim_node = node.template as<convolution>();
            const auto& prim = prim_node.get_primitive();

            // concatenate weights quantization factors
            if (prim->weights_quantization_factors.size() != 0)
            {
                const auto& weights_quantization_layout = node.get_dependency(dependency_offset).get_output_layout();
                auto target_layout = layout(weights_quantization_layout.data_type, cldnn::format::bfyx, { 1, 1, weights_quantization_layout.size.batch[0] * split, 1 });
                program_helpers::merge_buffers(p.engine, node, target_layout, dependency_offset, dependency_offset + split);
                dependency_offset++;
            }
            // concatenate output callibration factors
            if (prim->output_calibration_factors.size() != 0)
            {
                const auto& output_callibration_layout = node.get_dependency(dependency_offset).get_output_layout();
                auto target_layout = layout(output_callibration_layout.data_type, cldnn::format::bfyx, { 1, 1, output_callibration_layout.size.batch[0] * split, 1 });
                program_helpers::merge_buffers(p.engine, node, target_layout, dependency_offset, dependency_offset + split);
                dependency_offset++;
            }
        }

        if (node.get_primitive())
            //override node split, as only one kernel will be executed
            node.set_split(1);
    };

    //depthiwise separated convolution/deconvolution optimization
    for (auto& nm : p.nodes_map)
    {
        auto& prim = *nm.second;
        program_helpers::do_for_types<deconvolution, convolution>(prim,
            prep_opt_depthwise_sep,   //case for deconvolution
            prep_opt_depthwise_sep    //case for convolution
            );
    }
}