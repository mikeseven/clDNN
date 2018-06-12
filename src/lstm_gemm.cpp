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
#include "lstm_gemm_inst.h"
#include "primitive_type_base.h"
#include "error_handler.h"
#include "json_object.h"

namespace cldnn
{
primitive_type_id lstm_gemm_type_id()
{
    static primitive_type_base<lstm_gemm> instance;
    return &instance;
}


layout lstm_gemm_inst::calc_output_layout(lstm_gemm_node const& node)
{
    auto desc = node.get_primitive();
    auto input_layout = node.input().get_output_layout();
    auto weights_layout = node.weights().get_output_layout();

    // input     = [        1,  sequence,           batch,      input_size ]
    // weights   = [        1, direction, 4 * hidden_size,      input_size ]
    // recurrent = [        1, direction, 4 * hidden_size,     hidden_size ]
    // biases    = [        1,         1,       direction, 4 * hidden_size ]
    // hidden    = [        1, direction,           batch,     hidden_size ] optional
    // tempGEMM  = [        1, direction,           batch, 4 * hidden_size ] output
    auto result = layout(weights_layout.data_type, format::bfyx, tensor(1,
                         weights_layout.size.feature[0], weights_layout.size.spatial[1], input_layout.size.spatial[1]));
    return result;
}

std::string lstm_gemm_inst::to_string(lstm_gemm_node const& node)
{
    auto desc         = node.get_primitive();
    auto node_info    = node.desc_to_json();
    auto weights_id   = desc->weights;
    auto recurrent_id = desc->recurrent;
    auto bias_id      = desc->bias != "" ? desc->bias : "no bias";
    auto hidden_id    = desc->hidden != "" ? desc->hidden : "no inital hidden";

    std::stringstream primitive_description;

    json_composite lstm_gemm_info;
    lstm_gemm_info.add("weights id", weights_id);
    lstm_gemm_info.add("recurrent id", recurrent_id);
    lstm_gemm_info.add("bias id", bias_id);
    lstm_gemm_info.add("hidden id", hidden_id);
    node_info.add("lstm gemm info", lstm_gemm_info);
    node_info.dump(primitive_description);

    return primitive_description.str();
}

lstm_gemm_inst::typed_primitive_inst(network_impl& network, lstm_gemm_node const& node)
    :parent(network, node)
{
    auto input_size = input_memory().get_layout();
    CLDNN_ERROR_NOT_PROPER_FORMAT(node.id(), "input format", input_size.format.value, "expected format", format::bfyx);
}
}
