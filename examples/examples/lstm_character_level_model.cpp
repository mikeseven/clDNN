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

#include "common_tools.h"
#include "file.h"
#include <api/CPP/input_layout.hpp>
#include <api/CPP/fully_connected.hpp>
#include <api/CPP/data.hpp>
#include <api/CPP/embed.hpp>
#include <api/CPP/lstm.hpp>
#include <api/CPP/split.hpp>
#include <api/CPP/crop.hpp>
#include <string>

using namespace cldnn;


// Building basic lstm network with loading weights & biases from file
topology build_char_level(const std::string& weights_dir, const cldnn::engine& engine, cldnn::layout& input_layout, int32_t batch_size, int32_t sequence_length)
{
    topology topo;

    //Allocate weights
    auto embed_weights = file::create({ engine, join_path(weights_dir, "EmbedLayer_0.nnd") });
    auto embed_bias = file::create({ engine, join_path(weights_dir, "EmbedLayer_1.nnd") });

    auto lstm_1_weights = file::create({ engine, join_path(weights_dir, "lstm1_0.nnd") });
    auto lstm_1_bias = file::create({ engine, join_path(weights_dir, "lstm1_1.nnd") });
    auto lstm_1_recurrent = file::create({ engine, join_path(weights_dir, "lstm1_2.nnd") });;

    auto lstm_2_weights = file::create({ engine, join_path(weights_dir, "lstm2_0.nnd") });
    auto lstm_2_bias = file::create({ engine, join_path(weights_dir, "lstm2_1.nnd") });
    auto lstm_2_recurrent = file::create({ engine, join_path(weights_dir, "lstm2_2.nnd") });;

    auto ip_1_weights = file::create({ engine, join_path(weights_dir, "ip1_0.nnd") });
    auto ip_1_bias = file::create({ engine, join_path(weights_dir, "ip1_1.nnd") });

    topo.add(
        embed_weights,
        embed_bias,
        lstm_1_weights,
        lstm_1_bias,
        lstm_1_recurrent,
        lstm_2_weights,
        lstm_2_bias,
        lstm_2_recurrent,
        ip_1_weights,
        ip_1_bias
    );

    //Prepare input to lstm_1
    input_layout.size = { batch_size, 1, 1, sequence_length };
    input_layout.format = cldnn::format::bfyx;
    std::vector<primitive_id> lstm_inputs;
    std::vector<primitive_id> output_ids_offsets;
    std::vector<std::pair<primitive_id, tensor>> input_ids_offsets;
    for (int i = 0; i < sequence_length; ++i)
    {
        input_ids_offsets.push_back({ std::to_string(i),{ 0, i, 0, 0 } });
        lstm_inputs.push_back("lstm_input_:" + std::to_string(i));
    }

    //Allocate topology primitives
    auto input = cldnn::input_layout(
        "input",
        input_layout);

    auto embed_layer = cldnn::embed(
        "embed",
        input,
        embed_weights,
        embed_bias);

    auto splited_embeded_input = cldnn::split(
        "lstm_input_",
        embed_layer,
        input_ids_offsets
    );

    auto lstm_1 = cldnn::lstm(
        "lstm_1",
        lstm_inputs,
        lstm_1_weights,
        lstm_1_recurrent,
        lstm_1_bias,
        "",
        "",
        "",
        0.0f,
		0,
		{},
		{},
		cldnn_lstm_offset_order::cldnn_lstm_offset_order_ifoz);
    
    auto lstm_2 = cldnn::lstm(
        "lstm_2",
        { lstm_1 },
        lstm_2_weights,
        lstm_2_recurrent,
        lstm_2_bias,
        "",
        "",
        "",
        0.0f,
		0,
		{},
		{},
		cldnn_lstm_offset_order::cldnn_lstm_offset_order_ifoz);

    //we care only about last hidden state (we could run time_distrubted_dense <lstm_gemm without reccurent weights>, if we would care about every hidden state)
    auto hidden_size = cldnn::tensor(
        input_layout.size.batch[0],
        lstm_2_recurrent.mem.get_layout().size.feature[0],
        lstm_2_recurrent.mem.get_layout().size.spatial[0],
        1);

    auto last_hidden_state = cldnn::crop(
        "last_hidden_state",
        lstm_2,
        hidden_size,
        {0, sequence_length-1, 0, 0}
    );

    auto ip_1 = cldnn::fully_connected(
        "output",
        last_hidden_state,
        ip_1_weights,
        ip_1_bias
    );

    topo.add(
        input,
        embed_layer,
        splited_embeded_input,
        lstm_1,
        lstm_2,
        last_hidden_state,
        ip_1
    );

    return topo;
}
