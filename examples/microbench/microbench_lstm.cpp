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

#include "../common/common_tools.h"
#include "../common/file.h"
#include <cmath>
#include <string>
#include <api/CPP/input_layout.hpp>
#include <api/CPP/lstm.hpp>
#include <api/CPP/data.hpp>
#include <api/CPP/input_layout.hpp>

using namespace cldnn;

std::string getIdString(size_t i) {
    std::stringstream ss;
    ss << std::setw(5) << std::setfill('0') << i;
    return ss.str();
}

cldnn::topology build_microbench_lstm(const std::string&, const cldnn::engine& engine, const lstm_execution_params& ep, std::map<primitive_id, layout>& inputs)
{
    bool hasBias = !ep.lstm_no_biases;
    bool hasInitialHidden = ep.lstm_initial_hidden;
    bool hasInitialCell = ep.lstm_initial_cell;

    topology topology;
    auto data_type = data_types::f32;
    auto input_size = tensor(1, 1, ep.lstm_input_size, ep.lstm_batch_size);
    std::vector<primitive_id> lstm_inputs;
    for (size_t i = 0; i < ep.lstm_sequence_len; ++i) {
        primitive_id input_id = "input_" + getIdString(i);
        layout input_lay{ data_type, format::bfyx, input_size };
        inputs.insert({ input_id, input_lay });
        topology.add(input_layout(input_id, input_lay));
        lstm_inputs.push_back(input_id);
    }

    auto weight_size = tensor(1, 1, ep.lstm_input_size, 4 * ep.lstm_hidden_size);
    layout weights_lay{ data_type, format::bfyx, weight_size };
    topology.add(input_layout("weights", weights_lay));
    inputs.insert({ "weights", weights_lay });

    auto recurrent_size = tensor(1, 1, ep.lstm_hidden_size, 4 * ep.lstm_hidden_size);
    layout recurrent_lay{ data_type, format::bfyx, recurrent_size };
    topology.add(input_layout("recurrent", recurrent_lay));
    inputs.insert({ "recurrent", recurrent_lay });

    if (hasBias) {
        auto bias_size = tensor(1, 1, 4 * ep.lstm_hidden_size, 1);
        layout bias_lay{ data_type, format::bfyx, bias_size };
        topology.add(input_layout("biases", bias_lay));
        inputs.insert({ "biases", bias_lay });
    }

    if (hasInitialHidden) {
        auto hidden_size = tensor(1, 1, ep.lstm_hidden_size, ep.lstm_batch_size);
        layout hidden_lay{ data_type, format::bfyx, hidden_size };
        topology.add(input_layout("hidden", hidden_lay));
        inputs.insert({ "hidden", hidden_lay });
    }

    if (hasInitialCell) {
        auto cell_size = tensor(1, 1, ep.lstm_hidden_size, ep.lstm_batch_size);
        layout cell_lay{ data_type, format::bfyx, cell_size };
        topology.add(input_layout("cell", cell_lay));
        inputs.insert({ "cell", cell_lay });
    }

    topology.add(lstm("lstm", lstm_inputs, "weights", "recurrent",
                        hasBias ? "biases" : "", hasInitialHidden ? "hidden" : "", hasInitialCell ? "cell" : ""));
    return topology;
}
