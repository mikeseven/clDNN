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
#include <gtest/gtest.h>
#include "api/CPP/memory.hpp"
#include <api/CPP/input_layout.hpp>
#include "api/CPP/lstm.hpp"
#include <api/CPP/split.hpp>
#include <api/CPP/crop.hpp>
#include <api/CPP/concatenation.hpp>
#include <api/CPP/topology.hpp>
#include <api/CPP/tensor.hpp>
#include <api/CPP/network.hpp>
#include <api/CPP/engine.hpp>
#include "test_utils/test_utils.h"
#include <api/CPP/data.hpp>
#include "instrumentation.h"
#include <boost/filesystem.hpp>

#include <sstream>
#include <iomanip>


using namespace cldnn;
using namespace tests;

#define FERROR 1E-4

namespace {
    float sigmoid(float x) {
        return 1.f / (1.f + std::exp((float)(-x)));
    }
}

// [ARIEL] TODO: use move semantics when required
template <typename T>
VVVVF<T> lstm_elt_reference(VVVVF<T> &tempGEMM, VVVVF<T> &cell, bool hasCell = true) {
    size_t hidden_size = tempGEMM[0][0][0].size() /  4;
    size_t batch_size = tempGEMM[0][0].size();
    VVVVF<T> tempOut(2, VVVF<T>(1, VVF<T>(batch_size, VF<T>(hidden_size))));

    for (size_t b = 0; b < batch_size; ++b) {
        T *ft = &tempGEMM[0][0][b][0];
        T *it = &tempGEMM[0][0][b][hidden_size];
        T *ot = &tempGEMM[0][0][b][2 * hidden_size];
        T *zt = &tempGEMM[0][0][b][3 * hidden_size];
        for (size_t h = 0; h < hidden_size; ++h) {
            T val = sigmoid(it[h]) * std::tanh((float)zt[h]);
            if (hasCell) {
                val += cell[0][0][b][h] * sigmoid(ft[h]);
            }
            tempOut[0][0][b][h] = std::tanh((float)val) * sigmoid(ot[h]);
            tempOut[1][0][b][h] = val;
        }
    }
    return tempOut;
}

template <typename T>
VVVVF<T> lstm_gemm_reference(VVVVF<T> &input, VVVVF<T> &weights, VVVVF<T> &recurrent, VVVVF<T> &bias, VVVVF<T> &hidden,
                              bool hasBias = true, bool hasHidden = true) {
    size_t input_size = input[0][0][0].size();
    size_t hidden_size = hidden[0][0][0].size();
    size_t batch_size = input[0][0].size();

    // Temporary output from GEMM operations [f, i, o, z]
    VVVVF<T> tempGEMM(1, VVVF<T>(1, VVF<T>(batch_size, VF<T>(4 * hidden_size))));
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t y = 0; y < 4 * hidden_size; ++y) {
            T res = 0;
            for (size_t x = 0; x < input_size; ++x) {
                res += (T)weights[0][0][y][x] * (T)input[0][0][b][x];
            }
            if (hasHidden) {
                for (size_t x = 0; x < hidden_size; ++x) {
                    res += (T)recurrent[0][0][y][x] * (T)hidden[0][0][b][x];
                }
            }
            if (hasBias) {
                res += (T)bias[0][0][0][y];
            }
            tempGEMM[0][0][b][y] = res;
        }
    }
    return tempGEMM;
}


template<typename T>
void print(const std::string& s, VVVVF<T> &input) {
    printf("%s -------------\n", s.c_str());
    printf("Size = [%d, %d, %d, %d]\n", (int)input.size(), (int)input[0].size(), (int)input[0][0].size(), (int)input[0][0][0].size());
    for (size_t b = 0; b < input.size(); ++b) {
        for (size_t f = 0; f < input[0].size(); ++f) {
            for (size_t y = 0; y < input[0][0].size(); ++y) {
                for (size_t x = 0; x < input[0][0][0].size(); ++x) {
                    printf("%f ", input[b][f][y][x]);
                }
                printf("\n");
            }
        }
    }
    printf("---------------------------------------\n");
}

template<typename T>
VVVVF<T> lstm_split_reference(VVVVF<T> &input, size_t idx, size_t bufferId) {
    VVVVF<T> tempOut;
    switch (idx) {
        case 0:
            tempOut = VVVVF<T>(1, VVVF<T>(input[0].size(), VVF<T>(input[0][0].size(), VF<T>(input[0][0][0].size()))));
            tempOut[0] = input[bufferId];
            break;
        case 1:
            tempOut = VVVVF<T>(1, VVVF<T>(1, VVF<T>(input[0][0].size(), VF<T>(input[0][0][0].size()))));
            tempOut[0][0] = input[0][bufferId];
            break;
        case 2:
            tempOut = VVVVF<T>(1, VVVF<T>(1, VVF<T>(1, VF<T>(input[0][0][0].size()))));
            tempOut[0][0][0] = input[0][0][bufferId];
            break;
    }
    return tempOut;
}

template <typename T>
void lstm_reference(VVVVF<T> &input, VVVVF<T> &hidden, VVVVF<T> &cell, VVVVF<T> &weights, VVVVF<T> &recurrent, VVVVF<T> &bias,
                         VVVVF<T> &output, VVVVF<T> &last_hidden, VVVVF<T> & last_cell,
                         bool hasBias = true, bool hasInitialHidden = true, bool hasInitialCell = true) {

    size_t sequence_len = input[0].size();
    for (size_t seq = 0; seq < sequence_len; ++seq) {
        VVVVF<T> splitInput = lstm_split_reference(input, 1, seq);
        VVVVF<T> tempGEMM = lstm_gemm_reference(splitInput, weights, recurrent, bias, hidden, hasBias, hasInitialHidden);
        VVVVF<T> tempOutput = lstm_elt_reference(tempGEMM, cell, hasInitialCell);
        output[seq] = tempOutput[0]; // hidden
        hidden = lstm_split_reference(tempOutput, 0, 0);
        cell = lstm_split_reference(tempOutput, 0, 1);
        hasInitialHidden = true;
        hasInitialCell = true;
    }

    last_hidden = hidden;
    last_cell = cell;
}



template<typename T>
void generic_lstm_gemm_gpu_test(int sequence_len, int direction, int batch_size, int input_size, int hidden_size,
    bool hasBias = true, bool hasHidden = true) {
    int min_random = -2, max_random = 2;

    VVVVF<T> ref_input      = generate_random_4d<T>(1, sequence_len,      batch_size,      input_size, min_random, max_random);
    VVVVF<T> ref_weights    = generate_random_4d<T>(1,    direction, 4 * hidden_size,      input_size, min_random, max_random);
    VVVVF<T> ref_recurrent  = generate_random_4d<T>(1,    direction, 4 * hidden_size,     hidden_size, min_random, max_random);
    VVVVF<T> ref_bias       = generate_random_4d<T>(1,            1,       direction, 4 * hidden_size, min_random, max_random);
    VVVVF<T> ref_hidden     = generate_random_4d<T>(1,    direction,      batch_size,     hidden_size, min_random, max_random);
    VF<T> ref_input_vec = flatten_4d<T>(cldnn::format::bfyx, ref_input);
    VF<T> ref_weights_vec = flatten_4d<T>(cldnn::format::bfyx, ref_weights);
    VF<T> ref_recurrent_vec = flatten_4d<T>(cldnn::format::bfyx, ref_recurrent);
    VF<T> ref_bias_vec = flatten_4d<T>(cldnn::format::bfyx, ref_bias);
    VF<T> ref_hidden_vec = flatten_4d<T>(cldnn::format::bfyx, ref_hidden);

    VVVVF<T> ref_output = lstm_gemm_reference(ref_input, ref_weights, ref_recurrent, ref_bias, ref_hidden, hasBias, hasHidden);

    engine engine;
    memory input      = memory::allocate(engine, { type_to_data_type<T>::value, format::bfyx, { 1, sequence_len,      input_size,     batch_size  } });
    memory weights    = memory::allocate(engine, { type_to_data_type<T>::value, format::bfyx, { 1,    direction,      input_size, 4 * hidden_size } });
    memory recurrent  = memory::allocate(engine, { type_to_data_type<T>::value, format::bfyx, { 1,    direction,     hidden_size, 4 * hidden_size } });
    memory biases     = memory::allocate(engine, { type_to_data_type<T>::value, format::bfyx, { 1,            1, 4 * hidden_size,       direction } });
    memory hidden     = memory::allocate(engine, { type_to_data_type<T>::value, format::bfyx, { 1,    direction,     hidden_size,      batch_size } });

    set_values(input, ref_input_vec);
    set_values(weights, ref_weights_vec);
    set_values(recurrent, ref_recurrent_vec);
    set_values(biases, ref_bias_vec);
    set_values(hidden, ref_hidden_vec);

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(data("weights", weights));
    topology.add(data("recurrent", recurrent));
    if (hasBias) {
        topology.add(data("biases", biases));
    }
    if (hasHidden) {
        topology.add(input_layout("hidden", hidden.get_layout()));
    }

    topology.add(lstm_gemm("lstm_gemm", "input", "weights", "recurrent", hasBias ? "biases" : "", hasHidden ? "hidden" : ""));

    network network(engine, topology);
    network.set_input_data("input", input);
    if (hasHidden) {
        network.set_input_data("hidden", hidden);
    }

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));

    auto output = outputs.begin()->second.get_memory();
    auto output_ptr = output.pointer<T>();
    int i = 0;
    for (int b = 0; b < batch_size; ++b) {
        for (int x = 0; x <  4 * hidden_size; ++x)
             EXPECT_EQ(ref_output[0][0][b][x], output_ptr[i++]);
    }
}

template<typename T>
void generic_lstm_elt_gpu_test(int sequence_len, int direction, int batch_size, int input_size, int hidden_size, bool hasCell = true) {
    // tempGEMM  = [        1, direction,           batch, 4 * hidden_size ] input
    // cell      = [        1, direction,           batch,     hidden_size ] optional
    // output    = [        2, direction,           batch,     hidden_size ] output concat[hidden, cell]
    int min_random = -2, max_random = 2;

    VVVVF<T> ref_tempGEMM   = generate_random_4d<T>(1,    direction,      batch_size, 4 * hidden_size, min_random, max_random);
    VVVVF<T> ref_cell       = generate_random_4d<T>(1,    direction,      batch_size,     hidden_size, min_random, max_random);
    VF<T> ref_tempGEMM_vec  = flatten_4d<T>(cldnn::format::bfyx, ref_tempGEMM);
    VF<T> ref_cell_vec      = flatten_4d<T>(cldnn::format::bfyx, ref_cell);

    VVVVF<T> ref_output     = lstm_elt_reference(ref_tempGEMM, ref_cell, hasCell);

    engine engine;
    memory tempGEMM  = memory::allocate(engine, { type_to_data_type<T>::value, format::bfyx, { 1,    direction, 4 * hidden_size, batch_size } });
    memory cell      = memory::allocate(engine, { type_to_data_type<T>::value, format::bfyx, { 1,    direction,     hidden_size, batch_size } });
    set_values(tempGEMM, ref_tempGEMM_vec);
    set_values(cell, ref_cell_vec);

    topology topology;
    topology.add(input_layout("tempGEMM", tempGEMM.get_layout()));
    if (hasCell) {
        topology.add(input_layout("cell", cell.get_layout()));
    }
    topology.add(lstm_elt("lstm_elt", "tempGEMM", hasCell ? "cell" : ""));

    network network(engine, topology);
    network.set_input_data("tempGEMM", tempGEMM);
    if (hasCell) {
        network.set_input_data("cell", cell);
    }

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));

    auto output = outputs.begin()->second.get_memory();
    auto output_ptr = output.pointer<T>();
    int i = 0;
    for (int j = 0; j < 2; ++j) {
        for (int b = 0; b < batch_size; ++b) {
            for (int x = 0; x <  hidden_size; ++x)
                EXPECT_NEAR(ref_output[j][0][b][x], output_ptr[i++], FERROR);
        }
    }
}

std::string getIdString(size_t i) {
    std::stringstream ss;
    ss << std::setw(5) << std::setfill('0') << i;
    return ss.str();
}


// --------------- Manually constructed LSTM ----------------------------------------
// This function manually generates an lstm node sequence by conbining lstm_gemm and lstm_elt nodes
// it requires that the output of the lstm_elt node is croped to obtain the corresponding hidden and cell outputs
void generate_lstm_topology(topology& t, memory& input, memory& hidden, memory& cell,
                            memory& weights, memory& recurrent, memory& biases, int sequence_len,
                            bool hasBias = true, bool hasInitialHidden = true, bool hasInitialCell = true) {
    auto hidden_size = hidden.get_layout().size;
    t.add(input_layout("input", input.get_layout()));
    std::vector<std::pair<primitive_id, tensor>> input_ids_offsets;
    std::vector<primitive_id> output_ids_offsets;
    for (int i = 0; i < sequence_len; ++i)
        input_ids_offsets.push_back({getIdString(i), {0, i, 0, 0}});
    t.add(split("inputSplit", "input", input_ids_offsets ));
    t.add(data("weights", weights));
    t.add(data("recurrent", recurrent));

    std::string biasStr = "";
    std::string hiddenStr = "";
    std::string cellStr = "";
    if (hasBias)
    {
        t.add(data("biases", biases));
        biasStr = "biases";
    }
    if (hasInitialHidden)
    {
        t.add(input_layout("hidden", hidden.get_layout()));
        hiddenStr = "hidden";
    }
    if (hasInitialCell)
    {
        t.add(input_layout("cell", cell.get_layout()));
        cellStr = "cell";
    }
    for (int i = 0; i < sequence_len; ++i) {
        std::string lstm_gemm_id = "lstm_gemm" + getIdString(i);
        std::string lstm_elt_id = "lstm_elt" + getIdString(i);
        std::string crop_id = "crop" + getIdString(i);

        t.add(lstm_gemm(lstm_gemm_id, "inputSplit:" + getIdString(i), "weights", "recurrent", biasStr, hiddenStr));
        t.add(lstm_elt(lstm_elt_id, lstm_gemm_id, cellStr));

        hiddenStr = crop_id + ":hidden";
        t.add(crop(hiddenStr, lstm_elt_id, hidden_size, tensor{ 0,0,0,0 }));
        if (i < sequence_len - 1) {
            cellStr = crop_id + ":cell";
            t.add(crop(cellStr, lstm_elt_id, hidden_size, tensor{ 1,0,0,0 }));
        }
        output_ids_offsets.push_back(hiddenStr);
    }
    t.add(concatenation("concatenation", output_ids_offsets, concatenation::along_b));
}


template<typename T>
void generic_lstm_custom_gpu_test(int sequence_len, int direction, int batch_size,int input_size, int hidden_size,
                                   bool hasBias = true, bool hasInitialHidden = true, bool hasInitialCell = true) {
    std::cout << "Input Size = " << input_size << " Hidden Size = " << hidden_size << " Sequence Len = " << sequence_len << " Batch Size = " << batch_size << std::endl;
    int min_random = -2, max_random = 2;
    VVVVF<T> ref_input      = generate_random_4d<T>(1, sequence_len,      batch_size,      input_size, min_random, max_random);
    VVVVF<T> ref_weights    = generate_random_4d<T>(1,    direction, 4 * hidden_size,      input_size, min_random, max_random);
    VVVVF<T> ref_recurrent  = generate_random_4d<T>(1,    direction, 4 * hidden_size,     hidden_size, min_random, max_random);
    VVVVF<T> ref_bias       = generate_random_4d<T>(1,            1,       direction, 4 * hidden_size, min_random, max_random);
    VVVVF<T> ref_hidden     = generate_random_4d<T>(1,    direction,      batch_size,     hidden_size, min_random, max_random);
    VVVVF<T> ref_cell       = generate_random_4d<T>(1,    direction,      batch_size,     hidden_size, min_random, max_random);
    VVVVF<T> ref_output(sequence_len, VVVF<T>(direction, VVF<T>(batch_size, VF<T>(hidden_size))));
    VVVVF<T> last_hidden(1, VVVF<T>(direction, VVF<T>(batch_size, VF<T>(hidden_size))));
    VVVVF<T> last_cell(1, VVVF<T>(direction, VVF<T>(batch_size, VF<T>(hidden_size))));

    VF<T> ref_input_vec     = flatten_4d<T>(cldnn::format::bfyx, ref_input);
    VF<T> ref_weights_vec   = flatten_4d<T>(cldnn::format::bfyx, ref_weights);
    VF<T> ref_recurrent_vec = flatten_4d<T>(cldnn::format::bfyx, ref_recurrent);
    VF<T> ref_bias_vec      = flatten_4d<T>(cldnn::format::bfyx, ref_bias);
    VF<T> ref_hidden_vec    = flatten_4d<T>(cldnn::format::bfyx, ref_hidden);
    VF<T> ref_cell_vec      = flatten_4d<T>(cldnn::format::bfyx, ref_cell);
    lstm_reference(ref_input, ref_hidden, ref_cell, ref_weights, ref_recurrent, ref_bias, ref_output, last_hidden, last_cell,
                    hasBias, hasInitialHidden, hasInitialCell);

    engine engine;
    memory input      = memory::allocate(engine, { type_to_data_type<T>::value, format::bfyx, { 1, sequence_len,      input_size,      batch_size } });
    memory weights    = memory::allocate(engine, { type_to_data_type<T>::value, format::bfyx, { 1,    direction,      input_size, 4 * hidden_size } });
    memory recurrent  = memory::allocate(engine, { type_to_data_type<T>::value, format::bfyx, { 1,    direction,     hidden_size, 4 * hidden_size } });
    memory biases     = memory::allocate(engine, { type_to_data_type<T>::value, format::bfyx, { 1,            1, 4 * hidden_size,       direction } });
    memory hidden     = memory::allocate(engine, { type_to_data_type<T>::value, format::bfyx, { 1,    direction,     hidden_size,      batch_size } });
    memory cell       = memory::allocate(engine, { type_to_data_type<T>::value, format::bfyx, { 1,    direction,     hidden_size,      batch_size } });
    set_values(input, ref_input_vec);
    set_values(weights, ref_weights_vec);
    set_values(recurrent, ref_recurrent_vec);
    set_values(biases, ref_bias_vec);
    set_values(hidden, ref_hidden_vec);
    set_values(cell, ref_cell_vec);

    topology topology;
    generate_lstm_topology(topology, input, hidden, cell, weights, recurrent, biases, sequence_len,
                            hasBias, hasInitialHidden, hasInitialCell);

    network network(engine, topology);
    network.set_input_data("input", input);
    if (hasInitialHidden) network.set_input_data("hidden", hidden);
    if (hasInitialCell) network.set_input_data("cell", cell);
    auto outputs = network.execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    size_t output_size = outputs.begin()->second.get_memory().size() / sizeof(T);
    ASSERT_EQ(output_size, size_t(hidden_size * sequence_len * batch_size * direction));

    auto output = outputs.begin()->second.get_memory();
    auto output_ptr = output.pointer<T>();
    int i = 0;
    for (int j = 0; j < sequence_len; ++j) {
        for (int b = 0; b < batch_size; ++b) {
            for (int x = 0; x <  hidden_size; ++x) {
                ASSERT_NEAR(ref_output[j][0][b][x], output_ptr[i++], FERROR);
            }
        }
    }
}

// -------------------------------------------------------

template<typename T>
void generic_lstm_gpu_test(int sequence_len, int direction, int batch_size, int input_size, int hidden_size,
                            bool hasBias = true, bool hasInitialHidden = true, bool hasInitialCell = true) {
    std::cout << "Input Size = " << input_size << " Hidden Size = " << hidden_size << " Sequence Len = " << sequence_len << " Batch Size = " << batch_size << std::endl;
    int min_random = -2, max_random = 2;
    VVVVF<T> ref_input      = generate_random_4d<T>(1, sequence_len,      batch_size,      input_size, min_random, max_random);
    VVVVF<T> ref_weights    = generate_random_4d<T>(1,    direction, 4 * hidden_size,      input_size, min_random, max_random);
    VVVVF<T> ref_recurrent  = generate_random_4d<T>(1,    direction, 4 * hidden_size,     hidden_size, min_random, max_random);
    VVVVF<T> ref_bias       = generate_random_4d<T>(1,            1,       direction, 4 * hidden_size, min_random, max_random);
    VVVVF<T> ref_hidden     = generate_random_4d<T>(1,    direction,      batch_size,     hidden_size, min_random, max_random);
    VVVVF<T> ref_cell       = generate_random_4d<T>(1,    direction,      batch_size,     hidden_size, min_random, max_random);
    VVVVF<T> ref_output(sequence_len, VVVF<T>(direction, VVF<T>(batch_size, VF<T>(hidden_size))));
    VVVVF<T> last_hidden(1, VVVF<T>(direction, VVF<T>(batch_size, VF<T>(hidden_size))));
    VVVVF<T> last_cell(1, VVVF<T>(direction, VVF<T>(batch_size, VF<T>(hidden_size))));

    VF<T> ref_input_vec     = flatten_4d<T>(cldnn::format::bfyx, ref_input);
    VF<T> ref_weights_vec   = flatten_4d<T>(cldnn::format::bfyx, ref_weights);
    VF<T> ref_recurrent_vec = flatten_4d<T>(cldnn::format::bfyx, ref_recurrent);
    VF<T> ref_bias_vec      = flatten_4d<T>(cldnn::format::bfyx, ref_bias);
    VF<T> ref_hidden_vec    = flatten_4d<T>(cldnn::format::bfyx, ref_hidden);
    VF<T> ref_cell_vec      = flatten_4d<T>(cldnn::format::bfyx, ref_cell);
    lstm_reference(ref_input, ref_hidden, ref_cell, ref_weights, ref_recurrent, ref_bias, ref_output, last_hidden, last_cell,
                    hasBias, hasInitialHidden, hasInitialCell);

    engine engine;

    memory input      = memory::allocate(engine, { type_to_data_type<T>::value, format::bfyx, { 1, sequence_len,      input_size,      batch_size } });
    memory weights    = memory::allocate(engine, { type_to_data_type<T>::value, format::bfyx, { 1,    direction,      input_size, 4 * hidden_size } });
    memory recurrent  = memory::allocate(engine, { type_to_data_type<T>::value, format::bfyx, { 1,    direction,     hidden_size, 4 * hidden_size } });
    memory biases     = memory::allocate(engine, { type_to_data_type<T>::value, format::bfyx, { 1,            1, 4 * hidden_size,       direction } });
    memory hidden     = memory::allocate(engine, { type_to_data_type<T>::value, format::bfyx, { 1,    direction,     hidden_size,      batch_size } });
    memory cell       = memory::allocate(engine, { type_to_data_type<T>::value, format::bfyx, { 1,    direction,     hidden_size,      batch_size } });

    set_values(input, ref_input_vec);
    set_values(weights, ref_weights_vec);
    set_values(recurrent, ref_recurrent_vec);
    if (hasBias) set_values(biases, ref_bias_vec);
    if (hasInitialHidden) set_values(hidden, ref_hidden_vec);
    if (hasInitialCell) set_values(cell, ref_cell_vec);

    topology topology;
    std::vector<std::pair<primitive_id, tensor>> input_ids_offsets;
    std::vector<primitive_id> lstm_inputs;
    std::vector<primitive_id> output_ids_offsets;

    topology.add(input_layout("input", input.get_layout()));
    for (int i = 0; i < sequence_len; ++i) {
        input_ids_offsets.push_back({getIdString(i), {0, i, 0, 0}});
        lstm_inputs.push_back("inputSplit:"+getIdString(i));
    }
    topology.add(split("inputSplit", "input", input_ids_offsets ));
    topology.add(data("weights", weights));
    topology.add(data("recurrent", recurrent));
    if (hasBias) topology.add(data("biases", biases));
    if (hasInitialHidden) topology.add(input_layout("hidden", hidden.get_layout()));
    if (hasInitialCell) topology.add(input_layout("cell", cell.get_layout()));
    topology.add(lstm("lstm", lstm_inputs, "weights", "recurrent",
                        hasBias ? "biases" : "", hasInitialHidden ? "hidden" : "", hasInitialCell ? "cell" : ""));

    network network(engine, topology);
    network.set_input_data("input", input);
    if (hasInitialHidden) network.set_input_data("hidden", hidden);
    if (hasInitialCell) network.set_input_data("cell", cell);
    auto outputs = network.execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    size_t output_size = outputs.begin()->second.get_memory().size() / sizeof(T);
    ASSERT_EQ(output_size, size_t(hidden_size * sequence_len * batch_size * direction));

    auto output = outputs.begin()->second.get_memory();
    auto output_ptr = output.pointer<T>();
    int i = 0;
    for (int j = 0; j < sequence_len; ++j) {
        for (int b = 0; b < batch_size; ++b) {
            for (int x = 0; x <  hidden_size; ++x) {
                ASSERT_NEAR(ref_output[j][0][b][x], output_ptr[i++], FERROR);
            }
        }
    }
}

TEST(lstm_gemm_gpu, generic_lstm_gemm_test_f32) {
    generic_lstm_gemm_gpu_test<float>(1, 1, 3, 6, 2, true, true);
}

TEST(lstm_gemm_gpu, generic_lstm_gemm_no_bias_f32) {
    generic_lstm_gemm_gpu_test<float>(1, 1, 3, 6, 2, false, true);
}

TEST(lstm_gemm_gpu, generic_lstm_gemm_no_hidden_f32) {
    generic_lstm_gemm_gpu_test<float>(1, 1, 3, 6, 2, true, false);
}

TEST(lstm_gemm_gpu, generic_lstm_gemm_no_hidden_bias_f32) {
    generic_lstm_gemm_gpu_test<float>(1, 1, 3, 6, 2, false, false);
}

TEST(lstm_elt_gpu, generic_lstm_elt_test_f32) {
    generic_lstm_elt_gpu_test<float>(1, 1, 4, 6, 3, true);
}

TEST(lstm_elt_gpu, generic_lstm_elt_no_cell_f32) {
    generic_lstm_elt_gpu_test<float>(1, 1, 4, 6, 3, false);
}

TEST(lstm_custom_gpu, generic_lstm_custom_f32) {
    generic_lstm_custom_gpu_test<float>(3, 1, 3, 3, 2, true, true, true);
}

TEST(lstm_custom_gpu, generic_lstm_custom_no_biasf32) {
    generic_lstm_custom_gpu_test<float>(3, 1, 3, 3, 2, false, true, true);
}

TEST(lstm_custom_gpu, generic_lstm_custom_no_hidden_f32) {
    generic_lstm_custom_gpu_test<float>(3, 1, 3, 3, 2, true, false, true);
}

TEST(lstm_custom_gpu, generic_lstm_custom_no_bias_hidden_f32) {
    generic_lstm_custom_gpu_test<float>(3, 1, 3, 3, 2, false, false, true);
}

TEST(lstm_custom_gpu, generic_lstm_custom_no_cell_f32) {
    generic_lstm_custom_gpu_test<float>(3, 1, 3, 3, 2, true, true, false);
}

TEST(lstm_custom_gpu, generic_lstm_custom_no_bias_cell_f32) {
    generic_lstm_custom_gpu_test<float>(3, 1, 3, 3, 2, false, true, false);
}

TEST(lstm_custom_gpu, generic_lstm_custom_no_hidden_cell_f32) {
    generic_lstm_custom_gpu_test<float>(3, 1, 3, 3, 2, true, false, false);
}

TEST(lstm_custom_gpu, generic_lstm_custom_no_bias_hidden_cell_f32) {
    generic_lstm_custom_gpu_test<float>(3, 1, 3, 3, 2, false, false, false);
}

TEST(lstm_gpu, generic_lstm_f32) {
    generic_lstm_gpu_test<float>(3, 1, 3, 3, 2, true, true, true);
}

TEST(lstm_gpu, generic_lstm_no_bias_f32) {
    generic_lstm_gpu_test<float>(3, 1, 3, 3, 2, false, true, true);
}

TEST(lstm_gpu, generic_lstm_no_hidden_f32) {
    generic_lstm_gpu_test<float>(3, 1, 5, 4, 3, true, false, true);
}

TEST(lstm_gpu, generic_lstm_no_bias_hidden_f32) {
    generic_lstm_gpu_test<float>(3, 1, 5, 4, 3, false, false, true);
}

TEST(lstm_gpu, generic_lstm_no_cell_f32) {
    generic_lstm_gpu_test<float>(3, 1, 5, 4, 3, true, true, false);
}

TEST(lstm_gpu, generic_lstm_no_bias_cell_f32) {
    generic_lstm_gpu_test<float>(3, 1, 5, 4, 3, false, true, false);
}

TEST(lstm_gpu, generic_lstm_no_hidden_cell_f32) {
    generic_lstm_gpu_test<float>(3, 1, 5, 4, 3, true, false, false);
}

TEST(lstm_gpu, generic_lstm_no_bias_hidden_cell_f32) {
    generic_lstm_gpu_test<float>(3, 1, 5, 4, 3, false, false, false);
}

// TODO: Enable sampling testing once Win64 build is fixed sampling testing
// template<typename T>
// void sampling_lstm_custom_gpu_test() {
//     std::vector<std::vector<size_t>> variant = { { 1,1,1,1,1 }, { 2,1,1,3,2 }, { 101,1,3,11,7 } };
//     for (auto &e : variant)
//         generic_lstm_custom_gpu_test<T>(e[0], e[1], e[2], e[3], e[4]);
// }

// template<typename T>
// void sampling_lstm_gpu_test() {
//     std::vector<std::vector<size_t>> variant = { { 1,1,1,1,1 }, { 2,1,1,3,2 }, { 101,1,3,11,7 } };
//     for (auto &e : variant)
//         generic_lstm_gpu_test<T>(e[0], e[1], e[2], e[3], e[4]);
// }

// TEST(lstm_gpu, sampling_lstm_custom_gpu_test_f32) {
//     sampling_lstm_custom_gpu_test<float>();
// }

// TEST(lstm_gpu, sampling_lstm__gpu_test_f32) {
//     sampling_lstm_gpu_test<float>();
// }

// TODO: Add tests for the following:
// cell_clip_threshod
// coupled input_forget
// optional concatenate output
// optional last hidden
// optional last cell
// optional activation list

