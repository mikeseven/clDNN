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
#include <algorithm>

#include "common_tools.h"
#include "file.h"
#include <api/CPP/lstm.hpp>
#include <api/CPP/embed.hpp>
#include <api/CPP/input_layout.hpp>
#include <api/CPP/split.hpp>
#include <api/CPP/scale.hpp>
#include <api/CPP/crop.hpp>
#include <api/CPP/concatenation.hpp>
#include <api/CPP/permute.hpp>
#include <api/CPP/fully_connected.hpp>
#include <api/CPP/softmax.hpp>
#include <api/CPP/activation.hpp>
#include <api/CPP/reorder.hpp>
#include <api/CPP/arg_max_min.hpp>
#include <api/CPP/gemm.hpp>
#include <api/CPP/reshape.hpp>
#include <api/CPP/lookup_table.hpp>
#include <api/CPP/eltwise.hpp>
#include <api/CPP/broadcast.hpp>
#include <api/CPP/index_select.hpp>
#include <api/CPP/border.hpp>

using namespace cldnn;
using namespace cldnn::utils::examples;

std::string get_id_string(size_t i) {
    std::stringstream ss;
    ss << std::setw(5) << std::setfill('0') << i;
    return ss.str();
}

void build_encoder(cldnn::topology& topo_encoder, const std::string& weights_dir, const cldnn::engine& engine, std::vector<cldnn::layout>& input_layouts, const std::vector<std::pair<size_t, size_t>>& seq_len_and_batch, const size_t max_seq_len, const int32_t beam_size)
{
    auto encoder_embed_weights = file::create({ engine, join_path(weights_dir, "encoder_embed_weights.nnd") });
    topo_encoder.add(encoder_embed_weights);

    //TODO: MAKE LOOP FOR LSTMS: for(i ; i< num_layer; i++)

    auto encoder_lstm_0_weights = file::create({ engine, join_path(weights_dir, "encoder_0_weights.nnd") });
    auto encoder_lstm_0_bias = file::create({ engine, join_path(weights_dir, "encoder_0_bias.nnd") });
    auto encoder_lstm_0_recurrent = file::create({ engine, join_path(weights_dir, "encoder_0_recurrent.nnd") });
    auto encoder_lstm_0_reverse_weights = file::create({ engine, join_path(weights_dir, "encoder_0_reverse_weights.nnd") });
    auto encoder_lstm_0_reverse_bias = file::create({ engine, join_path(weights_dir, "encoder_0_reverse_bias.nnd") });
    auto encoder_lstm_0_reverse_recurrent = file::create({ engine, join_path(weights_dir, "encoder_0_reverse_recurrent.nnd") });

    topo_encoder.add(
        encoder_lstm_0_reverse_weights,
        encoder_lstm_0_reverse_bias,
        encoder_lstm_0_reverse_recurrent
    );
    topo_encoder.add(
        encoder_lstm_0_weights,
        encoder_lstm_0_bias,
        encoder_lstm_0_recurrent
    );

    auto concat_weights = cldnn::concatenation(
        "concat_encoder_lstm_0_weights",
        { encoder_lstm_0_weights, encoder_lstm_0_reverse_weights },
        cldnn::concatenation::along_f
    );
    auto concat_bias = cldnn::concatenation(
        "concat_encoder_lstm_0_bias",
        { encoder_lstm_0_bias, encoder_lstm_0_reverse_bias },
        cldnn::concatenation::along_y
    );
    auto concat_recurrent = cldnn::concatenation(
        "concat_encoder_lstm_0_recurrent",
        { encoder_lstm_0_recurrent, encoder_lstm_0_reverse_recurrent },
        cldnn::concatenation::along_f
    );

    topo_encoder.add(concat_weights, concat_bias, concat_recurrent);

    auto encoder_lstm_1_weights = file::create({ engine, join_path(weights_dir, "encoder_1_weights.nnd") });
    auto encoder_lstm_1_bias = file::create({ engine, join_path(weights_dir, "encoder_1_bias.nnd") });
    auto encoder_lstm_1_recurrent = file::create({ engine, join_path(weights_dir, "encoder_1_recurrent.nnd") });
    auto encoder_lstm_1_reverse_weights = file::create({ engine, join_path(weights_dir, "encoder_1_reverse_weights.nnd") });
    auto encoder_lstm_1_reverse_bias = file::create({ engine, join_path(weights_dir, "encoder_1_reverse_bias.nnd") });
    auto encoder_lstm_1_reverse_recurrent = file::create({ engine, join_path(weights_dir, "encoder_1_reverse_recurrent.nnd") });

    topo_encoder.add(
        encoder_lstm_1_reverse_weights,
        encoder_lstm_1_reverse_bias,
        encoder_lstm_1_reverse_recurrent
    );
    topo_encoder.add(
        encoder_lstm_1_weights,
        encoder_lstm_1_bias,
        encoder_lstm_1_recurrent
    );

    auto concat_weights_1 = cldnn::concatenation(
        "concat_encoder_lstm_1_weights",
        { encoder_lstm_1_weights, encoder_lstm_1_reverse_weights },
        cldnn::concatenation::along_f
    );
    auto concat_bias_1 = cldnn::concatenation(
        "concat_encoder_lstm_1_bias",
        { encoder_lstm_1_bias, encoder_lstm_1_reverse_bias },
        cldnn::concatenation::along_y
    );
    auto concat_recurrent_1 = cldnn::concatenation(
        "concat_encoder_lstm_1_recurrent",
        { encoder_lstm_1_recurrent, encoder_lstm_1_reverse_recurrent },
        cldnn::concatenation::along_f
    );

    topo_encoder.add(concat_weights_1, concat_bias_1, concat_recurrent_1);

    for (size_t i = 0; i < seq_len_and_batch.size(); i++)
    {
        auto& input_layout = input_layouts.at(i);
        auto sequence_length = static_cast<int32_t>(seq_len_and_batch.at(i).second);
        auto batch_size = static_cast<int32_t>(seq_len_and_batch.at(i).first);
        std::string prefix = "b_" + std::to_string(batch_size) + "_s_" + std::to_string(sequence_length) + "_";
    //Prepare input to lstm
        input_layout.size = { batch_size, 1, sequence_length, 1 }; 
        input_layout.format = cldnn::format::bfyx;
        std::vector<primitive_id> lstm_inputs;
        std::vector<std::pair<primitive_id, tensor>> input_ids_offsets;
        for (int i = 0; i < sequence_length; ++i)
        {
            input_ids_offsets.push_back({ std::to_string(i),{ 0, i, 0, 0 } });
            lstm_inputs.push_back(prefix + "lstm_input_:" + std::to_string(i));
        }

        auto input = cldnn::input_layout(
            prefix + "input",
            input_layout
        );

        auto input_embedding = cldnn::embed(
            prefix + "input_embedding",
            input,
            encoder_embed_weights
        );

        auto splited_embeded_input = cldnn::split(
            prefix + "lstm_input_",
            input_embedding,
            input_ids_offsets
        );

        topo_encoder.add(
            splited_embeded_input,
            input_embedding,
            input
        );

        //used for unrolling lstms
        cldnn::primitive_id hidden_name = "";
        cldnn::primitive_id cell_name = "";

        //prepare first forward LSTM
        auto encoder_lstm_0_hidden_size = cldnn::tensor(
            input_layout.size.batch[0],
            encoder_lstm_0_recurrent.mem.get_layout().size.feature[0],
            encoder_lstm_0_recurrent.mem.get_layout().size.spatial[0],
            1);
        std::vector<cldnn::primitive_id> enc_0_forward;
        std::vector<cldnn::primitive_id> enc_0_forward_cell;
        for (size_t i = 0; i < sequence_length; ++i) {
            std::string lstm_gemm_id = prefix + "lstm_gemm" + get_id_string(i);
            std::string lstm_elt_id = prefix + "lstm_elt" + get_id_string(i);
            std::string crop_id = prefix + "crop" + get_id_string(i);

            topo_encoder.add(lstm_gemm(lstm_gemm_id, lstm_inputs.at(i), concat_weights, concat_recurrent, concat_bias, hidden_name));
            topo_encoder.add(lstm_elt(lstm_elt_id, lstm_gemm_id, cell_name, 0.0f, false, {}, {}, cldnn_lstm_offset_order::cldnn_lstm_offset_order_izof));

            hidden_name = prefix + crop_id + ":hidden";
            topo_encoder.add(crop(hidden_name, lstm_elt_id, encoder_lstm_0_hidden_size, tensor{ 0,0,0,0 }));
            cell_name = prefix + crop_id + ":cell";
            topo_encoder.add(crop(cell_name, lstm_elt_id, encoder_lstm_0_hidden_size, tensor{ 0,1,0,0 }));
            enc_0_forward.push_back(hidden_name);
            enc_0_forward_cell.push_back(cell_name);
        }

        //prepare first backward LSTM
        hidden_name = "";
        cell_name = "";
        std::vector<cldnn::primitive_id> enc_0_backward;
        std::vector<cldnn::primitive_id> enc_0_backward_cell;
        for (size_t i = 0; i < sequence_length; ++i) {
            std::string lstm_gemm_id = prefix + "lstm_gemm_back" + get_id_string(i);
            std::string lstm_elt_id = prefix + "lstm_elt_back" + get_id_string(i);
            std::string crop_id = prefix + "crop_back" + get_id_string(i);

            topo_encoder.add(lstm_gemm(lstm_gemm_id, lstm_inputs.at(sequence_length - i - 1), concat_weights, concat_recurrent, concat_bias, hidden_name, (uint32_t)1));
            topo_encoder.add(lstm_elt(lstm_elt_id, lstm_gemm_id, cell_name, 0.0f, false, {}, {}, cldnn_lstm_offset_order::cldnn_lstm_offset_order_izof, (uint32_t)1));

            hidden_name = prefix + crop_id + ":hidden";
            topo_encoder.add(crop(hidden_name, lstm_elt_id, encoder_lstm_0_hidden_size, tensor{ 0,0,0,0 }));
            cell_name = prefix + crop_id + ":cell";
            topo_encoder.add(crop(cell_name, lstm_elt_id, encoder_lstm_0_hidden_size, tensor{ 0,1,0,0 }));
            enc_0_backward.push_back(hidden_name);
            enc_0_backward_cell.push_back(cell_name);
        }

        for (size_t i = 0; i < sequence_length; ++i) {
            topo_encoder.add(cldnn::concatenation(
                prefix + "encoder_0_lstm_concated_hidden_states" + get_id_string(i),
                { enc_0_forward.at(i), enc_0_backward.at(sequence_length - i - 1) },
                cldnn::concatenation::along_x
            ));
        };

        //prepare second forward LSTM
        hidden_name = "";
        cell_name = "";
        auto encoder_lstm_1_hidden_size = cldnn::tensor(
            input_layout.size.batch[0],
            encoder_lstm_1_recurrent.mem.get_layout().size.feature[0],
            encoder_lstm_1_recurrent.mem.get_layout().size.spatial[0],
            1);
        std::vector<cldnn::primitive_id> enc_1_forward;
        std::vector<cldnn::primitive_id> enc_1_forward_cell;
        for (size_t i = 0; i < sequence_length; ++i) {
            cldnn::primitive_id input_id = prefix + "encoder_0_lstm_concated_hidden_states" + get_id_string(i);
            std::string lstm_gemm_id = prefix + "1_lstm_gemm" + get_id_string(i);
            std::string lstm_elt_id = prefix + "1_lstm_elt" + get_id_string(i);
            std::string crop_id = prefix + "1_crop" + get_id_string(i);

            topo_encoder.add(lstm_gemm(lstm_gemm_id, input_id, concat_weights_1, concat_recurrent_1, concat_bias_1, hidden_name));
            topo_encoder.add(lstm_elt(lstm_elt_id, lstm_gemm_id, cell_name, 0.0f, false, {}, {}, cldnn_lstm_offset_order::cldnn_lstm_offset_order_izof));

            hidden_name = prefix + crop_id + ":hidden";
            topo_encoder.add(crop(hidden_name, lstm_elt_id, encoder_lstm_1_hidden_size, tensor{ 0,0,0,0 }));
            cell_name = prefix + crop_id + ":cell";
            topo_encoder.add(crop(cell_name, lstm_elt_id, encoder_lstm_1_hidden_size, tensor{ 0,1,0,0 }));
            enc_1_forward.push_back(hidden_name);
            enc_1_forward_cell.push_back(cell_name);

        }

        //prepare second backward LSTM
        hidden_name = "";
        cell_name = "";
        std::vector<cldnn::primitive_id> enc_1_backward;
        std::vector<cldnn::primitive_id> enc_1_backward_cell;
        for (size_t i = 0; i < sequence_length; ++i) {
            cldnn::primitive_id input_id = prefix + "encoder_0_lstm_concated_hidden_states" + get_id_string(sequence_length - i - 1);
            std::string lstm_gemm_id = prefix + "1_lstm_gemm_back" + get_id_string(i);
            std::string lstm_elt_id = prefix + "1_lstm_elt_back" + get_id_string(i);
            std::string crop_id = prefix + "1_crop_back" + get_id_string(i);

            topo_encoder.add(lstm_gemm(lstm_gemm_id, input_id, concat_weights_1, concat_recurrent_1, concat_bias_1, hidden_name, (uint32_t)1));
            topo_encoder.add(lstm_elt(lstm_elt_id, lstm_gemm_id, cell_name, 0.0f, false, {}, {}, cldnn_lstm_offset_order::cldnn_lstm_offset_order_izof, (uint32_t)1));

            hidden_name = prefix + crop_id + ":hidden";
            topo_encoder.add(crop(hidden_name, lstm_elt_id, encoder_lstm_1_hidden_size, tensor{ 0,0,0,0 }));
            cell_name = prefix + crop_id + ":cell";
            topo_encoder.add(crop(cell_name, lstm_elt_id, encoder_lstm_1_hidden_size, tensor{ 0,1,0,0 }));
            enc_1_backward.push_back(hidden_name);
            enc_1_backward_cell.push_back(cell_name);
        }

        // GET NAMES OF LAST LSTM's STATES
        cldnn::primitive_id encoder_lstm_0_forward_last_hidden_state = enc_0_forward.at(sequence_length - 1);
        cldnn::primitive_id encoder_lstm_0_forward_last_cell_state = enc_0_forward_cell.at(sequence_length - 1);

        cldnn::primitive_id encoder_lstm_0_backward_last_hidden_state = enc_0_backward.at(sequence_length - 1);
        cldnn::primitive_id encoder_lstm_0_backward_last_cell_state = enc_0_backward_cell.at(sequence_length - 1);

        cldnn::primitive_id encoder_lstm_1_forward_last_hidden_state = enc_1_forward.at(sequence_length - 1);
        cldnn::primitive_id encoder_lstm_1_forward_last_cell_state = enc_1_forward_cell.at(sequence_length - 1);

        cldnn::primitive_id encoder_lstm_1_backward_last_hidden_state = enc_1_backward.at(sequence_length - 1);
        cldnn::primitive_id encoder_lstm_1_backward_last_cell_state = enc_1_backward_cell.at(sequence_length - 1);

        //prepare inital hidden and inital cell for decoder
        //auto cropped_enc_0_bck_lsh = cldnn::crop
        //(
        //    prefix + "cropped_enc_0_bck_lsh",
        //    prefix + "encoder_0_lstm_concated_hidden_states" + get_id_string(0),
        //    { input_layout.size.batch[0], 1, 250, 1 },
        //    { 0, 0, 0, 1 }
        //);
        //topo_encoder.add(cropped_enc_0_bck_lsh);

        auto concat_encoder_0_last_hiddens = cldnn::concatenation
        (
            prefix + "concat_encoder_0_last_hidden_states",
            {
                encoder_lstm_0_forward_last_hidden_state, encoder_lstm_0_backward_last_hidden_state
            },
            cldnn::concatenation::along_x
        );
        topo_encoder.add(concat_encoder_0_last_hiddens);

        std::vector<primitive_id> initial_hidden_to_decoder_0_inputs;
        std::vector<std::pair<primitive_id, tensor>> initial_hidden_to_decoder_0_offsets;
        for (int i = 0; i < input_layout.size.batch[0]; ++i)
        {
            initial_hidden_to_decoder_0_offsets.push_back({ std::to_string(i), { i, 0, 0, 0 } });
            initial_hidden_to_decoder_0_inputs.push_back(prefix + "splited_concat_encoder_0_last_hiddens_:" + std::to_string(i));
        }

        auto splited_concat_encoder_0_last_hiddens = cldnn::split(
            prefix + "splited_concat_encoder_0_last_hiddens_",
            concat_encoder_0_last_hiddens,
            initial_hidden_to_decoder_0_offsets
        );
        topo_encoder.add(splited_concat_encoder_0_last_hiddens);


        for (auto b = 0; b < input_layout.size.batch[0]; b++)
        {
            topo_encoder.add(cldnn::concatenation
            {
                prefix + "initial_hidden_to_decoder_0_batch_" + std::to_string(b),
                { static_cast<uint32_t>(beam_size), initial_hidden_to_decoder_0_inputs.at(b) },
                concatenation::along_b
            });
        }

        auto concat_encoder_1_last_hiddens = cldnn::concatenation
        (
            prefix + "concat_encoder_1_last_hidden_states",
            {
                encoder_lstm_1_forward_last_hidden_state, encoder_lstm_1_backward_last_hidden_state
            },
            cldnn::concatenation::along_x
        );
        topo_encoder.add(concat_encoder_1_last_hiddens);

        std::vector<primitive_id> initial_hidden_to_decoder_1_inputs;
        std::vector<std::pair<primitive_id, tensor>> initial_hidden_to_decoder_1_offsets;
        for (int i = 0; i < input_layout.size.batch[0]; ++i)
        {
            initial_hidden_to_decoder_1_offsets.push_back({ std::to_string(i),{ i, 0, 0, 0 } });
            initial_hidden_to_decoder_1_inputs.push_back(prefix + "splited_concat_encoder_1_last_hiddens_:" + std::to_string(i));
        }

        auto splited_concat_encoder_1_last_hiddens = cldnn::split(
            prefix + "splited_concat_encoder_1_last_hiddens_",
            concat_encoder_1_last_hiddens,
            initial_hidden_to_decoder_1_offsets
        );
        topo_encoder.add(splited_concat_encoder_1_last_hiddens);


        for (auto b = 0; b < input_layout.size.batch[0]; b++)
        {
            topo_encoder.add(cldnn::concatenation
                {
                    prefix + "initial_hidden_to_decoder_1_batch_" + std::to_string(b),
                    { static_cast<uint32_t>(beam_size), initial_hidden_to_decoder_1_inputs.at(b) },
                    concatenation::along_b
                });
        }

        //now inital cell
        auto concat_encoder_0_last_cells = cldnn::concatenation
        (
            prefix + "concat_encoder_0_last_cells",
            {
                encoder_lstm_0_forward_last_cell_state, encoder_lstm_0_backward_last_cell_state
            },
            cldnn::concatenation::along_x
        );
        topo_encoder.add(concat_encoder_0_last_cells);

        std::vector<primitive_id> initial_cell_to_decoder_0_inputs;
        std::vector<std::pair<primitive_id, tensor>> initial_cell_to_decoder_0_offsets;
        for (int i = 0; i < input_layout.size.batch[0]; ++i)
        {
            initial_cell_to_decoder_0_offsets.push_back({ std::to_string(i),{ i, 0, 0, 0 } });
            initial_cell_to_decoder_0_inputs.push_back(prefix + "splited_concat_encoder_0_last_cells_:" + std::to_string(i));
        }

        auto splited_concat_encoder_0_last_cells = cldnn::split(
            prefix + "splited_concat_encoder_0_last_cells_",
            concat_encoder_0_last_cells,
            initial_cell_to_decoder_0_offsets
        );
        topo_encoder.add(splited_concat_encoder_0_last_cells);


        for (auto b = 0; b < input_layout.size.batch[0]; b++)
        {
            topo_encoder.add(cldnn::concatenation
                {
                prefix + "initial_cell_to_decoder_0_batch_" + std::to_string(b),
                { static_cast<uint32_t>(beam_size), initial_cell_to_decoder_0_inputs.at(b) },
                concatenation::along_b
                });
        }

        auto concat_encoder_1_last_cells = cldnn::concatenation
        (
            prefix + "concat_encoder_1_last_cells",
            {
                encoder_lstm_1_forward_last_cell_state, encoder_lstm_1_backward_last_cell_state
            },
            cldnn::concatenation::along_x
        );
        topo_encoder.add(concat_encoder_1_last_cells);

        std::vector<primitive_id> initial_cell_to_decoder_1_inputs;
        std::vector<std::pair<primitive_id, tensor>> initial_cell_to_decoder_1_offsets;
        for (int i = 0; i < input_layout.size.batch[0]; ++i)
        {
            initial_cell_to_decoder_1_offsets.push_back({ std::to_string(i),{ i, 0, 0, 0 } });
            initial_cell_to_decoder_1_inputs.push_back(prefix + "splited_concat_encoder_1_last_cells_:" + std::to_string(i));
        }

        auto splited_concat_encoder_1_last_cells = cldnn::split(
            prefix + "splited_concat_encoder_1_last_cells_",
            concat_encoder_1_last_cells,
            initial_cell_to_decoder_1_offsets
        );
        topo_encoder.add(splited_concat_encoder_1_last_cells);


        for (auto b = 0; b < input_layout.size.batch[0]; b++)
        {
            topo_encoder.add(cldnn::concatenation
                {
                prefix + "initial_cell_to_decoder_1_batch_" + std::to_string(b),
                { static_cast<uint32_t>(beam_size), initial_cell_to_decoder_1_inputs.at(b) },
                concatenation::along_b
                });
        }

        std::vector<cldnn::primitive_id> encoder_hidden_states;
        for (size_t i = 0; i < sequence_length; ++i) {
            encoder_hidden_states.push_back(prefix + "encoder hidden_state" + get_id_string(i));
            topo_encoder.add(cldnn::concatenation(
                encoder_hidden_states.back(),
                { enc_1_forward.at(i), enc_1_backward.at(sequence_length - i - 1) },
                cldnn::concatenation::along_x
            ));
        };
        auto encoder_concated_all_hidden_states = cldnn::concatenation
        (
            prefix + "encoder_concated_all_hidden_states",
            encoder_hidden_states,
            cldnn::concatenation::along_f
        );
        topo_encoder.add(encoder_concated_all_hidden_states);
        
        std::vector<primitive_id> memory_bank_inputs;
        std::vector<std::pair<primitive_id, tensor>> memory_bank_offsets;
        for (int i = 0; i < input_layout.size.batch[0]; ++i)
        {
            memory_bank_offsets.push_back({ std::to_string(i),{ i, 0, 0, 0 } });
            memory_bank_inputs.push_back(prefix + "splited_memory_bank_:" + std::to_string(i));
        }

        auto splited_memory_bank = cldnn::split(
            prefix + "splited_memory_bank_",
            encoder_concated_all_hidden_states,
            memory_bank_offsets
        );
        topo_encoder.add(splited_memory_bank);

        for (auto b = 0; b < input_layout.size.batch[0]; b++)
        {
            cldnn::primitive_id mem_bank_id = prefix + "memory_bank_batch_" + std::to_string(b);
            topo_encoder.add(
                cldnn::concatenation(
                    mem_bank_id,
                    { static_cast<uint32_t>(beam_size), memory_bank_inputs.at(b) },
                    cldnn::concatenation::along_b
                ));
            //PAD THE MEMORY BANK
            topo_encoder.add(cldnn::border
            (
                "padded_" + mem_bank_id,
                mem_bank_id,
                { 0, 0, 0, 0 },
                { 0, static_cast<int32_t>(max_seq_len - input_layout.size.spatial[0]), 0, 0 },
                cldnn::border_type::zero
            ));
        }
    }
}


void build_decoder(cldnn::topology& topo_decoder, const std::string& weights_dir, const cldnn::engine& engine, const size_t max_seq_len, const int32_t beam_size)
{
    /*

    ________BEAM AND INPUT FEED________

    */
    auto decoder_embed_weights = file::create({ engine, join_path(weights_dir, "decoder_embed_weights.nnd") });
    auto padded_tokens = cldnn::input_layout
    (
        "padded_tokens",
        cldnn::layout(
            data_types::f32, format::bfyx,
            { 1, 1, beam_size, 1 })
    );
    auto embed_padded_tokens = cldnn::embed("embed_padded_tokens", padded_tokens, decoder_embed_weights);

    topo_decoder.add(
        padded_tokens,
        decoder_embed_weights,
        embed_padded_tokens);

    auto memory_bank = cldnn::input_layout
    ( 
        "memory_bank",
        cldnn::layout(
            data_types::f32, format::bfyx, 
            { beam_size, static_cast<int32_t>(max_seq_len), decoder_embed_weights.mem.get_layout().size.batch[0], 1 })
    );
    topo_decoder.add(memory_bank);

    auto input_feed = cldnn::input_layout
    ( 
        "input_feed",
        cldnn::layout(
            data_types::f32, format::bfyx, 
            { 1, beam_size, decoder_embed_weights.mem.get_layout().size.batch[0], 1 })
    );
    topo_decoder.add(input_feed);

    auto indices_to_index_select = cldnn::input_layout
    (
        "indices_to_index_select",
        cldnn::layout(
            data_types::i32, format::bfyx,
            { 1, 1, beam_size, 1 })
    );
    topo_decoder.add(indices_to_index_select);

    auto input_feed_index_selected = cldnn::index_select
    (
        "input_feed_index_selected",
        input_feed,
        indices_to_index_select,
        cldnn::index_select_axis_name::along_f
    );
    topo_decoder.add(input_feed_index_selected);

    auto input_best_scores = cldnn::input_layout
    (
        "input_best_scores",
        cldnn::layout(
            data_types::f32, format::bfyx,
            { beam_size, 1, 1, 1 })
    );
    topo_decoder.add(input_best_scores);

    auto best_scores_broadcasted = cldnn::broadcast
    (
        "best_scores_broadcasted",
        input_best_scores,
        {5, 1, 24725, 1}
    );
    topo_decoder.add(best_scores_broadcasted);

    auto concat_input_feed = cldnn::concatenation
    {
        "concated_input_feed",
        { embed_padded_tokens, input_feed_index_selected },
        concatenation::along_x
    };
    topo_decoder.add(concat_input_feed);

    auto reshaped_concated_input_feed = cldnn::permute
    (
        "reshaped_concated_input_feed",
        concat_input_feed,
        { 1, 0, 2, 3 }
    );
    topo_decoder.add(reshaped_concated_input_feed);

    /*

    ________LSTMS________

    */

    //weights for decoder LSTMS
    auto decoder_lstm_0_weights = file::create({ engine, join_path(weights_dir, "decoder_0_weights.nnd") });
    auto decoder_lstm_0_bias = file::create({ engine, join_path(weights_dir, "decoder_0_bias.nnd") });
    auto decoder_lstm_0_recurrent = file::create({ engine, join_path(weights_dir, "decoder_0_recurrent.nnd") });
    topo_decoder.add(
        decoder_lstm_0_weights,
        decoder_lstm_0_bias,
        decoder_lstm_0_recurrent
    );

    auto decoder_lstm_1_weights = file::create({ engine, join_path(weights_dir, "decoder_1_weights.nnd") });
    auto decoder_lstm_1_bias = file::create({ engine, join_path(weights_dir, "decoder_1_bias.nnd") });
    auto decoder_lstm_1_recurrent = file::create({ engine, join_path(weights_dir, "decoder_1_recurrent.nnd") });
    topo_decoder.add(
        decoder_lstm_1_weights,
        decoder_lstm_1_bias,
        decoder_lstm_1_recurrent
    );

    auto initital_hidden_0 = cldnn::input_layout
    (
        "initital_hidden_0",
        cldnn::layout(cldnn::data_types::f32, cldnn::format::bfyx, { beam_size, 1, 500, 1 })
    );
    topo_decoder.add(initital_hidden_0);

    auto initial_hidden_0_index_selected = cldnn::index_select
    (
        "initial_hidden_0_index_selected",
        initital_hidden_0,
        indices_to_index_select,
        cldnn::index_select_axis_name::along_b
    );
    topo_decoder.add(initial_hidden_0_index_selected);

    auto initital_cell_0 = cldnn::input_layout
    (
        "initital_cell_0",
        cldnn::layout(cldnn::data_types::f32, cldnn::format::bfyx, { beam_size, 1, 500, 1 })
    );
    topo_decoder.add(initital_cell_0);

    auto initial_cell_0_index_selected = cldnn::index_select
    (
        "initial_cell_0_index_selected",
        initital_cell_0,
        indices_to_index_select,
        cldnn::index_select_axis_name::along_b
    );
    topo_decoder.add(initial_cell_0_index_selected);

    auto initital_hidden_1 = cldnn::input_layout
    (
        "initital_hidden_1",
        cldnn::layout(cldnn::data_types::f32, cldnn::format::bfyx, { beam_size, 1, 500, 1 })
    );
    topo_decoder.add(initital_hidden_1);

    auto initital_hidden_1_index_selected = cldnn::index_select
    (
        "initital_hidden_1_index_selected",
        initital_hidden_1,
        indices_to_index_select,
        cldnn::index_select_axis_name::along_b
    );
    topo_decoder.add(initital_hidden_1_index_selected);

    auto initital_cell_1 = cldnn::input_layout
    (
        "initital_cell_1",
        cldnn::layout(cldnn::data_types::f32, cldnn::format::bfyx, { beam_size, 1, 500, 1 })
    );
    topo_decoder.add(initital_cell_1);

    auto initital_cell_1_index_selected = cldnn::index_select
    (
        "initial_cell_1_index_selected",
        initital_cell_1,
        indices_to_index_select,
        cldnn::index_select_axis_name::along_b
    );
    topo_decoder.add(initital_cell_1_index_selected);

    // build first decoder LSTM!!!!!!!!!
    auto deccoder_lstm_0_hidden_size = cldnn::tensor(
        beam_size,
        decoder_lstm_0_weights.mem.get_layout().size.feature[0],
        decoder_lstm_0_recurrent.mem.get_layout().size.spatial[0],
        1);
    cldnn::primitive_id hidden_name = initial_hidden_0_index_selected;
    cldnn::primitive_id cell_name = initial_cell_0_index_selected;
    cldnn::primitive_id input_to_second_lstm;
    {
        cldnn::primitive_id input_id = reshaped_concated_input_feed;
        std::string lstm_gemm_id = "dec_lstm_gemm" + get_id_string(0);
        std::string lstm_elt_id = "dec_lstm_elt" + get_id_string(0);
        std::string crop_id = "dec_crop" + get_id_string(0);

        topo_decoder.add(lstm_gemm(lstm_gemm_id, input_id, decoder_lstm_0_weights, decoder_lstm_0_recurrent, decoder_lstm_0_bias, hidden_name, (uint32_t)0));
        topo_decoder.add(lstm_elt(lstm_elt_id, lstm_gemm_id, cell_name, 0.0f, false, {}, {}, cldnn_lstm_offset_order::cldnn_lstm_offset_order_izof));

        hidden_name = crop_id + ":hidden";
        topo_decoder.add(crop(hidden_name, lstm_elt_id, deccoder_lstm_0_hidden_size, tensor{ 0,0,0,0 }));
        cell_name = crop_id + ":cell";
        topo_decoder.add(crop(cell_name, lstm_elt_id, deccoder_lstm_0_hidden_size, tensor{ 0,1,0,0 }));
        input_to_second_lstm = hidden_name;
    }

    auto deccoder_lstm_1_hidden_size = cldnn::tensor(
        beam_size,
        decoder_lstm_1_weights.mem.get_layout().size.feature[0],
        decoder_lstm_1_recurrent.mem.get_layout().size.spatial[0],
        1);
    hidden_name = initital_hidden_1_index_selected;
    cell_name = initital_cell_1_index_selected;
    {
        cldnn::primitive_id input_id = input_to_second_lstm;
        std::string lstm_gemm_id = "1_dec_lstm_gemm" + get_id_string(0);
        std::string lstm_elt_id = "1_dec_lstm_elt" + get_id_string(0);
        std::string crop_id = "1_dec_crop" + get_id_string(0);

        topo_decoder.add(lstm_gemm(lstm_gemm_id, input_id, decoder_lstm_1_weights, decoder_lstm_1_recurrent, decoder_lstm_1_bias, hidden_name, (uint32_t)0));
        topo_decoder.add(lstm_elt(lstm_elt_id, lstm_gemm_id, cell_name, 0.0f, false, {}, {}, cldnn_lstm_offset_order::cldnn_lstm_offset_order_izof));

        hidden_name = crop_id + ":hidden";
        topo_decoder.add(crop(hidden_name, lstm_elt_id, deccoder_lstm_1_hidden_size, tensor{ 0,0,0,0 }));
        cell_name = crop_id + ":cell";
        topo_decoder.add(crop(cell_name, lstm_elt_id, deccoder_lstm_1_hidden_size, tensor{ 0,1,0,0 }));
    }

    /*
    ATTENTION PART
    */
    auto attention_fc_in_weights = file::create({ engine, join_path(weights_dir, "attention_fc_in.nnd") });
    auto attention_fc_out_weights = file::create({ engine, join_path(weights_dir, "attention_fc_out.nnd") });
    topo_decoder.add(
        attention_fc_in_weights,
        attention_fc_out_weights
    );

    auto linear_in = cldnn::fully_connected
    (
        "linear_in",
        hidden_name,
        attention_fc_in_weights
    );
    topo_decoder.add(linear_in);
    
    auto reord_linear_in = cldnn::reorder
    (
        "reord_linear_in",
        linear_in,
        cldnn::format::bfyx,
        cldnn::data_types::f32
    );
    topo_decoder.add(reord_linear_in);

    auto reshape_reoder_linear_in = cldnn::reshape
    (
        "reshape_reoder_linear_in",
        reord_linear_in,
        {beam_size, 1, 500, 1}
    );
    topo_decoder.add(reshape_reoder_linear_in);

    auto permuted_memory_bank = cldnn::permute
    (
        "permuted_memory_bank",
        memory_bank,
        { 0, 2, 3, 1 }
    );
    topo_decoder.add(permuted_memory_bank);

    auto gemm_score = cldnn::gemm
    (
        "gemm_score",
        reshape_reoder_linear_in,
        permuted_memory_bank
    );
    topo_decoder.add(gemm_score);

    auto aligned_vectors = cldnn::softmax
    (
        "aligned_vectors",
        gemm_score
    );
    topo_decoder.add(aligned_vectors);

    auto permuted_memory_bank_2 = cldnn::permute
    (
        "permuted_memory_bank_2",
        memory_bank,
        { 0, 2, 1, 3 }
    );
    topo_decoder.add(permuted_memory_bank_2);

    auto context_vector = cldnn::gemm
    (
        "context_vector",
        aligned_vectors,
        permuted_memory_bank_2
    );
    topo_decoder.add(context_vector);

    auto concated_context = cldnn::concatenation
    (
        "concated_context",
        { context_vector, hidden_name },
        cldnn::concatenation::along_x
    );
    topo_decoder.add(concated_context);

    auto linear_out = cldnn::fully_connected
    (
        "linear_out",
        concated_context,
        attention_fc_out_weights
    );
    topo_decoder.add(linear_out);

    auto tanh_activation_linear_out = cldnn::activation //we also need linear_out in further calcualtions
    (
        "tanh_activation_linear_out",
        linear_out,
        activation_hyperbolic_tan
    );
    topo_decoder.add(tanh_activation_linear_out);

    auto reord_linear_out = cldnn::reorder //we need linear_out to feed lstms in next iteration
    (
        "reord_linear_out",
        tanh_activation_linear_out,
        cldnn::format::bfyx,
        cldnn::data_types::f32
    );
    topo_decoder.add(reord_linear_out);

    auto permute_linear_out = cldnn::reshape
    (
        "permute_linear_out",
        reord_linear_out,
        {1, beam_size, 500, 1}
    );
    topo_decoder.add(permute_linear_out);

    auto projection_weights = file::create({ engine, join_path(weights_dir, "generator_weights.nnd") });
    auto projections_bias = file::create({ engine, join_path(weights_dir, "generator_bias.nnd") });
    topo_decoder.add(
        projection_weights,
        projections_bias
    );

    auto projection_layer = cldnn::fully_connected
    {
        "projection_layer",
        tanh_activation_linear_out,
        projection_weights,
        projections_bias
    };
    topo_decoder.add(projection_layer);

    auto softmax_after_projection = cldnn::softmax
    (
        "softmax_after_projection",
        projection_layer
    );
    topo_decoder.add(softmax_after_projection);

    auto log_softmax = cldnn::activation
    (
        "log_softmax",
        softmax_after_projection,
        activation_log
    );
    topo_decoder.add(log_softmax);

    auto calibrated_softmax = cldnn::eltwise
    (
        "calibrated_softmax",
        log_softmax,
        best_scores_broadcasted,
        cldnn::eltwise_mode::sum
    );
    topo_decoder.add(calibrated_softmax);

    auto output = cldnn::reorder
    (
        "output",
        log_softmax,
        cldnn::format::bfyx,
        cldnn::data_types::f32
    );
    topo_decoder.add(output);

    auto arg_max = cldnn::arg_max_min
    {
        "arg_max",
        output,
        cldnn::arg_max_min::max,
        static_cast<uint32_t>(beam_size)
    };
    topo_decoder.add(arg_max);

        auto cropped_arg_max = cldnn::crop
        {
            "cropped_arg_max",
            arg_max,
            { 1, 1, beam_size,1 },
            { 0, 0, 0, 0 }
        };
        topo_decoder.add(cropped_arg_max);

        auto best_scores = cldnn::lookup_table //get real best scores of the output
        {
            "best_scores",
            output,
            cropped_arg_max
        };
        topo_decoder.add(best_scores);

        auto reshaped_best_scores = cldnn::reshape
        (
            "reshaped_best_scores",
            best_scores,
            { beam_size, 1, 1, 1 }
        );
        topo_decoder.add(reshaped_best_scores);

    //IF ITERATION NUMBER >= 1 (FIRST ITERATION GOES TO ELSE-IF BLOCK)
    
        auto flattened_output = cldnn::reshape
        (
            "flattened_output",
            calibrated_softmax,
            { 1, 1, beam_size * 24725, 1 }
        );
        topo_decoder.add(flattened_output);

        auto reord_flat_output = cldnn::reorder
        (
            "reord_flat_output",
            flattened_output,
            cldnn::format::bfyx,
            cldnn::data_types::f32
        );
        topo_decoder.add(reord_flat_output);

        auto flattened_arg_max = cldnn::arg_max_min
        {
            "flattened_arg_max",
            reord_flat_output,
            cldnn::arg_max_min::max,
            static_cast<uint32_t>(beam_size),
            cldnn::arg_max_min::x
        };
        topo_decoder.add(flattened_arg_max);

        auto flattened_best_scores = cldnn::lookup_table //get real best scores of the output
        {
            "flattened_best_scores",
            reord_flat_output,
            flattened_arg_max
        };
        topo_decoder.add(flattened_best_scores);

        auto reshaped_flattened_best_scores = cldnn::reshape
        (
            "reshaped_flattened_best_scores",
            flattened_best_scores,
            { beam_size, 1, 1, 1 }
        );
        topo_decoder.add(reshaped_flattened_best_scores);
}


// Building basic lstm network with loading weights & biases from file
topology build_onmt_ger_to_eng_6_layers_encoder(const std::string& weights_dir, const cldnn::engine& engine, std::vector<cldnn::layout>& input_layouts, const std::vector<std::pair<size_t, size_t>>& seq_len_and_batch, const size_t max_seq_len, const int32_t beam_size)
{
    topology topo;
    build_encoder(topo, weights_dir, engine, input_layouts, seq_len_and_batch, max_seq_len, beam_size);
    return topo;
}

topology build_onmt_ger_to_eng_6_layers_decoder(const std::string& weights_dir, const cldnn::engine& engine, const size_t max_seq_len, const int32_t beam_size)
{
    topology topo;
    build_decoder(topo, weights_dir, engine, max_seq_len, beam_size);
    return topo;
}

