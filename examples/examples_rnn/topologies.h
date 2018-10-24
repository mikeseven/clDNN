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
#pragma once

#include "api/CPP/topology.hpp"


/**
* \brief Builds topologies
* \param input_layout - will be set to the layout of the "input" primitive
* \return topology where final primitive has id "output"
*/
cldnn::topology build_char_level(const std::string& weights_dir, const cldnn::engine& engine, cldnn::layout& input_layout, int32_t batch_size, int32_t sequence_length);

cldnn::topology build_onmt_ger_to_eng_6_layers_encoder(const std::string& weights_dir, const cldnn::engine& engine, std::vector<cldnn::layout>& input_layouts, const std::vector<std::pair<size_t, size_t>>& seq_len_and_batch, const size_t max_seq_len, const int32_t beam_size);

cldnn::topology build_onmt_ger_to_eng_6_layers_decoder(const std::string& weights_dir, const cldnn::engine& engine, const size_t max_seq_len, const int32_t beam_size);