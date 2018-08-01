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
#pragma once

#include "api/CPP/topology.hpp"


/**
* \brief Builds topologies
* \param input_layout - will be set to the layout of the "input" primitive
* \return topology where final primitive has id "output"
*/
cldnn::topology build_microbench_conv(const std::string& weights_dir, const cldnn::engine& wo, std::map<cldnn::primitive_id, cldnn::layout>& inputs, int32_t batch_size);

cldnn::topology build_microbench_lstm(const std::string& weights_dir, const cldnn::engine& engine, const lstm_execution_params& ep, std::map<cldnn::primitive_id, cldnn::layout>& inputs);