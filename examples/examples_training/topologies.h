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
cldnn::topology build_lenet(const std::string& weights_dir, const cldnn::engine& wo, cldnn::layout& input_layout, int32_t batch_size);

cldnn::topology build_lenet_train(const std::string& weights_dir, const cldnn::engine& wo, cldnn::layout& input_layout, int32_t batch_size, bool use_existing_weights, std::vector<cldnn::primitive_id>& outputs);

cldnn::topology build_vgg16(const std::string& weights_dir, const cldnn::engine& wo, cldnn::layout& input_layout, int32_t batch_size);

cldnn::topology build_vgg16_train(const std::string& weights_dir, const cldnn::engine& wo, cldnn::layout& input_layout, int32_t batch_size, bool use_existing_weights, std::vector<cldnn::primitive_id>& outputs);

cldnn::topology build_resnet50_train(const std::string& weights_dir, const cldnn::engine& wo, cldnn::layout& input_layout, int32_t batch_size, const bool mean_subtract, bool use_existing_weights);