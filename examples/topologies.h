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

#include "api/topology.hpp"
#include "weights_optimizer.h"


/**
 * \brief Builds ALEXNET topology
 * \param input_layout - will be set to the layout of the "input" primitive
 * \return topology for Alexnet network where final primitive has id "output"
 */
cldnn::topology build_alexnet(const std::string& weights_dir, weights_optimizer& wo, cldnn::layout& input_layout, int32_t batch_size, bool use_half);

cldnn::topology build_vgg16(const std::string& weights_dir, weights_optimizer& wo, cldnn::layout& input_layout, int32_t batch_size, bool use_half);

cldnn::topology build_googlenetv1(const std::string& weights_dir, weights_optimizer& wo, cldnn::layout& input_layout, int32_t batch_size, bool use_half);

cldnn::topology build_gender(const std::string& weights_dir, weights_optimizer& wo, cldnn::layout& input_layout, int32_t batch_size, bool use_half);

cldnn::topology build_microbench(const std::string& weights_dir, weights_optimizer& wo, cldnn::layout& input_layout, int32_t batch_size, bool use_half);

cldnn::topology build_squeezenet(const std::string& weights_dir, weights_optimizer& wo, cldnn::layout& input_layout, int32_t batch_size, bool use_half);