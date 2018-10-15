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

#pragma once

#include <api/CPP/engine.hpp>
#include <api/CPP/layout.hpp>
#include <api/CPP/topology.hpp>


/// @brief Builds simple test transpose transformation of image.
cldnn::topology build_test_transpose(const std::string& weights_dir, const cldnn::engine& engine,
                                     cldnn::layout& input_layout, bool mean_subtract = false);

/// @brief Builds FNS (Fast Neural Style) instance normalization (Feed-Forward) topology
///        based on FNS-Candy.
cldnn::topology build_fns_instance_norm(const std::string& weights_dir, const cldnn::engine& engine,
                                        cldnn::layout& input_layout, bool mean_subtract = true);
