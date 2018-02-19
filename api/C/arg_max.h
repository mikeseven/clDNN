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
#ifndef ARG_MAX_H
#define ARG_MAX_H

#include "cldnn.h"
/// @addtogroup c_api C API
/// @{
/// @addtogroup c_topology Network Topology
/// @{
/// @addtogroup c_primitives Primitives
/// @{

#ifdef __cplusplus
extern "C" {
#endif

/// @brief Finds the index of the k max values of input.
/// Also supports built-in Relu @CLDNN_PRIMITIVE_DESC{activation} available by setting it in arguments.
CLDNN_BEGIN_PRIMITIVE_DESC(arg_max)
/// @brief Enable Relu activation.
uint32_t with_activation;
/// @brief Relu activation slope.
float activation_negative_slope;
/// @brief Number of maximal indexes to output.
uint32_t top_k;
/// @brief Enables outputing vector of pairs (maximum index, maximum value).
uint32_t output_max_value;
/// @brief Indicates that the primitive has user defined axis to maximize along;
uint32_t with_axis;
/// @brief Axis to maximize along. If not set, maximize the flattened trailing dimensions for each index of the first dimension.
uint32_t axis;
CLDNN_END_PRIMITIVE_DESC(arg_max)

CLDNN_DECLARE_PRIMITIVE_TYPE_ID(arg_max);

#ifdef __cplusplus
}
#endif

/// @}
/// @}
/// @}
#endif /* ARG_MAX.H */

