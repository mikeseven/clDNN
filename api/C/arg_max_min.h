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
#ifndef ARG_MAX_MIN_H
#define ARG_MAX_MIN_H

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

/// @brief Enum type to specify output type - index of max or min values
typedef enum
{
	cldnn_arg_max,
	cldnn_arg_min,
} cldnn_arg_max_min_out;


/// @brief Finds the index of the k max/min values of input.
/// Also supports built-in Relu @CLDNN_PRIMITIVE_DESC{activation} available by setting it in arguments.
CLDNN_BEGIN_PRIMITIVE_DESC(arg_max_min)
/// @brief Enable Relu activation.
uint32_t with_activation;
/// @brief Relu activation slope.
float activation_negative_slope;
/// @brief Number of maximal indexes to output.
uint32_t top_k;
/// @brief Type of output - max or mix.
cldnn_arg_max_min_out output_type;
/// @brief Indicates that the primitive has user defined axis to maximize along.
uint32_t with_axis;
/// @brief Axis to maximize along. If not set, maximize the flattened trailing dimensions for each index of the first dimension.
uint32_t axis;
CLDNN_END_PRIMITIVE_DESC(arg_max_min)

CLDNN_DECLARE_PRIMITIVE_TYPE_ID(arg_max_min);

#ifdef __cplusplus
}
#endif

/// @}
/// @}
/// @}
#endif /* ARG_MAX.H */

