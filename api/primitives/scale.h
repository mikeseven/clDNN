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
#ifndef SCALE_H
#define SCALE_H

#include "../cldnn.h"
/// @addtogroup c_api C API
/// @{
/// @addtogroup c_topology Network Topology
/// @{
/// @addtogroup c_primitives Primitives
/// @{

#ifdef __cplusplus
extern "C" {
#endif

/// @brief Performs elementwise product of input and scale_input.
/// @details Scale input dimension should be equal to input dimension or be 1 if it is not there.
/// Input size : 2x3x4x5(BFYX)
///     Scale inputs sizes :
///     2x3x4x5 - works the same as(axis == 0 == -4) in caffe
///     1x3x4x5 - works the same as(axis == 1 == -3) in caffe
///     1x1x4x5 - works the same as(axis == 2 == -2) in caffe
///     1x1x1x5 - works the same as(axis == 3 == -1) in caffe
///     1x1x1x1 - works the same as empty shape(scalar) in caffe
/// When scale_input is the same as input, the behavior is the same as eltwise with product operation.
/// Optionally it can also add provided biases by setting bias_term.
CLDNN_BEGIN_PRIMITIVE_DESC(scale)
/// @brief Scale input primitive id with values needed for product computation.
cldnn_primitive_id scale_input;
/// @brief Flag to set optional adding biases.
bool bias_term;
/// @brief Primitive id containing bias data.
cldnn_primitive_id bias;
CLDNN_END_PRIMITIVE_DESC(scale)

CLDNN_DECLARE_PRIMITIVE_TYPE_ID(scale);

#ifdef __cplusplus
}
#endif

/// @}
/// @}
/// @}
#endif /* SCALE_H */

