﻿/*
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
#ifndef ELTWISE_H
#define ELTWISE_H

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

/// @brief Select mode for eltwise layer ( @CLDNN_PRIMITIVE_DESC{eltwise} ​).
typedef enum /*:int32_t*/
{
    /// @brief Eltwise sum.
    cldnn_eltwise_sum,
    /// @brief Eltwise subtract.
    cldnn_eltwise_sub,
    /// @brief Eltwise max.
    cldnn_eltwise_max,
    /// @brief Eltwise product.
    cldnn_eltwise_prod
} cldnn_eltwise_mode;

/// @brief Performs elementwise operations (sum, subtract, max or product) on two input primitives
/// Also supports built-in Relu @CLDNN_PRIMITIVE_DESC{activation} available by setting it in arguments.
/// @details Second input (input2) needs to be the same size as input.
CLDNN_BEGIN_PRIMITIVE_DESC(eltwise)
/// @brief Second input primitive id with values needed for eltwise computation.
cldnn_primitive_id input2;
/// @brief Eltwise mode. See #cldnn_eltwise_mode.
int32_t mode; /*cldnn_eltwise_mode*/
/// @brief Enables Relu activation.
uint32_t with_activation;
/// @brief Relu activation slope.
float activation_negative_slope;
CLDNN_END_PRIMITIVE_DESC(eltwise)

CLDNN_DECLARE_PRIMITIVE_TYPE_ID(eltwise);

#ifdef __cplusplus
}
#endif

/// @}
/// @}
/// @}
#endif /* ELTWISE_H */

