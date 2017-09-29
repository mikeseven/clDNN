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
#ifndef upsampling_H
#define upsampling_H

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

/// @brief Select mode for upsampling layer ( @CLDNN_PRIMITIVE_DESC{upsampling} ​).
typedef enum /*:int32_t*/
{
    /// @brief upsampling sum.
    cldnn_upsampling_nearest,
    /// @brief upsampling subtract.
    cldnn_upsampling_bilinear,
} cldnn_upsampling_sample_type;

/// @brief Performs elementwise operations (sum, subtract, max or product) on two input primitives
/// Also supports built-in Relu @CLDNN_PRIMITIVE_DESC{activation} available by setting it in arguments.
/// @notes
/// - both inputs have to have equal sizes in all dimensions
/// - format of both inputs has to be the same
CLDNN_BEGIN_PRIMITIVE_DESC(upsampling)
float scale;
uint32_t num_filter;
/// @brief upsampling mode. See #cldnn_upsampling_mode.
int32_t sample_type; /*cldnn_sample_type*/
/// @brief Enables Relu activation.
uint32_t with_activation;
/// @brief Relu activation slope.
float activation_negative_slope;
CLDNN_END_PRIMITIVE_DESC(upsampling)

CLDNN_DECLARE_PRIMITIVE_TYPE_ID(upsampling);

#ifdef __cplusplus
}
#endif

/// @}
/// @}
/// @}
#endif /* upsampling_H */

