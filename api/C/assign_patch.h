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
#ifndef assign_patch_H
#define assign_patch_H

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

/// @brief Performs custom patching algorithm from mxnet
/// Also supports built-in Relu @ref activation available by setting it in arguments.
/// @notes
/// See: https://github.com/zhaw/neural_style/blob/master/fast_mrf_cnn/symbol.py
CLDNN_BEGIN_PRIMITIVE_DESC(assign_patch)
/// @brief Enables Relu activation.
uint32_t with_activation;
/// @brief Relu activation slope.
float activation_negative_slope;
CLDNN_END_PRIMITIVE_DESC(assign_patch)

CLDNN_DECLARE_PRIMITIVE_TYPE_ID(assign_patch);

#ifdef __cplusplus
}
#endif

/// @}
/// @}
/// @}
#endif /* assign_patch_H */

