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
#ifndef MEAN_SUBSTRACT_H
#define MEAN_SUBSTRACT_H

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

/// @brief Subtract mean from input.
CLDNN_BEGIN_PRIMITIVE_DESC(mean_substract)
/// @brief Primitive id to get mean subtract values.
cldnn_primitive_id mean;
CLDNN_END_PRIMITIVE_DESC(mean_substract)

CLDNN_DECLARE_PRIMITIVE_TYPE_ID(mean_substract);

#ifdef __cplusplus
}
#endif

/// @}
/// @}
/// @}
#endif /* MEAN_SUBSTRACT_H */

