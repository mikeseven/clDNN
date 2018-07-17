/*
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
*/

///////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef FILLER_H
#define FILLER_H

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

/// @brief Enum type to specify function for weights filling.
typedef enum
{
    xavier
} cldnn_filler_type;

/// @brief Provides filled mutable data.
/// @details This primitive allows to pass data which can be written to during training. Data is filled with specified filler.
/// For example, weights and biases for scoring networks.
/// This primitive can be also set as other primitive's output. In this case the underlying buffer will be the same in weights_filler and preceding primitive.
CLDNN_BEGIN_PRIMITIVE_DESC(filler)
/// @brief Memory object which contains data.
/// @note If memory is attached by ::cldnn_attach_memory(),
/// attached buffer should be valid on ::cldnn_build_network() call.
cldnn_memory mem;

/// @brief Specifies function which will be used to fill weights.
cldnn_filler_type fill_type;

CLDNN_END_PRIMITIVE_DESC(filler)

CLDNN_DECLARE_PRIMITIVE_TYPE_ID(filler);

#ifdef __cplusplus
}
#endif

/// @}
/// @}
/// @}
#endif /* FILLER_H */

