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

///////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef INDEX_SELECT_H
#define INDEX_SELECT_H

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

typedef enum
{
    cldnn_along_b,
    cldnn_along_f,
    cldnn_along_x,
    cldnn_along_y,
} cldnn_index_select_axis;

CLDNN_BEGIN_PRIMITIVE_DESC(index_select)

cldnn_index_select_axis axis;

CLDNN_END_PRIMITIVE_DESC(index_select)


CLDNN_DECLARE_PRIMITIVE_TYPE_ID(index_select);

#ifdef __cplusplus
}
#endif

/// @}
/// @}
/// @}
#endif // INDEX_SELECT_H
