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

CLDNN_BEGIN_PRIMITIVE_DESC(scale)
cldnn_primitive_id scale_input;
bool bias_term;
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

