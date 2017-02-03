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
#ifndef FULLY_CONNECTED_H
#define FULLY_CONNECTED_H

#include "../cldnn.h"

#ifdef __cplusplus
extern "C" {
#endif

CLDNN_BEGIN_PRIMITIVE_DESC(fully_connected)
uint32_t with_activation;
float activation_negative_slope;
cldnn_primitive_id weights;
cldnn_primitive_id bias;
CLDNN_END_PRIMITIVE_DESC(fully_connected)

CLDNN_DECLARE_PRIMITIVE_TYPE_ID(fully_connected);

#ifdef __cplusplus
}
#endif

#endif /* FULLY_CONNECTED_H */

