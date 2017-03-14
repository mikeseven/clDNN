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
#ifndef ELTWISE_H
#define ELTWISE_H

#include "../cldnn.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum /*:int32_t*/
{
    cldnn_eltwise_sum,
    cldnn_eltwise_sub,
    cldnn_eltwise_max,
    cldnn_eltwise_prod
} cldnn_eltwise_mode;

CLDNN_BEGIN_PRIMITIVE_DESC(eltwise)
cldnn_primitive_id input2;
int32_t mode; /*cldnn_eltwise_mode*/
uint32_t with_activation;
float activation_negative_slope;
CLDNN_END_PRIMITIVE_DESC(eltwise)

CLDNN_DECLARE_PRIMITIVE_TYPE_ID(eltwise);

#ifdef __cplusplus
}
#endif

#endif /* ELTWISE_H */

