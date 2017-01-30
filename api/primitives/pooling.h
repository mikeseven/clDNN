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
#ifndef POOLING_H
#define POOLING_H

#include "api/cldnn.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum /*:int32_t*/
{
    cldnn_pooling_max,
    cldnn_pooling_average
} cldnn_pooling_mode;

CLDNN_BEGIN_PRIMITIVE_DESC(pooling)
int32_t mode; /*cldnn_pooling_mode*/
cldnn_tensor stride;
cldnn_tensor size;
CLDNN_END_PRIMITIVE_DESC(pooling)

CLDNN_DECLARE_PRIMITIVE_TYPE_ID(pooling);

#ifdef __cplusplus
}
#endif

#endif /* POOLING_H */

