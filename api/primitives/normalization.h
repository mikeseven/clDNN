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
#ifndef NORMALIZATION_H
#define NORMALIZATION_H

#include "../cldnn.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum /*:int32_t*/
{
    cldnn_lrn_norm_region_across_channel,
    cldnn_lrn_norm_region_within_channel
} cldnn_lrn_norm_region;


CLDNN_BEGIN_PRIMITIVE_DESC(normalization)
uint32_t size;
float k;
float alpha;
float beta;
cldnn_lrn_norm_region norm_region;
CLDNN_END_PRIMITIVE_DESC(normalization)

CLDNN_DECLARE_PRIMITIVE_TYPE_ID(normalization);

#ifdef __cplusplus
}
#endif

#endif /* NORMALIZATION_H */

