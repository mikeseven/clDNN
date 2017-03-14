/*
// Copyright (c) 2017 Intel Corporation
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
#ifndef ROI_POOLING_H
#define ROI_POOLING_H

#include "../cldnn.h"

#ifdef __cplusplus
extern "C" {
#endif


CLDNN_BEGIN_PRIMITIVE_DESC(roi_pooling)
int pooled_height;
int pooled_width;
float spatial_scale;
CLDNN_END_PRIMITIVE_DESC(roi_pooling)

CLDNN_DECLARE_PRIMITIVE_TYPE_ID(roi_pooling);

#ifdef __cplusplus
}
#endif

#endif /* ROI_POOLING_H */

