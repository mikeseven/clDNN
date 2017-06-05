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
#ifndef SIMPLER_NMS_H
#define SIMPLER_NMS_H

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

#define CLDNN_ROI_VECTOR_SIZE 5

CLDNN_BEGIN_PRIMITIVE_DESC(simpler_nms)
    int max_proposals;
    float iou_threshold;
    int min_bbox_size;
    int feature_stride;
    int pre_nms_topn;
    int post_nms_topn;
    cldnn_float_arr scales;
CLDNN_END_PRIMITIVE_DESC(simpler_nms)

CLDNN_DECLARE_PRIMITIVE_TYPE_ID(simpler_nms);

#ifdef __cplusplus
}
#endif

/// @}
/// @}
/// @}
#endif /* SIMPLER_NMS_H */

