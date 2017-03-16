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
#ifndef PRIOR_BOX_H
#define PRIOR_BOX_H

#include "../cldnn.h"

#ifdef __cplusplus
extern "C" {
#endif

CLDNN_BEGIN_PRIMITIVE_DESC(prior_box) 
cldnn_tensor img_size; // Image width + height
cldnn_float_arr min_sizes; // Minimum box size
cldnn_float_arr max_sizes; // Maximum box size
cldnn_float_arr aspect_ratios; // Various of aspect ratios. Duplicate ratios will be ignored.
uint32_t flip; // If not 0, will flip each aspect ratio. For example, if there is aspect ratio "r", aspect ratio "1.0/r" we will generated as well.
uint32_t clip; // If not 0, will clip the prior so that it is within [0, 1].
cldnn_float_arr variance; // Variance for adjusting the prior boxes.
float step_width; // Step width
float step_height; // Step height
float offset; // Offset to the top left corner of each cell.
CLDNN_END_PRIMITIVE_DESC(prior_box)

CLDNN_DECLARE_PRIMITIVE_TYPE_ID(prior_box);

#ifdef __cplusplus
}
#endif

#endif /* PRIOR_BOX_H */

