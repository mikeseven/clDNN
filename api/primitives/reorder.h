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
#ifndef REORDER_H
#define REORDER_H

#include "api/cldnn.h"

#ifdef __cplusplus
extern "C" {
#endif

CLDNN_BEGIN_PRIMITIVE_DESC(reorder)
cldnn_layout output_layout;
cldnn_primitive_id mean_substract;
cldnn_float_arr substract_per_feature;
CLDNN_END_PRIMITIVE_DESC(reorder)

CLDNN_DECLARE_PRIMITIVE_TYPE_ID(reorder);

#ifdef __cplusplus
}
#endif

#endif /* REORDER_H */

