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
#ifndef CUSTOM_GPU_PRIMITIVE_H
#define CUSTOM_GPU_PRIMITIVE_H

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

/// @brief Changes how data is ordered in memory. Value type is not changed & all information is preserved.
/// @details Corresponding values are bitwise equal before/after custom_gpu_primitive.
/// Also merged with subtraction layer, which can subtract values while doing reordering.
/// NOTE THAT THIS WILL SUBTRACT THE SAME VALUES FROM EACH BATCH.
CLDNN_BEGIN_PRIMITIVE_DESC(custom_gpu_primitive)
/// @brief Second input primitive id with values needed for eltwise computation.
cldnn_primitive_id_arr inputs;

cldnn_primitive_id_arr kernels_code;

cldnn_kernel_entry_point kernel_entry_point;

cldnn_kernel_arguments kernel_arguments;

int kernel_arguments_num;

cldnn_kernel_build_options build_options;

cldnn_layout output_layout;

cldnn_work_group_sizes gws;

int gws_num;

cldnn_work_group_sizes lws;

int lws_num;

CLDNN_END_PRIMITIVE_DESC(custom_gpu_primitive)

CLDNN_DECLARE_PRIMITIVE_TYPE_ID(custom_gpu_primitive);

#ifdef __cplusplus
}
#endif

/// @}
/// @}
/// @}
#endif /* CUSTOM_GPU_PRIMITIVE_H */

