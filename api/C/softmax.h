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
#ifndef SOFTMAX_H
#define SOFTMAX_H

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

typedef enum
{
    cldnn_softmax_normalize_bfyx = 0,
    cldnn_softmax_normalize_fyx = CLDNN_TENSOR_BATCH_DIM_MAX,
    cldnn_softmax_normalize_x = CLDNN_TENSOR_BATCH_DIM_MAX + CLDNN_TENSOR_FEATURE_DIM_MAX,
    cldnn_softmax_normalize_yx = cldnn_softmax_normalize_x + 1
} cldnn_softmax_dimension;

/// @brief Normalizes results so they sum to 1. The scope of normalization is defined by a member @p dimension.
/// @details
/// @par Algorithm:
///   b = e^a/sum(N-1; j=0; e^j)
/// @par Where:
///   @li N : number of values to normalize
///   @li b : value after normalization
///   @li a : value before normalization
CLDNN_BEGIN_PRIMITIVE_DESC(softmax)
/// @brief Defines a scope of a single softmax normalization as well as number of normalizations.
/// @details
/// Being given a 4-dimensional input, which consists of b,f,y,x dimensions, softmax normalizes n data sets of m elements each.
/// The values of n and m are calculated from b,f,y,x in such way that as a result whole input is processed. Specific behaviour is
/// determined by this parameter, as follows:
/// - when set to @p cldnn_softmax_normalize_x only x-dimension is used as a number of elements @p m and the number of data sets @p n is calculated as b*f*y (each image row is normalized independently)
/// - when set to @p cldnn_softmax_normalize_yx @m is defined as x*y and @n as b*f (each 2d image is normalized independently)
/// - when set to @P cldnn_softmax_normalize_fyx @m is defined as f*y*x and @n as b (each 3d image is normalized independently)
/// - when set to @p cldnn_softmax_normalize_bfyx the whole input is normalized as a single data set
cldnn_softmax_dimension dimension;
CLDNN_END_PRIMITIVE_DESC(softmax)

CLDNN_DECLARE_PRIMITIVE_TYPE_ID(softmax);

#ifdef __cplusplus
}
#endif

/// @}
/// @}
/// @}
#endif /* SOFTMAX_H */

