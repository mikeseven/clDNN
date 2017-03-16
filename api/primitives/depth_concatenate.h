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
#ifndef DEPTH_CONCATENATE_H
#define DEPTH_CONCATENATE_H

#include "../cldnn.h"

#ifdef __cplusplus
extern "C" {
#endif

/// @details Depth concatenation is used to concatenate features from multiple sources into one destination.
/// Note that all sources must have the same spatial and batch sizes and also the same format as output.
/// @par Alogrithm:
/// \code
///     int outputIdx = 0
///     for(i : input)
///     {
///         for(f : i.features)
///         {
///             output[outputIdx] = f
///             outputIdx += 1
///         }
///     }
/// \endcode
/// @par Where: 
///   @li input : data structure holding all source inputs for this primitive
///   @li output : data structure holding output data for this primitive
///   @li i.features : number of features in currently processed input
///   @li outputIdx : index of destination feature 
CLDNN_BEGIN_PRIMITIVE_DESC(depth_concatenate)
CLDNN_END_PRIMITIVE_DESC(depth_concatenate)

CLDNN_DECLARE_PRIMITIVE_TYPE_ID(depth_concatenate);

#ifdef __cplusplus
}
#endif

#endif /* DEPTH_CONCATENATE_H */

