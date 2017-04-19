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
#pragma once
#include "../C/depth_concatenate.h"
#include "primitive.hpp"

namespace cldnn
{
/// @addtogroup cpp_api C++ API
/// @{
/// @addtogroup cpp_topology Network Topology
/// @{
/// @addtogroup cpp_primitives Primitives
/// @{

/// @details Depth concatenation is used to concatenate features from multiple sources into one destination.
/// @notes 
/// - all sources must have the same spatial and batch sizes and also the same format as output.
/// - order of arguments in primitive creation has impact on order of feature maps in output primitive. 
/// 
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
struct depth_concatenate : public primitive_base<depth_concatenate, CLDNN_PRIMITIVE_DESC(depth_concatenate)>
{
    CLDNN_DECLATE_PRIMITIVE(depth_concatenate)

    /// @li Constructs Depth concatenate primitive.
    /// @param id This primitive id.
    /// @param input Vector of input primitives ids.
    depth_concatenate(
        const primitive_id& id,
        const std::vector<primitive_id>& input,
        const padding& output_padding = { format::bfyx,{ 0,0,0,0 } }
    )
        // We're not using input padding but we must provide it, so it will always be 0,0
        :primitive_base(id, { input }, output_padding)
    {}

    /// @brief Constructs a copy from C API @CLDNN_PRIMITIVE_DESC(depth_concatenate)
    depth_concatenate(const dto* dto)
        :primitive_base(dto)
    {}

private:
    void update_dto(dto&) const override {}
};
/// @}
/// @}
/// @}
}
