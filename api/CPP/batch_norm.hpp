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
#include "../C/batch_norm.h"
#include "primitive.hpp"

namespace cldnn
{
/// @addtogroup cpp_api C++ API
/// @{
/// @addtogroup cpp_topology Network Topology
/// @{
/// @addtogroup cpp_primitives Primitives
/// @{

/// @brief Batch normalization primitive.
/// @details Performs batch normalization as discribed in
/// "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift" by Ioffe, Szegedy
/// @n See: http://arxiv.org/abs/1502.03167
/// 
/// <b>Algorithm:</b>
/// @n global stats can be computed as:
/// @n mean = &Sigma;(whole_feature_map_accros_batch)/(batch_size*in_x*in_y)
/// @n variance = &Sigma;((x[i] - mean)^2)/(batch_size*in_x*in_y) 
/// @n out[i] = in[i] - mean[b] / sqrt(variance[b] + epsilon)
/// @n when global_stats argument is set, mean and variance computation is skipped, values are provided from API.
/// Otherwise mean and variance primitives are ignored

struct batch_norm : public primitive_base<batch_norm, CLDNN_PRIMITIVE_DESC(batch_norm)>
{
    CLDNN_DECLATE_PRIMITIVE(batch_norm)

    /// @brief Constructs batch normalization primitive.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param mean Primitive id containing mean data.
    /// @param variance Primitive id containing variance.
    /// @param use_global_stats Use global statistics.
    /// @param epsilon Epsilon.
    batch_norm(
        const primitive_id& id,
        const primitive_id& input,
        const primitive_id& mean,
        const primitive_id& variance,
        bool use_global_stats,
        float epsilon,
        const padding& output_padding = padding()
    )
        :primitive_base(id, {input}, output_padding)
        , mean(mean)
        , variance(variance)
        , use_global_stats(use_global_stats)
        , epsilon(epsilon)
    {
    }

    /// @brief Constructs a copy from C API @CLDNN_PRIMITIVE_DESC{batch_norm}
    batch_norm(const dto* dto)
        :primitive_base(dto)
        , mean(dto->mean)
        , variance(dto->variance)
        , use_global_stats(dto->use_global_stats != 0)
        , epsilon(dto->epsilon)
    {
    }

    /// @brief Primitive id containing mean data.
    primitive_id mean;
    /// @brief Primitive id containing variance.
    primitive_id variance;
    /// @brief Use global statistics.
    bool use_global_stats;
    /// @brief Epsilon.
    float epsilon;

protected:
    std::vector<std::reference_wrapper<const primitive_id>> get_dependencies() const override { return{ mean, variance }; }

    void update_dto(dto& dto) const override
    {
        dto.mean = mean.c_str();
        dto.variance = variance.c_str();
        dto.use_global_stats = use_global_stats;
        dto.epsilon = epsilon;
    }
};
/// @}
/// @}
/// @}
}
