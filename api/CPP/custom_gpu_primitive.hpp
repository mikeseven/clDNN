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
#include "../C/custom_gpu_primitive.h"
#include "primitive.hpp"
#include "memory.hpp"

namespace cldnn
{
/// @addtogroup cpp_api C++ API
/// @{
/// @addtogroup cpp_topology Network Topology
/// @{
/// @addtogroup cpp_primitives Primitives
/// @{

/// @brief Changes how data is ordered in memory. Value type is not changed & all information is preserved.
/// @details Corresponding values are bitwise equal before/after custom_gpu_primitive.
/// Also merged with subtraction layer, which can subtract values while doing reordering.
/// NOTE THAT THIS WILL SUBTRACT THE SAME VALUES FROM EACH BATCH.
    struct custom_gpu_primitive : public primitive_base<custom_gpu_primitive, CLDNN_PRIMITIVE_DESC(custom_gpu_primitive)>
    {
        CLDNN_DECLATE_PRIMITIVE(custom_gpu_primitive)

            /// @brief Constructs custom_gpu_primitive primitive with directly provided mean subtract values.
            /// @param id This primitive id.
            /// @param input Input primitive id.
            /// @param output_layout Requested memory layout.
            /// @param values_to_subtract Array of mean subtract values.
            custom_gpu_primitive(
                const primitive_id& id,
                const std::vector<primitive_id>& inputs,
                const std::vector<std::string>& kernels_code,
                const std::string& kernel_entry_point,
                const std::vector<cldnn_arg>& parameters,
                const std::string& build_options,
                const layout& output_layout,
                const std::vector<size_t>& gws = {},
                const std::vector<size_t>& lws = {}
            )
            : primitive_base(id, { inputs[0] }, output_layout.data_padding)
            , inputs(_inputs.cpp_ids)
            , kernels_code(_kernels_code.cpp_ids)
            , kernel_entry_point(kernel_entry_point)
            , parameters(parameters)
            , build_options(build_options)
            , output_layout(output_layout)
            , gws(gws.size() ? gws : std::vector<size_t>{ output_layout.count() })
            , lws(lws)
            , _inputs(inputs)
            , _kernels_code(kernels_code)
        {
        }

        /// @brief Constructs a copy from basic C API @CLDNN_PRIMITIVE_DESC{custom_gpu_primitive}
        custom_gpu_primitive(const dto* dto)
            : primitive_base(dto)
            , inputs(_inputs.cpp_ids)
            , kernels_code(_kernels_code.cpp_ids)
            , kernel_entry_point(dto->kernel_entry_point)
            , parameters(dto->kernel_arguments, dto->kernel_arguments + dto->kernel_arguments_num)
            , build_options(dto->build_options)
            , output_layout(dto->output_layout)
            , gws(dto->gws, dto->gws + dto->gws_num)
            , lws(dto->lws, dto->lws + dto->lws_num)
            , _inputs(dto->inputs)
            , _kernels_code(dto->kernels_code)
    {
    }

    /// @brief List of primitive ids containing weights data.
    fixed_size_vector_ref inputs;

    fixed_size_vector_ref kernels_code;
    const std::string kernel_entry_point;
    const std::vector<cldnn_arg> parameters;
    const std::string build_options;
    const layout output_layout;
    const std::vector<size_t> gws;
    const std::vector<size_t> lws;
    

protected:
    primitive_id_arr _inputs;
    primitive_id_arr _kernels_code;

    std::vector<std::reference_wrapper<const primitive_id>> get_dependencies() const override
    {
        std::vector<std::reference_wrapper<const primitive_id>> ret;
        ret.reserve(inputs.size() - 1);
        for (size_t i = 1; i < inputs.size(); i++)
        {
            ret.push_back(inputs[i]);
        }

        return ret;
    }

    void update_dto(dto& dto) const override
    {
        dto.inputs                  = _inputs.ref();
        dto.kernels_code            = _kernels_code.ref();
        dto.kernel_entry_point      = kernel_entry_point.c_str();
        dto.kernel_arguments        = parameters.data();
        dto.kernel_arguments_num    = (int)parameters.size();
        dto.build_options           = build_options.c_str();
        dto.output_layout           = (cldnn_layout)output_layout;
        dto.gws                     = gws.data();
        dto.gws_num                 = (int)gws.size();
        dto.lws                     = lws.data();
        dto.lws_num                 = (int)lws.size();
    }
};
/// @}
/// @}
/// @}
}
