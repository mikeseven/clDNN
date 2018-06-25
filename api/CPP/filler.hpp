/*
// Copyright (c) 2018 Intel Corporation
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
#include "../C/filler.h"
#include "primitive.hpp"
#include "memory.hpp"
#include "cldnn_defs.h"

namespace cldnn
{
/// @addtogroup cpp_api C++ API
/// @{
/// @addtogroup cpp_topology Network Topology
/// @{
/// @addtogroup cpp_primitives Primitives
/// @{

/// @brief Provides filled mutable data.
/// @details This primitive allows to pass data which can be written to during training. Data is filled with specified filler.
/// For example, weights and biases for scoring networks.
/// This primitive can be also set as other primitive's output. In this case the underlying buffer will be the same in weights_filler and preceding primitive.
struct filler : public primitive_base<filler, CLDNN_PRIMITIVE_DESC(filler)>
{
    CLDNN_DECLARE_PRIMITIVE(filler)

    /// @brief Enum type to specify function for weights filling.
    enum filler_type
    {
        xavier
    };

    /// @brief Constructs filler primitive.
    /// @param id This primitive id.
    /// @param mem @ref memory object which contains data.
    /// @note If memory is attached by memory::attach(), the attached buffer should be valid till network build.
    filler(const primitive_id& id, const memory& mem, const filler_type fill_type)
        :primitive_base(id, {}, padding())
        , fill_type(fill_type)
        , mem(mem)
    {}

    /// @brief Constructs filler primitive with inputs.
    /// @param id This primitive id.
    /// @param input Vector of input primitives ids.
    /// @param mem @ref memory object which contains data.
    /// @note If memory is attached by memory::attach(), the attached buffer should be valid till network build.
    filler(const primitive_id& id, const std::vector<primitive_id>& input, const memory& mem, const filler_type fill_type)
        :primitive_base(id, { input }, padding())
        , fill_type(fill_type)
        , mem(mem)
    {}

    /// @brief Constructs a copy from C API @CLDNN_PRIMITIVE_DESC{filler}
    explicit filler(const dto* dto)
        :primitive_base(dto)
        , mem(dto->mem)
        , fill_type(static_cast<filler_type>(dto->fill_type))
    {
        mem.retain();
    }

    /// @brief Specifies function which will be used to fill weights.
    filler_type fill_type;

    /// @brief @ref memory object which contains data.
    /// @note If memory is attached by memory::attach(), the attached buffer should be valid till network build.
    memory mem;

protected:
    void update_dto(dto& dto) const override
    {
        dto.mem = mem.get();
        dto.fill_type = static_cast<cldnn_filler_type>(fill_type);
    }
};
/// @}
/// @}
/// @}
}
