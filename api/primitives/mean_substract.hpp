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
#include "../primitive.hpp"

namespace cldnn
{
BEGIN_DTO(mean_substract)
        primitive_id_ref mean;
END_DTO(mean_substract)

struct mean_substract : public primitive_base<mean_substract, DTO(mean_substract)>
{
    DLL_SYM static primitive_type_id type_id();
    typedef DTO(mean_substract) dto;

    mean_substract(
        const primitive_id& id,
        const primitive_id& input,
        const primitive_id& mean,
        const padding& input_padding = padding(),
        const padding& output_padding = padding()
    )
        :primitive_base(id, {input}, input_padding, output_padding)
        , mean(mean)
    {
        init_dto();
    }

    mean_substract(const dto* dto)
        :primitive_base(dto)
        , mean(dto->mean)
    {
        init_dto();
    }

    const primitive_id mean;

protected:
    std::vector<primitive_id> get_dependencies() const override { return{ mean }; }

    void init_dto()
    {
        _dto.mean = mean;
    }
};
}
