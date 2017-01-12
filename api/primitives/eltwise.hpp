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
    enum class eltwise_mode { sum, sub, max, prod };

    BEGIN_DTO(eltwise)
        primitive_id_ref input2;
        eltwise_mode mode;
    END_DTO(eltwise)

        struct eltwise : public primitive_base<eltwise, DTO(eltwise)>
    {
        DLL_SYM static primitive_type_id type_id();
        typedef DTO(eltwise) dto;

        eltwise(
            const primitive_id& id,
            const primitive_id& input,
            const primitive_id& input2,
            eltwise_mode mode,
            const padding& input_padding = padding(),
            const padding& output_padding = padding()
        )
            :primitive_base(id, { input }, input_padding, output_padding, input2, mode)
            , input2(input2)
            , mode(_dto.mode)
        {
            init_dto();
        }

        eltwise(const dto* dto)
            :primitive_base(dto)
            , input2(dto->input2)
            , mode(_dto.mode)
        {
            init_dto();
        }

        const primitive_id input2;
        const eltwise_mode& mode;

    protected:
        std::vector<primitive_id> get_dependencies() const override { return{ input2 }; }

        void init_dto()
        {
            _dto.input2 = input2;
        }
    };
}
