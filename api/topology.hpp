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
#include <cstdint>
#include "cldnn_defs.h"
#include "compounds.h"
#include "primitive.hpp"
#include "memory.hpp"

namespace cldnn {

BEGIN_DTO(data)
    memory mem;
END_DTO(data)

    class data : public primitive_base<data, DTO(data)>
{
public:
    typedef DTO(data) dto;
    DLL_SYM static primitive_type_id type_id();

    data(const primitive_id& id, const memory& mem)
        :primitive_base(id, {}, { format::x, 0,{ 0 } }, { format::x, 0,{ 0 } }, padding_types::zero, mem)
    {}

    explicit data(const dto* dto)
        :primitive_base(dto)
    {}
    const memory& mem() const { return _dto.mem; }
};

BEGIN_DTO(input_layout)
    layout layout;
END_DTO(input_layout)

    class input_layout : public primitive_base<input_layout, DTO(input_layout)>
{
public:
    typedef DTO(input_layout) dto;
    DLL_SYM static primitive_type_id type_id();

    input_layout(const primitive_id& id, const layout& layout)
        :primitive_base(id, {}, { format::x, 0,{ 0 } }, { format::x, 0,{ 0 } }, padding_types::zero, layout)
    {}

    explicit input_layout(const dto* dto)
        :primitive_base(dto)
    {}

    layout layout() const { return _dto.layout; }
};

class topology_impl;
struct topology
{
    static topology create()
    {
        return create_obj<topology, topology_impl>("failed to create topology", create_topology_impl);
    }

    DLL_SYM topology(const topology& other);
    DLL_SYM topology& operator=(const topology& other);
    DLL_SYM ~topology();
    friend bool operator==(const topology& lhs, const topology& rhs) { return lhs._impl == rhs._impl; }
    friend bool operator!=(const topology& lhs, const topology& rhs) { return !(lhs == rhs); }

    void add_data(primitive_id id, const memory& mem)
    {
        add_primitive(data(id, mem));
    }

    void add_input(primitive_id id, const layout& layout)
    {
        add_primitive(input_layout(id, layout));
    }

    template<class PType>
    void add_primitive(PType&& desc)
    {
        status_t status = add_primitive_dto(desc.get_dto());
        if (status != CLDNN_SUCCESS)
            CLDNN_THROW("primitive add failed", status);
    }

private:
    friend struct engine;
    topology_impl* _impl;
    topology(topology_impl* impl) :_impl(impl) {}

    DLL_SYM static topology_impl* create_topology_impl(status_t* status) noexcept;
    DLL_SYM status_t add_primitive_dto(const primitive_dto* dto) noexcept;
};

API_CLASS(topology)

}