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
#include "api/topology.hpp"
#include "api/reorder.hpp"
#include "api/convolution.hpp"
#include "api/cldnn.hpp"
#include <map>
#include "topology_impl.h"

namespace cldnn
{
namespace {
std::shared_ptr<primitive_desc> primitive_from_dto(const primitive_dto* dto)
{
    switch (dto->type)
    {
    case reorder:
        return std::make_shared<reorder_desc>(dto);
    //case mean_substract: break;
    case convolution:
        return std::make_shared<convolution_desc>(dto);
    //case fully_connected: break;
    //case activation: break;
    //case pooling: break;
    //case normalization: break;
    //case softmax: break;
    //case depth_concat: break;
    default:
        throw std::runtime_error("unknown primitive type");
    }
}
}

BEGIN_DTO(data)
memory data;
END_DTO(data)

BEGIN_DESC(data)
public:
    data_desc(primitive_id id, const memory& data)
        :primitive_desc_base({id})
    {
        _dto.data = data;
    }
END_DESC(data)

BEGIN_DTO(input)
layout layout;
END_DTO(input)

BEGIN_DESC(input)
public:
    input_desc(primitive_id id, const layout& layout)
        :primitive_desc_base({ id })
    {
        _dto.layout = layout;
    }
END_DESC(input)

status_t topology::add_data(primitive_id id, const memory& mem)
{
    try
    {
        _impl->add(id, std::make_shared<data_desc>(id, mem));
    }
    catch (...)
    {
        return CLDNN_ERROR;
    }
    return CLDNN_SUCCESS;
}

status_t topology::add_input(primitive_id id, const layout& layout)
{
    try
    {
        _impl->add(id, std::make_shared<input_desc>(id, layout));
    }
    catch (...)
    {
        return CLDNN_ERROR;
    }
    return CLDNN_SUCCESS;
}

status_t topology::add_primitive_dto(primitive_id_ref id, const primitive_dto* dto)
{
    try
    {
        _impl->add(id, primitive_from_dto(dto));
    }
    catch(...)
    {
        return CLDNN_ERROR;
    }
    return CLDNN_SUCCESS;
}

context topology::get_context() const
{
    return _impl->get_context();
}

topology::topology(const topology& other):_impl(other._impl)
{
    _impl->add_ref();
}

topology& topology::operator=(const topology& other)
{
    if (_impl == other._impl) return *this;
    _impl->release();
    _impl = other._impl;
    _impl->add_ref();
    return *this;
}

topology::~topology()
{
    _impl->release();
}
}
