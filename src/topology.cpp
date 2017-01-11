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
#include "topology_impl.h"
#include "primitive_type.h"

namespace cldnn
{
topology_impl* topology::create_topology_impl(status_t* status)
{
    try
    {
        if (status)
            *status = CLDNN_SUCCESS;
        return new topology_impl();
    }
    catch (...)
    {
        if (status)
            *status = CLDNN_ERROR;
        return nullptr;
    }
}

status_t topology::add_primitive_dto(cldnn_topology_t topology, const primitive_dto* dto)
{
    try
    {
        topology->add(dto->type->from_dto(dto));
        return CLDNN_SUCCESS;
    }
    catch(...)
    {
        return CLDNN_ERROR;
    }
}

void topology::retain_topology(cldnn_topology_t topology)
{
    topology->add_ref();
}

void topology::release_topology(cldnn_topology_t topology)
{
    topology->release();
}
}
