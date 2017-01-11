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
#include "api/memory.hpp"
#include "api/primitive.hpp"
#include "api/topology.hpp"
#include "topology_impl.h"

namespace cldnn
{
struct network_impl;
class primitive_arg;
struct primitive_type
{
    virtual std::shared_ptr<const primitive> from_dto(const primitive_dto* dto) const = 0;
    virtual std::shared_ptr<const primitive_arg> create_arg(network_impl& network, std::shared_ptr<const primitive> desc) const = 0;
    virtual ~primitive_type() = default;
    virtual layout calc_output_layout(const topology_map& topology_map, std::shared_ptr<const primitive> desc) const = 0;
};
}
