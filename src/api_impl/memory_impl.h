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
#include "primitive_arg.h"
#include "network_builder.h"

namespace cldnn
{
class data_arg : public primitive_arg
{
public:
    data_arg(network_builder& builder, std::shared_ptr<const data> desc)
        :primitive_arg(builder, desc, desc->mem())
    {}
};

class input_arg : public primitive_arg
{
public:
    input_arg(network_builder& builder, std::shared_ptr<const input_layout> desc)
        :primitive_arg(builder, desc, builder.get_engine().allocate_memory(desc->layout()))
    {}
};
}