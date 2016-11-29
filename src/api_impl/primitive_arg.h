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

#include "api/cldnn.hpp"
#include "topology_impl.h"

namespace cldnn
{
class network_builder;
class primitive_arg
{
public:
    virtual ~primitive_arg() = default;
    const memory& input_memory(size_t index) const { return _inputs.at(index)->output_memory(); }
    const memory& output_memory() const { return _output; }

    std::shared_ptr<const primitive> argument() const { return _desc; }
    primitive_type_id type() const { return _desc->type(); }
    primitive_id_ref id() const { return _desc->get_dto()->id; }

protected:
    primitive_arg(network_builder& builer, std::shared_ptr<const primitive> desc, const memory& output);

private:
    std::shared_ptr<const primitive> _desc;
    std::vector<std::shared_ptr<const primitive_arg>> _inputs;
    memory _output;
};

}