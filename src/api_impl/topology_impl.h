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
#include <map>
#include "api/cldnn.hpp"
#include "refcounted_obj.h"
#include "api/reorder.hpp"
#include "api/convolution.hpp"

namespace cldnn
{
class primitive_arg;
struct primitive_type_impl
{
    virtual std::shared_ptr<const primitive> from_dto(const primitive_dto* dto) const = 0;
    virtual std::shared_ptr<const primitive_arg> create_arg(network_impl* net, std::shared_ptr<const primitive> desc) const = 0;
    virtual ~primitive_type_impl() = default;
};

typedef std::map<primitive_id, std::shared_ptr<const primitive>> descriptions_map;

class topology_impl : public refcounted_obj<topology_impl>
{
public:
    
    topology_impl(const context& ctx)
        : _context(ctx)
    {}

    void add(std::shared_ptr<const primitive> desc)
    {
        auto id = desc->id();
        if (_primitives.count(id) != 0)
            throw std::runtime_error("primitive '" + id + "' exists already");
        _primitives.insert({ id, desc });
    }

    context get_context() const { return _context; }

    const descriptions_map& get_primitives() const { return _primitives; }

private:
    descriptions_map _primitives;
    context _context;
};
}
