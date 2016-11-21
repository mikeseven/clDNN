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

namespace cldnn
{
class topology_impl : public refcounted_obj<topology_impl>
{
public:
    typedef std::map<primitive_id, std::shared_ptr<primitive_desc>> primitives_map;
    topology_impl(const context& ctx)
        : _context(ctx)
    {}

    void add(primitive_id id, std::shared_ptr<primitive_desc> desc)
    {
        if (_primitives.count(id) != 0)
            throw std::runtime_error("primitive '" + id + "' exists already");
        _primitives.insert({ id, desc });
    }

    context get_context() const { return _context; }

    const primitives_map& get_primitives() const { return _primitives; }

private:
    primitives_map _primitives;
    context _context;
};
}
