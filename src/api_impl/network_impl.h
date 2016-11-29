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
#include "refcounted_obj.h"
#include "primitive_arg.h"
#include <map>

namespace cldnn
{
class network_impl : public refcounted_obj<network_impl>
{
public:
    network_impl(const engine& engine, const std::map<primitive_id, std::shared_ptr<const primitive_arg>>& primitives)
        : _completed(false)
        , _engine(engine)
        , _primitives(primitives)
        , _primitive_names(get_primitive_names())
    {
        for(auto& p : _primitives)
        {
            if(p.second->type() == input_layout::type_id())
            {
                _inputs.insert({ p.second->id(), false });
            }
        }
    }

    engine get_engine() const { return _engine; }

    const memory& get_output_of(const primitive_id& id) const;
    array_ref<primitive_id_ref> get_primitive_keys() const { return _primitive_names; }
    void set_input_data(const primitive_id& id, const memory& data);

private:
    bool _completed;
    const engine _engine;
    const std::map<primitive_id, std::shared_ptr<const primitive_arg>> _primitives;
    const std::vector<primitive_id_ref> _primitive_names;
    std::map<primitive_id, bool> _inputs;

    std::vector<primitive_id_ref> get_primitive_names() const
    {
        std::vector<primitive_id_ref> result;
        for(auto& pair:_primitives)
        {
            // it should be reference to the constant primitive_id store.
            result.push_back(pair.second->id());
        }
        return result;
    }
};
}
