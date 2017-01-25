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
#include "event_impl.h"
#include <memory>

namespace neural { namespace gpu { class gpu_toolkit; } }
namespace cldnn
{
struct network_impl;
struct primitive_impl {
    virtual refcounted_obj_ptr<event_impl> execute(const std::vector<refcounted_obj_ptr<event_impl>>& events) = 0;
    virtual ~primitive_impl() = default;
};

class primitive_arg
{
public:
    virtual ~primitive_arg() = default;
    const std::vector<std::shared_ptr<const primitive_arg>>& input() const { return _inputs; }
    const memory& input_memory(size_t index) const { return input().at(index)->output_memory(); }
    const memory& output_memory() const { return _output; }
    layout non_padded_output_layout() const
    {
        layout tmp = _output.get_layout();
        tmp.size = tmp.size.sub(_desc->output_padding().size().mul(2));
        return tmp;
    }
    primitive_type_id type() const { return _desc->type(); }
    primitive_id id() const { return _desc->id(); }
    const std::shared_ptr<const primitive>& desc() const { return _desc; }
    network_impl& get_network() const { return _network; }
    std::unique_ptr<primitive_impl> _impl;

    refcounted_obj_ptr<event_impl> execute(const std::vector<refcounted_obj_ptr<event_impl>>& events) const;

protected:
    primitive_arg(network_impl& network, std::shared_ptr<const primitive> desc, const memory& output_memory);
    primitive_arg(network_impl& network, std::shared_ptr<const primitive> desc, const layout& output_layout);
    static memory allocate_output(network_impl& network, std::shared_ptr<const primitive> desc, const layout& output_layout);
    network_impl& _network;
    std::shared_ptr<const primitive> _desc;
    std::vector<std::shared_ptr<const primitive_arg>> _inputs;
    memory _output;
};

template<class PType>
class primitive_arg_base : public primitive_arg
{
public:
    const PType& argument;
protected:
    primitive_arg_base(network_impl& network, std::shared_ptr<const PType> desc, const memory& output_memory)
        : primitive_arg(network, desc, output_memory)
        , argument(*std::static_pointer_cast<const PType>(_desc))
    {}

    primitive_arg_base(network_impl& network, std::shared_ptr<const PType> desc, const layout& output_layout)
        : primitive_arg(network, desc, output_layout)
        , argument(*std::static_pointer_cast<const PType>(_desc))
    {}
};
}
