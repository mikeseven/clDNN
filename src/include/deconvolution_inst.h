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
#include "api/CPP/deconvolution.hpp"
#include "primitive_inst.h"

namespace cldnn
{

template <>
struct typed_program_node<deconvolution> : public typed_program_node_base<deconvolution>
{
    using parent = typed_program_node_base<deconvolution>;

    typed_program_node(std::shared_ptr<primitive> prim, program_impl& prog)
        : parent(prim, prog)
        , _split(this->get_primitive()->split())
    {
    }

private:
    int32_t _split;

public:
    using parent::parent;

    void set_split(int32_t node_split) { _split = node_split; }
    int32_t get_split() const { return _split; }

    auto& input() const { return get_dependency(0); }

    auto& weights(size_t idx = 0) const
    {
        if (static_cast<int32_t>(idx) >= this->get_split())
            throw std::range_error("weights offset too big");

        return get_dependency(1 + idx);
    }

    auto& bias(size_t idx = 0) const 
    { 
        if (static_cast<int32_t>(idx) >= this->get_split())
            throw std::range_error("bias offset too big");

        return get_dependency(1 + this->get_split() + idx);
    }

    bool bias_term() const
    {
        if (get_primitive()->bias.size() != 0)
            return true;
        else
            return false;
    }
};

using deconvolution_node = typed_program_node<deconvolution>;

template <>
class typed_primitive_inst<deconvolution> : public typed_primitive_inst_base<deconvolution>
{
    using parent = typed_primitive_inst_base<deconvolution>;

public:
    static layout calc_output_layout(deconvolution_node const& node);
    static std::string to_string(deconvolution_node const& node);

public:
    typed_primitive_inst(network_impl& network, deconvolution_node const& node);

    const memory& input_memory() const { return dep_memory(0); }

    const memory& weights_memory(size_t index) const
    {
        if (static_cast<int32_t>(index) >= node.get_split())
            throw std::range_error("weights offset too big");

        return dep_memory(1 + index);
    }

    const memory& bias_memory(size_t index) const
    {
        if (argument.bias.size() == 0 && static_cast<int32_t>(index) >= node.get_split())
            throw std::range_error("no bias data");

        if (static_cast<int32_t>(index) > node.get_split())
            throw std::range_error("bias offset too big");

        return dep_memory(1 + node.get_split() + index);
    }

    bool bias_term() const
    {
        if (argument.bias.size() != 0)
            return true;
        else
            return false;
    }
};

using deconvolution_inst = typed_primitive_inst<deconvolution>;

}
