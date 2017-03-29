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
#include "api/primitives/deconvolution.hpp"
#include "primitive_inst.h"
#include "topology_impl.h"

namespace cldnn
{

template <>
class typed_primitive_inst<deconvolution> : public typed_primitive_inst_base<deconvolution>
{
    using parent = typed_primitive_inst_base<deconvolution>;

public:
    static layout calc_output_layout(const topology_map& topology_map, std::shared_ptr<const deconvolution> desc);

public:
    typed_primitive_inst(network_impl& network, std::shared_ptr<const deconvolution> desc);

    const memory& input_memory() const { return dep_memory(0); }

    const memory& weights_memory(size_t index) const
    {
        if (index >= argument.weights.size())
            throw std::range_error("weights offset too big");

        return dep_memory(1 + index);
    }

    const memory& bias_memory(size_t index) const
    {
        if (index >= argument.bias.size())
            throw std::range_error("bias offset too big");

        return dep_memory(1 + argument.weights.size() + index);
    }
};

using deconvolution_inst = typed_primitive_inst<deconvolution>;

}
