/*
// Copyright (c) 2017 Intel Corporation
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
#include "api/primitives/roi_pooling.hpp"
#include "primitive_inst.h"
#include "topology_impl.h"

namespace cldnn
{
template <>
class typed_primitive_inst<roi_pooling> : public typed_primitive_inst_base<roi_pooling>
{
    using parent = typed_primitive_inst_base<roi_pooling>;

public:
    enum input_index
    {
        data_index,
        rois_index
    };

    static layout calc_output_layout(const topology_map& topology_map, std::shared_ptr<const roi_pooling> desc);

public:    
    typed_primitive_inst(network_impl& network, std::shared_ptr<const roi_pooling> desc);

    const memory& input_memory() const { return dep_memory(0); }
    const memory& rois_memory() const { return dep_memory(1); }
};

using roi_pooling_inst = typed_primitive_inst<roi_pooling>;

}
