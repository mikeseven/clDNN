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
#include "primitive_arg.h"
#include <memory>
#include "topology_impl.h"


namespace cldnn
{
    class roi_pooling_arg : public primitive_arg_base<roi_pooling>
    {
    public:
        enum input_index
        {
            data_index,
            rois_index
        };
    
        roi_pooling_arg(network_impl& network, std::shared_ptr<const roi_pooling> desc);

        static layout calc_output_layout(const topology_map& topology_map, std::shared_ptr<const roi_pooling> desc);
    };
}
