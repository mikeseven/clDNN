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

#include "neural_impl.h"
//#include "engine_impl.h"
#include "network_impl.h"
//#include "implementation_map.h"
#include "primitive_type_base.h"

//#include <cmath>

namespace cldnn
{


primitive_type_id roi_pooling_type_id()
{
    static primitive_type_base<roi_pooling, roi_pooling_arg> instance;
    return &instance;
}


layout roi_pooling_arg::calc_output_layout(const topology_map& topology_map, std::shared_ptr<const roi_pooling> desc)
{
    auto input_desc = topology_map.at(desc->input()[data_index])->primitive_desc;
    layout input_layout = input_desc->type()->calc_output_layout(topology_map, input_desc);    
    int fm = input_layout.size.feature[0];

    input_desc = topology_map.at(desc->input()[rois_index])->primitive_desc;
    input_layout = input_desc->type()->calc_output_layout(topology_map, input_desc);    
    int num_rois = input_layout.size.batch[0];

    return layout( input_layout.data_type, { format::bfyx, { num_rois, fm, desc->pooled_height, desc->pooled_width }});
}


roi_pooling_arg::roi_pooling_arg(network_impl& network, std::shared_ptr<const roi_pooling> desc)
    :primitive_arg_base(network, desc, calc_output_layout(network.get_topology()->get_primitives(), desc))
{}

}
