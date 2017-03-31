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

#include "roi_pooling_inst.h"
#include "primitive_type_base.h"

namespace cldnn
{


primitive_type_id roi_pooling_type_id()
{
    static primitive_type_base<roi_pooling, roi_pooling_inst> instance;
    return &instance;
}

layout roi_pooling_inst::calc_output_layout(roi_pooling_node const& node)
{
    auto desc = node.get_primitive();
    layout data_layout = node.input().get_output_layout();
    int fm = data_layout.size.feature[0];

    layout rois_layout = node.rois().get_output_layout();
    int num_rois = rois_layout.size.batch[0];

    return layout(rois_layout.data_type, { format::bfyx,{ num_rois, fm, desc->pooled_height, desc->pooled_width } });
}
}
