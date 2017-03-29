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
#include "api/primitives/simpler_nms.hpp"
#include "primitive_inst.h"
#include "topology_impl.h"

namespace cldnn
{

template <>
class typed_primitive_inst<simpler_nms> : public typed_primitive_inst_base<simpler_nms>
{
    using parent = typed_primitive_inst_base<simpler_nms>;

public:
    struct anchor
    {
        float start_x;
        float start_y;
        float end_x;
        float end_y;

        anchor()
        {
            start_x = start_y = end_x = end_y = 0.0f;
        }

        anchor(float s_x, float s_y, float e_x, float e_y)
        {
            start_x = s_x;
            start_y = s_y;
            end_x = e_x;
            end_y = e_y;
        }
    };

    // indices of the memory objects used by the layer
    enum input_index {
        cls_scores_index,
        bbox_pred_index,
		image_info_index
    };

    // indices of the image info parameters inside the image_info memory object (the object
    // is an integer array of these parameters)
	enum image_info_size_index {
		image_info_width_index = 0,
		image_info_height_index = 1,
		image_info_depth_index = 2
	};

    static layout calc_output_layout(const topology_map& topology_map, std::shared_ptr<const simpler_nms> desc);

public:    
    typed_primitive_inst(network_impl& network, std::shared_ptr<const simpler_nms> desc);

    const std::vector<anchor>& get_anchors() const { return _anchors; }

private:
    std::vector<anchor> _anchors;
};

using simpler_nms_inst = typed_primitive_inst<simpler_nms>;

}
