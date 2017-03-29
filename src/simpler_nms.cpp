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

#include "simpler_nms_inst.h"
#include "primitive_type_base.h"
#include "network_impl.h"

#include <cmath>

namespace cldnn
{

static void generate_anchors(unsigned int base_size, const std::vector<float>& ratios, const std::vector<float>& scales,   // input
                             std::vector<simpler_nms_inst::anchor>& anchors);                                                                // output


primitive_type_id simpler_nms_type_id()
{
    static primitive_type_base<simpler_nms, simpler_nms_inst> instance;
    return &instance;
}


layout simpler_nms_inst::calc_output_layout(const topology_map& topology_map, std::shared_ptr<const simpler_nms> desc)
{
	auto input_desc = topology_map.at(desc->input[simpler_nms_inst::cls_scores_index])->primitive_desc;
	layout input_layout = input_desc->type->calc_output_layout(topology_map, input_desc);

	return layout(input_layout.data_type, { format::bx, { desc->post_nms_topn, CLDNN_ROI_VECTOR_SIZE}});
}


simpler_nms_inst::typed_primitive_inst(network_impl& network, std::shared_ptr<const simpler_nms> desc)
    :parent(network, desc, calc_output_layout(network.get_topology()->get_primitives(), desc))
{
	std::vector<float> default_ratios = { 0.5f, 1.0f, 2.0f };

	int default_size = 16;

	generate_anchors(default_size, default_ratios, desc->scales, _anchors);             	
}

static void calc_basic_params(const simpler_nms_inst::anchor& base_anchor,                                       // input
                            float& width, float& height, float& x_center, float& y_center)   // output
{
    width  = base_anchor.end_x - base_anchor.start_x + 1.0f;
    height = base_anchor.end_y - base_anchor.start_y + 1.0f;

    x_center = base_anchor.start_x + 0.5f * (width - 1.0f);
    y_center = base_anchor.start_y + 0.5f * (height - 1.0f);
}


static void make_anchors(const std::vector<float>& ws, const std::vector<float>& hs, float x_center, float y_center,   // input
                        std::vector<simpler_nms_inst::anchor>& anchors)                                                            // output
{
    size_t len = ws.size();
    anchors.clear();
    anchors.resize(len);

    for (unsigned int i = 0 ; i < len ; i++) {
        // transpose to create the anchor
        anchors[i].start_x = x_center - 0.5f * (ws[i] - 1.0f);
        anchors[i].start_y = y_center - 0.5f * (hs[i] - 1.0f);
        anchors[i].end_x   = x_center + 0.5f * (ws[i] - 1.0f);
        anchors[i].end_y   = y_center + 0.5f * (hs[i] - 1.0f);
    }
}


static void calc_anchors(const simpler_nms_inst::anchor& base_anchor, const std::vector<float>& scales,        // input
                        std::vector<simpler_nms_inst::anchor>& anchors)                                       // output
{
    float width = 0.0f, height = 0.0f, x_center = 0.0f, y_center = 0.0f;

    calc_basic_params(base_anchor, width, height, x_center, y_center);

    size_t num_scales = scales.size();
    std::vector<float> ws(num_scales), hs(num_scales);

    for (unsigned int i = 0 ; i < num_scales ; i++) {
        ws[i] = width * scales[i];
        hs[i] = height * scales[i];
    }

    make_anchors(ws, hs, x_center, y_center, anchors);
}


static void calc_ratio_anchors(const simpler_nms_inst::anchor& base_anchor, const std::vector<float>& ratios,        // input
                             std::vector<simpler_nms_inst::anchor>& ratio_anchors)                                 // output
{
    float width = 0.0f, height = 0.0f, x_center = 0.0f, y_center = 0.0f;

    calc_basic_params(base_anchor, width, height, x_center, y_center);

    float size = width * height;

    size_t num_ratios = ratios.size();

    std::vector<float> ws(num_ratios), hs(num_ratios);

    for (unsigned int i = 0 ; i < num_ratios ; i++) {
        float new_size = size / ratios[i];
        ws[i] = round(sqrt(new_size));
        hs[i] = round(ws[i] * ratios[i]);
    }

    make_anchors(ws, hs, x_center, y_center, ratio_anchors);
}

static void generate_anchors(unsigned int base_size, const std::vector<float>& ratios, const std::vector<float>& scales,   // input
                     std::vector<simpler_nms_inst::anchor>& anchors)                                                           // output
{
    float end = (float)(base_size - 1);        // because we start at zero

    simpler_nms_inst::anchor base_anchor(0.0f, 0.0f, end, end);

    std::vector<simpler_nms_inst::anchor> ratio_anchors;
    calc_ratio_anchors(base_anchor, ratios, ratio_anchors);

    size_t num_ratio_anchors = ratio_anchors.size();

    for (unsigned int i = 0 ; i < num_ratio_anchors ; i++) {
        std::vector<simpler_nms_inst::anchor> temp_anchors;
        calc_anchors(ratio_anchors[i], scales, temp_anchors);

        size_t num_temp_anchors = temp_anchors.size();

        for (unsigned int j = 0 ; j < num_temp_anchors ; j++) {
            anchors.push_back(temp_anchors[j]);
        }
    }
}


}
