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
#include <fstream>

#include <gtest/gtest.h>
#include <api/memory.hpp>
#include <api/primitives/input_layout.hpp>
#include <api/primitives/simpler_nms.hpp>
#include <include/simpler_nms_arg.h>
#include <api/topology.hpp>
#include <api/network.hpp>
#include <api/engine.hpp>
#include "test_utils/test_utils.h"

using namespace cldnn;
using namespace tests;
using namespace std;

extern float cls_scores_data[];
extern size_t cls_scores_data_size;
extern float bbox_pred_data[];
extern size_t bbox_pred_data_size;
extern float simpler_nms_ref[];
extern size_t simpler_nms_ref_size;

const float epsilon = 0.00025f;

// !!!!!!!!
// The data for this test (input and ref) was generated in clCaffe using the zf truncated prototxt with the following modifications:
// input height: 420 -> 210
// input width: 700 -> 350
// max proposals: 300 -> 50
// post nms topn: 150 -> 25
// !!!!!!!!

TEST(simpler_nms, basic) {

    primitive_id layer_name = "simpler_nms";

    int max_proposals = 50;
    float iou_threshold = 0.7f;
    int min_bbox_size = 16;
    int feature_stride = 16;
    int pre_nms_topn = 6000;
    int post_nms_topn = 25;
    int image_w = 350;
    int image_h = 210;
    int image_z = 1;
    std::vector<float> scales = { 8.0f, 16.0f, 32.0f };

    //  Brief test description.
    //    
    engine engine;

    // copy input into cldnn objects
    memory cls_scores = memory::allocate(engine, { data_types::f32, { format::bfyx, { 1, 18, 14, 23 } } });
    char* p = cls_scores.pointer<char>().data();
    memcpy(p, cls_scores_data, cls_scores_data_size * sizeof(float));

    memory bbox_pred  = memory::allocate(engine, { data_types::f32, { format::bfyx, { 1, 36, 14, 23 } } });
    p = bbox_pred.pointer<char>().data();
    memcpy(p, &bbox_pred_data[0], bbox_pred_data_size * sizeof(float));

	memory image_info = memory::allocate(engine, { data_types::i32,{ format::x,{ 3 } } });
	float* image_info_mem = image_info.pointer<float>().data();	
	image_info_mem[cldnn::simpler_nms_arg::image_info_width_index]  = (float)image_w - 0.0000001f;  // check handling of fp robustness
	image_info_mem[cldnn::simpler_nms_arg::image_info_height_index] = (float)image_h;
	image_info_mem[cldnn::simpler_nms_arg::image_info_depth_index]  = (float)image_z;

    // prepare the network
    topology topology;
    topology.add(input_layout("cls_scores", cls_scores.get_layout()));
    topology.add(input_layout("bbox_pred", bbox_pred.get_layout()));
	topology.add(input_layout("image_info", image_info.get_layout()));

    simpler_nms test_layer( layer_name, 
                            "cls_scores", 
                            "bbox_pred",
							"image_info",
                            max_proposals,
                            iou_threshold,
                            min_bbox_size,
                            feature_stride,
                            pre_nms_topn,
                            post_nms_topn,
                            scales,
                            padding(),
                            padding());

    topology.add(test_layer);

    network network(engine, topology);

    network.set_input_data("cls_scores", cls_scores);
    network.set_input_data("bbox_pred", bbox_pred);
	network.set_input_data("image_info", image_info);

    std::map<primitive_id, network_output> network_output = network.execute();
    EXPECT_EQ(network_output.begin()->first, layer_name);
    const memory& output_mem = network_output.at(layer_name).get_memory();
    EXPECT_EQ((unsigned int)output_mem.get_layout().count(), simpler_nms_ref_size);

    float* f = output_mem.pointer<float>().data();

    for (unsigned int i = 0 ; i < simpler_nms_ref_size ; i++) {
        EXPECT_NEAR(f[i], simpler_nms_ref[i], epsilon);
    }
}

