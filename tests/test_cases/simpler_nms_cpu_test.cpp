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
#include "test_utils/float16.h"

namespace cldnn
{
template<> struct type_to_data_type<FLOAT16> { static const data_types value = data_types::f16; };
}


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
const float epsilon_fp16 = 0.125f;

// !!!!!!!!
// The data for this test (input and ref) was generated in clCaffe using the zf truncated prototxt with the following modifications:
// input height: 420 -> 210
// input width: 700 -> 350
// max proposals: 300 -> 50
// post nms topn: 150 -> 25
// !!!!!!!!

const primitive_id cls_scores_name = "cls_scores";
const primitive_id bbox_pred_name = "bbox_pred";
const primitive_id image_info_name = "image_info";
const primitive_id layer_name = "simpler_nms";

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


template <typename Dtype>
class TestRunnerSimplerNMS 
{
    public:
        TestRunnerSimplerNMS();

        ~TestRunnerSimplerNMS();

        memory Run(std::vector<Dtype>& data, 
                   std::vector<Dtype>& rois);

    private:

        engine _engine;
        layout _cls_scores_layout;
        layout _bbox_pred_layout;
        layout _image_info_layout;
        topology _topology;
        simpler_nms _test_layer;
        network* _network = NULL;
};


template <typename Dtype>
TestRunnerSimplerNMS<Dtype>::TestRunnerSimplerNMS() :
                            _cls_scores_layout(cldnn::type_to_data_type<Dtype>::value, { format::bfyx, { 1, 18, 14, 23 } } ),
                            _bbox_pred_layout(cldnn::type_to_data_type<Dtype>::value, { format::bfyx, { 1, 36, 14, 23 } } ),
                            _image_info_layout(data_types::f32, { format::x, { 3 } } ),
                            _test_layer(layer_name, 
                                        cls_scores_name, 
                                        bbox_pred_name,
                                        image_info_name,
                                        max_proposals,
                                        iou_threshold,
                                        min_bbox_size,
                                        feature_stride,
                                        pre_nms_topn,
                                        post_nms_topn,
                                        scales,
                                        padding(),
                                        padding())
{    
    _topology.add(input_layout(cls_scores_name, _cls_scores_layout));
    _topology.add(input_layout(bbox_pred_name, _bbox_pred_layout));
	_topology.add(input_layout(image_info_name, _image_info_layout));

    _topology.add(_test_layer);

    _network = new network(_engine, _topology);
}

template <typename Dtype>
TestRunnerSimplerNMS<Dtype>::~TestRunnerSimplerNMS()
{
    delete _network;
}

template <typename Dtype>
memory TestRunnerSimplerNMS<Dtype>::Run(std::vector<Dtype>& cls_scores_vals, 
                              std::vector<Dtype>& bbox_pred_vals)
{
    memory cls_scores = memory::attach(_cls_scores_layout, cls_scores_vals.data(), cls_scores_vals.size());
    memory bbox_pred  = memory::attach(_bbox_pred_layout, bbox_pred_vals.data(), bbox_pred_vals.size());

    float image_info_vals[] = { (float)image_w - 0.0000001f, // check fp robustness of the layer
                                (float)image_h + 0.0000001f, // check fp robustness of the layer 
                                (float)image_z };
    memory image_info = memory::attach(_image_info_layout, &image_info_vals[0], 3);
   
    _network->set_input_data(cls_scores_name, cls_scores);
    _network->set_input_data(bbox_pred_name, bbox_pred);
	_network->set_input_data(image_info_name, image_info);

    std::map<primitive_id, network_output> network_output = _network->execute();
    EXPECT_EQ(network_output.begin()->first, layer_name);
    return network_output.at(layer_name).get_memory();    
}

TEST(simpler_nms, basic) {

    // copy input into cldnn objects
    std::vector<float> cls_scores(cls_scores_data_size);
    for (size_t i = 0 ; i < cls_scores_data_size ; i++) {
        cls_scores[i] = cls_scores_data[i];
    }

    std::vector<float> bbox_pred(bbox_pred_data_size);
    for (size_t i = 0 ; i < bbox_pred_data_size ; i++) {
        bbox_pred[i] = bbox_pred_data[i];
    }

    TestRunnerSimplerNMS<float> t;

    const memory& output = t.Run(cls_scores, bbox_pred);
    EXPECT_EQ((unsigned int)output.get_layout().count(), simpler_nms_ref_size);

    float* f = output.pointer<float>().data();

    for (unsigned int i = 0 ; i < simpler_nms_ref_size ; i++) {
        EXPECT_NEAR(f[i], simpler_nms_ref[i], epsilon);
    }
}


TEST(simpler_nms, fp16) {

    // copy input into cldnn objects
    std::vector<FLOAT16> cls_scores(cls_scores_data_size);
    for (size_t i = 0 ; i < cls_scores_data_size ; i++) {
        cls_scores[i] = cls_scores_data[i];
    }

    std::vector<FLOAT16> bbox_pred(bbox_pred_data_size);
    for (size_t i = 0 ; i < bbox_pred_data_size ; i++) {
        bbox_pred[i] = bbox_pred_data[i];
    }
    
    TestRunnerSimplerNMS<FLOAT16> t;

    const memory& output = t.Run(cls_scores, bbox_pred);
    EXPECT_EQ((unsigned int)output.get_layout().count(), simpler_nms_ref_size);

    half_t* d = output.pointer<half_t>().data();

    for (unsigned int i = 0 ; i < simpler_nms_ref_size ; i++) {
        FLOAT16 f((int16_t)d[i]);
        FLOAT16 ref(simpler_nms_ref[i]);        
        EXPECT_NEAR((float)f, (float)ref, epsilon_fp16);
    }
}
