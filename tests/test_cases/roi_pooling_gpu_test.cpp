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
#include <gtest/gtest.h>
#include <api/memory.hpp>
#include <api/primitives/input_layout.hpp>
#include "api/primitives/roi_pooling.hpp"
#include "api/primitives/simpler_nms.hpp"
#include <api/topology.hpp>
#include <api/network.hpp>
#include <api/engine.hpp>
#include "test_utils/test_utils.h"
#include "test_utils/float16.h"
#include <api/compounds.h>
namespace cldnn
{
template<> struct type_to_data_type<FLOAT16> { static const data_types value = data_types::f16; };
}

using namespace cldnn;
using namespace tests;

extern float roi_pooling_data[];
extern size_t roi_pooling_data_size;
extern float rois_input[];
extern size_t rois_input_size;
extern float roi_pooling_ref[];
extern size_t roi_pooling_ref_size;

const primitive_id data_name = "data";
const primitive_id rois_name = "rois";
const primitive_id layer_name = "roi_pooling";

template <typename Dtype>
class TestRunner 
{
    public:
        TestRunner(int num_rois,
                   int channels,
                   int width,
                   int height,
                   int pooled_width,
                   int pooled_height,
                   float spatial_scale);

        ~TestRunner();

        memory Run(std::vector<Dtype>& data, 
                   std::vector<Dtype>& rois);

    private:

        engine _engine;
        layout _data_layout;
        layout _rois_layout;
        topology _topology;
        roi_pooling _test_layer;
        network* _network = NULL;
};


template <typename Dtype>
TestRunner<Dtype>::TestRunner(
                       int num_rois,
                       int channels,
                       int width,
                       int height,
                       int pooled_width,
                       int pooled_height,
                       float spatial_scale) :
                            _data_layout(cldnn::type_to_data_type<Dtype>::value, { format::bfyx, { 1, channels, height, width } } ),
                            _rois_layout(cldnn::type_to_data_type<Dtype>::value, { format::bx, { num_rois, CLDNN_ROI_VECTOR_SIZE } }),
                            _test_layer(roi_pooling( layer_name, 
                                    data_name, 
                                    rois_name,
                                    pooled_width,
                                    pooled_height,
                                    spatial_scale,
                                    padding(),
                                    padding()))
{    
    _topology.add(input_layout(data_name, _data_layout));
    _topology.add(input_layout(rois_name, _rois_layout));

    _topology.add(_test_layer);

    _network = new network(_engine, _topology);
}

template <typename Dtype>
TestRunner<Dtype>::~TestRunner()
{
    delete _network;
}

template <typename Dtype>
memory TestRunner<Dtype>::Run(std::vector<Dtype>& data_vals, 
                              std::vector<Dtype>& rois_vals)
{
    EXPECT_EQ(rois_vals.size() % CLDNN_ROI_VECTOR_SIZE, 0u);

    memory data = memory::attach(_data_layout, data_vals.data(), data_vals.size());
    memory rois = memory::attach(_rois_layout, rois_vals.data(), rois_vals.size());
    
    _network->set_input_data(data_name, data);
    _network->set_input_data(rois_name, rois);    

    std::map<primitive_id, network_output> network_output = _network->execute();
    EXPECT_EQ(network_output.begin()->first, layer_name);
    return network_output.at(layer_name).get_memory();    
}

TEST(roi_pooling_forward_gpu, basic_test1) {

    int channels = 1;
    int width = 2;
    int height = 2;
    int pooled_width = 1;
    int pooled_height = 1;
    float spatial_scale = 1.0f;

    std::vector<float> data = { 1.0f, 2.0f, 
                                3.0f, 4.0f };
    std::vector<float> rois = { 0.0f, 0.0f, 0.0f, 1.0f, 1.0f };

    int num_rois = (int)rois.size() / CLDNN_ROI_VECTOR_SIZE;

    TestRunner<float> t(num_rois, channels, width, height, pooled_width, pooled_height, spatial_scale);

    memory output = t.Run(data, rois);

    EXPECT_EQ(output.get_layout().count(), 1u);

    float* f = output.pointer<float>().data();

    EXPECT_EQ(f[0], 4.0f);
}


TEST(roi_pooling_forward_gpu, basic_test2) {
    int channels = 256;
    int width = 5;
    int height = 4;
    int pooled_width = 6;
    int pooled_height = 6;
    float spatial_scale = 0.0625f;

    std::vector<float> data(roi_pooling_data_size);
    memcpy(&data[0], roi_pooling_data, roi_pooling_data_size * sizeof(float));

    std::vector<float> rois(rois_input_size);
    memcpy(&rois[0], rois_input, rois_input_size * sizeof(float));

    int num_rois = (int)rois_input_size / CLDNN_ROI_VECTOR_SIZE;

    TestRunner<float> t(num_rois, channels, width, height, pooled_width, pooled_height, spatial_scale);

    memory output = t.Run(data, rois);    

    EXPECT_EQ(output.get_layout().count(), (unsigned int)(num_rois * channels * pooled_width * pooled_height));

    float* f = output.pointer<float>().data();

    for (unsigned int i = 0 ; i < roi_pooling_ref_size ; i++) {
        EXPECT_EQ(f[i], roi_pooling_ref[i]);
    }
}

TEST(roi_pooling_forward_gpu, test_fp16) {
    int channels = 256;
    int width = 5;
    int height = 4;
    int pooled_width = 6;
    int pooled_height = 6;
    float spatial_scale = 0.0625f;

    std::vector<FLOAT16> data(roi_pooling_data_size);
    for (unsigned int i = 0 ; i < roi_pooling_data_size ; i++) {
        data[i] = roi_pooling_data[i];
    }

    std::vector<FLOAT16> rois(rois_input_size);
    for (unsigned int i = 0 ; i < rois_input_size ; i++) {
        rois[i] = rois_input[i];
    }

    int num_rois = (int)rois_input_size / CLDNN_ROI_VECTOR_SIZE;

    TestRunner<FLOAT16> t(num_rois, channels, width, height, pooled_width, pooled_height, spatial_scale);

    memory output = t.Run(data, rois);

    EXPECT_EQ(output.get_layout().count(), (unsigned int)(num_rois * channels * pooled_width * pooled_height));

    half_t* d = output.pointer<half_t>().data();

    for (unsigned int i = 0 ; i < roi_pooling_ref_size ; i++) {
        FLOAT16 f((int16_t)d[i]);
        FLOAT16 ref(roi_pooling_ref[i]);        
        EXPECT_FLOAT_EQ((float)f, (float)ref);
    }
}
