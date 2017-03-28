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

    std::vector<float> data = {1.0f, 2.0f,
        3.0f, 4.0f};
    std::vector<float> rois = {0.0f, 0.0f, 0.0f, 1.0f, 1.0f};

    int num_rois = (int) rois.size() / CLDNN_ROI_VECTOR_SIZE;

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
    memcpy(&data[0], roi_pooling_data, roi_pooling_data_size * sizeof (float));

    std::vector<float> rois(rois_input_size);
    memcpy(&rois[0], rois_input, rois_input_size * sizeof (float));

    int num_rois = (int) rois_input_size / CLDNN_ROI_VECTOR_SIZE;

    TestRunner<float> t(num_rois, channels, width, height, pooled_width, pooled_height, spatial_scale);

    memory output = t.Run(data, rois);

    EXPECT_EQ(output.get_layout().count(), (unsigned int) (num_rois * channels * pooled_width * pooled_height));

    float* f = output.pointer<float>().data();

    for (unsigned int i = 0; i < roi_pooling_ref_size; i++) {
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
    for (unsigned int i = 0; i < roi_pooling_data_size; i++) {
        data[i] = roi_pooling_data[i];
    }

    std::vector<FLOAT16> rois(rois_input_size);
    for (unsigned int i = 0; i < rois_input_size; i++) {
        rois[i] = rois_input[i];
    }

    int num_rois = (int) rois_input_size / CLDNN_ROI_VECTOR_SIZE;

    TestRunner<FLOAT16> t(num_rois, channels, width, height, pooled_width, pooled_height, spatial_scale);

    memory output = t.Run(data, rois);

    EXPECT_EQ(output.get_layout().count(), (unsigned int) (num_rois * channels * pooled_width * pooled_height));

    half_t* d = output.pointer<half_t>().data();

    for (unsigned int i = 0; i < roi_pooling_ref_size; i++) {
        FLOAT16 f((int16_t) d[i]);
        FLOAT16 ref(roi_pooling_ref[i]);
        EXPECT_FLOAT_EQ((float) f, (float) ref);
    }
}


class roi_pooling_test : public tests::generic_test
{

public:
    
    roi_pooling_test()
    {        
    }

	static void TearDownTestCase() 
	{
		generic_test::TearDownTestCase();

		for (auto generic_params : all_generic_params)
		{
			delete generic_params;
		}

		for (auto layer_params : all_layer_params)
		{
			delete layer_params;
		}
	}
    
	static std::vector<test_params*> generate_input_buffers_params()
	{        
        std::vector<int> test_rois_sizes = { 1, 20 };          
        
		for (cldnn::data_types data_type : test_data_types)
		{
			for (cldnn::format fmt : test_formats)
			{        
                for (int batch_size : test_batch_sizes)
                {
                    for (int feature_size : test_feature_sizes)
                    {
                        for (tensor input_size : test_input_sizes)
                        {
                            for (int num_rois : test_rois_sizes)
                            {
                                test_params* tp = new test_params();

                                tp->data_type = data_type;
                                tp->input_layouts.push_back(cldnn::tensor(fmt,{batch_size, feature_size, input_size.spatial[1], input_size.spatial[0]}));
                                tp->input_layouts.push_back(cldnn::tensor(cldnn::format::bx,{num_rois, CLDNN_ROI_VECTOR_SIZE}));

                                all_generic_params.push_back(tp);
                            }
                        }
                    }
                }
            }
        }

		return all_generic_params;
	}
    

	static std::vector<cldnn::primitive*> generate_layer_params()
	{
        float spatial_scale = 0.0625f;
        
        struct {
            int pooled_width;
            int pooled_height;
        } test_cases[] = {
           // { 1, 1 },
            { 4, 4}
            //{ 10, 30}
        };

        for (size_t i = 0; i < ARRAY_SIZE(test_cases); i++) {
            std::string test_data_name = "input0"; // currently the framework assumes input0, input1,... naming
            std::string test_rois_name = "input1";
            all_layer_params.push_back(new roi_pooling(layer_name,
                    test_data_name,
                    test_rois_name,
                    test_cases[i].pooled_width,
                    test_cases[i].pooled_height,
                    spatial_scale,
                    padding(),
                    padding()));
        }

        return all_layer_params;
    }

    virtual void prepare_input_for_test(std::vector<cldnn::memory>& inputs) 
    {       
        if (generic_params->data_type == data_types::f32) 
        {
            prepare_input_for_test_typed<float>(inputs);
        }
        else 
        {
            prepare_input_for_test_typed<FLOAT16>(inputs);
        }
    }
    
   
    virtual memory generate_reference(const std::vector<cldnn::memory>& inputs) 
    {
        if (generic_params->data_type == data_types::f32) 
        {
            return generate_reference_typed<float>(inputs);
        } 
        else 
        {
            return generate_reference_typed<FLOAT16>(inputs);
        }
    }

    virtual bool is_format_supported(cldnn::format format) 
    {
        return (format == cldnn_format_type::cldnn_format_bfyx || 
                format == cldnn_format_type::cldnn_format_bx );
    }

	virtual cldnn::tensor get_expected_output_tensor()
	{
		return get_output_layout();
	}

    static std::string custom_param_name(const ::testing::TestParamInfo<std::tuple<test_params*, cldnn::primitive*>>& info)
    {
        std::stringstream res;

        const auto & p = std::get<0>(info.param);
        const auto & v = std::get<1>(info.param);

        assert (p->data_type == data_types::f32 ||
                p->data_type == data_types::f16);

        res << info.index
            << "_" << (p->data_type == data_types::f32 ? "f32" : "f16");

        for (unsigned i = 0; i < p->input_layouts.size(); ++i)
        {
            const auto chans = format::traits(p->input_layouts[i].format).order;

            res << "_" << "Input" << i;
            for (unsigned int j = 0; j < p->input_layouts[i].sizes().size(); ++j)
            {
                res << chans[j] << p->input_layouts[i].sizes()[j];
            }
        }

        const auto layer = static_cast<cldnn::roi_pooling *>(v);
        res << "_PooledW" << layer->pooled_width
            << "_PooledH" << layer->pooled_height;
        //TODO: we should remove spatial scale altogether
        //TODO: while it exists it should be escaped in the following
//            << "_SpatialScale" << layer->spatial_scale;

        return res.str();
    }

private:

    template<typename Type>
    void prepare_input_for_test_typed(std::vector<cldnn::memory>& inputs) 
    {    
        Type* bottom_rois = inputs[1].pointer<Type>().data();
        int num_rois = inputs[1].get_layout().size.batch[0];

        for (int n = 0; n < num_rois; ++n) 
        {
            bottom_rois[0] = 0;
            bottom_rois += 5;
        }
    }

    template<typename Type>
    memory generate_reference_typed(const std::vector<cldnn::memory>& inputs) 
    {
        data_types dt = inputs[0].get_layout().data_type;
        auto output = memory::allocate(engine, cldnn::layout(dt, get_output_layout().transform(cldnn::format::bfyx, 0)));

        const cldnn::roi_pooling* roi_layer = (cldnn::roi_pooling*)layer_params;
        int pooled_width = (*roi_layer).pooled_width;
        int pooled_height = (*roi_layer).pooled_height;
        Type spatial_scale = (Type)((*roi_layer).spatial_scale);
        Type* output_mem = output.pointer<Type>().data();
        int num_rois = inputs[1].get_layout().size.batch[0];
        int fm = inputs[0].get_layout().size.feature[0];
        int height = inputs[0].get_layout().size.spatial[1];
        int width = inputs[0].get_layout().size.spatial[0];

        const Type* bottom_data = inputs[0].pointer<Type>().data();
        const Type* bottom_rois = inputs[1].pointer<Type>().data();

        int batch_size = inputs[0].get_layout().size.batch[0];
        //int* argmax_data = max_idx_.mutable_cpu_data();

        // For each ROI R = [batch_index x1 y1 x2 y2]: max pool over R
        for (int n = 0; n < num_rois; ++n) 
        {
            int roi_batch_ind = (int)bottom_rois[0];
            int roi_start_x = (int)round((float)(bottom_rois[1] * spatial_scale));
            int roi_start_y = (int)round((float)(bottom_rois[2] * spatial_scale));
            int roi_end_x = (int)round((float)(bottom_rois[3] * spatial_scale));
            int roi_end_y = (int)round((float)(bottom_rois[4] * spatial_scale));
            EXPECT_GE(roi_batch_ind, 0);
            EXPECT_LT(roi_batch_ind, batch_size);                       
            
            int roi_height = std::max(roi_end_y - roi_start_y + 1, 1);
            int roi_width = std::max(roi_end_x - roi_start_x + 1, 1);
            
            const Type* batch_data = bottom_data + roi_batch_ind * fm * height * width;

            for (int c = 0; c < fm; ++c) 
            {
                for (int ph = 0; ph < pooled_height; ++ph) 
                {
                    for (int pw = 0; pw < pooled_width; ++pw) 
                    {

                        //if (c == 1 && ph == 0 && pw == 0) printf("ROI X %d-%d Y %d-%d\n", roi_start_x, roi_end_x, roi_start_y, roi_end_y);
                        // Compute pooling region for this output unit:
                        //  start (included) = floor(ph * roi_height / pooled_height_)
                        //  end (excluded) = ceil((ph + 1) * roi_height / pooled_height_)

                        // The following computation of ystart, xstart, yend, xend is
                        // done with integers due to floating precision errors.
                        // As the floating point computing on GPU is not identical to CPU,
                        // integer computing is used as a workaround.
                        // The following approach also works but requires a rigorous
                        // analysis:
                        // int ystart = static_cast<int>(floor((static_cast<Dtype>(ph)
                        //                              * static_cast<Dtype>(roi_height))
                        //                            / static_cast<Dtype>(pooled_height_)));
                        // int xstart = static_cast<int>(floor((static_cast<Dtype>(pw)
                        //                              * static_cast<Dtype>(roi_width))
                        //                            / static_cast<Dtype>(pooled_width_)));
                        // int yend = static_cast<int>(ceil((static_cast<Dtype>(ph + 1)
                        //                            * static_cast<Dtype>(roi_height))
                        //                            / static_cast<Dtype>(pooled_height_)));
                        // int xend = static_cast<int>(ceil((static_cast<Dtype>(pw + 1)
                        //                            * static_cast<Dtype>(roi_width))
                        //                            / static_cast<Dtype>(pooled_width_)));

                        int ystart = (ph * roi_height) / pooled_height;
                        int xstart = (pw * roi_width) / pooled_width;
                        int yend = ((ph + 1) * roi_height) / pooled_height;
                        int xend = ((pw + 1) * roi_width) / pooled_width;
                        
                        if ((ystart * pooled_height) > (ph * roi_height)) 
                        {
                            --ystart;
                        }
                        
                        if ((xstart * pooled_width) > (pw * roi_width)) 
                        {
                            --xstart;
                        }
                        
                        if ((yend * pooled_height) < ((ph + 1) * roi_height)) 
                        {
                            ++yend;
                        }
                        
                        if ((xend * pooled_width) < ((pw + 1) * roi_width)) 
                        {
                            ++xend;
                        }
                        
                        ystart = std::min(std::max(ystart + roi_start_y, 0), height);
                        yend = std::min(std::max(yend + roi_start_y, 0), height);
                        xstart = std::min(std::max(xstart + roi_start_x, 0), width);
                        xend = std::min(std::max(xend + roi_start_x, 0), width);

                        bool is_empty = (yend <= ystart) || (xend <= xstart);

                        const int pool_index = ph * pooled_width + pw;
                        if (is_empty) 
                        {
                            output_mem[pool_index] = 0;
                            //   argmax_data[pool_index] = -1;
                        }
						else
						{
							if (sizeof(Type) == 4)
							{
								output_mem[pool_index] = -FLT_MAX;
							}
							else
							{
								output_mem[pool_index] = FLOAT16::min_val();
							}
						}

                        for (int h = ystart; h < yend; ++h) 
                        {
                            for (int w = xstart; w < xend; ++w) 
                            {
                                const int index = h * width + w;
                                Type f1 = batch_data[index];
                                Type f2 = output_mem[pool_index];
                                if (f1 > f2) 
                                {
                                    output_mem[pool_index] = batch_data[index];
                                    //      argmax_data[pool_index] = index;
                                }
                            }
                        }
                    }
                }

                // Increment all data pointers by one channel
                batch_data += width * height;
                output_mem += pooled_width * pooled_height;
                //  argmax_data += max_idx_.offset(0, 1);
            }

            // Increment ROI data pointer
            bottom_rois += 5;
        }

        return output;
    }

private:

    cldnn::tensor get_output_layout() 
    {
        const cldnn::roi_pooling* roi_pooling = (cldnn::roi_pooling*)layer_params;
        
        int fm = generic_params->input_layouts[0].feature[0];
        int num_rois = generic_params->input_layouts[1].batch[0];
        cldnn::tensor output_layout = { format::bfyx, { num_rois, fm, roi_pooling->pooled_height, roi_pooling->pooled_width }};
        return output_layout;
    }

    static std::vector<tests::test_params*> all_generic_params;
    static std::vector<cldnn::primitive*> all_layer_params;
};

std::vector<cldnn::primitive*> roi_pooling_test::all_layer_params = {};
std::vector<tests::test_params*> roi_pooling_test::all_generic_params = {};

TEST_P(roi_pooling_test, test_all) 
{
    run_single_test();
}

INSTANTIATE_TEST_CASE_P(ROI_POOLING,
        roi_pooling_test,
        ::testing::Combine(::testing::ValuesIn(roi_pooling_test::generate_input_buffers_params()),
        ::testing::ValuesIn(roi_pooling_test::generate_layer_params())),
        roi_pooling_test::custom_param_name);
