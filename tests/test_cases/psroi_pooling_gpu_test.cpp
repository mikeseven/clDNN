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
#include "api/CPP/memory.hpp"
#include <api/CPP/input_layout.hpp>
#include "api/CPP/proposal.hpp"
#include <api/CPP/topology.hpp>
#include <api/CPP/network.hpp>
#include <api/CPP/engine.hpp>
#include "test_utils/test_utils.h"
#include "test_utils/float16.h"
#include <api/CPP/compounds.h>

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
    TestRunner(
            int num_rois,
            int channels,
            int width,
            int height,
            pooling_mode mode,
            int pooled_width,
            int pooled_height,
            float spatial_scale,
            int group_sz);

    memory Run(std::vector<Dtype>& data,
            std::vector<Dtype>& rois);

private:
    engine _engine;
    layout _data_layout;
    layout _rois_layout;
    topology _topology;
    roi_pooling _test_layer;

    std::unique_ptr<network> _network;
};

template <typename Dtype>
TestRunner<Dtype>::TestRunner(
        int num_rois,
        int channels,
        int width,
        int height,
        pooling_mode mode,
        int pooled_width,
        int pooled_height,
        float spatial_scale,
        int group_sz) :
    _data_layout(cldnn::type_to_data_type<Dtype>::value, format::bfyx, { 1, channels, width, height } ),
    _rois_layout(cldnn::type_to_data_type<Dtype>::value, format::bfyx, { num_rois, 1, CLDNN_ROI_VECTOR_SIZE, 1 }),
    _test_layer(roi_pooling( layer_name, 
                data_name, 
                rois_name,
                mode,
                pooled_width,
                pooled_height,
                spatial_scale,
                group_sz,
                { { 0, 0, 0, 0 }, 0 }))
{
    _topology.add(input_layout(data_name, _data_layout));
    _topology.add(input_layout(rois_name, _rois_layout));
    _topology.add(_test_layer);

    _network.reset(new network(_engine, _topology));
}

template <typename Dtype>
memory TestRunner<Dtype>::Run(std::vector<Dtype>& data_vals,
                              std::vector<Dtype>& rois_vals)
{
    EXPECT_EQ(rois_vals.size() % CLDNN_ROI_VECTOR_SIZE, 0u);
    //TODO(ruv): expect eq group_w^2 * ouptut_c == input_c

    memory data = memory::attach(_data_layout, data_vals.data(), data_vals.size());
    memory rois = memory::attach(_rois_layout, rois_vals.data(), rois_vals.size());

    _network->set_input_data(data_name, data);
    _network->set_input_data(rois_name, rois);

    std::map<primitive_id, network_output> network_output = _network->execute();
    EXPECT_EQ(network_output.begin()->first, layer_name);
    return network_output.at(layer_name).get_memory();
}

TEST(psroi_pooling_forward_gpu, basic_test1_max) {

    int channels = 1;
    int width = 2;
    int height = 2;

    auto mode = pooling_mode::max;
    int pooled_width = 1;
    int pooled_height = 1;
    float spatial_scale = 1.0f;
    int group_sz = 1;

    std::vector<float> data = {
        1.0f, 2.0f,
        3.0f, 4.0f
    };
    std::vector<float> rois = {0.0f, 0.0f, 0.0f, 1.0f, 1.0f};

    int num_rois = (int) rois.size() / CLDNN_ROI_VECTOR_SIZE;

    TestRunner<float> t(num_rois, channels, width, height, mode, pooled_width, pooled_height, spatial_scale, group_sz);

    memory output = t.Run(data, rois);

    EXPECT_EQ(output.get_layout().count(), 1u);

    auto f = output.pointer<float>();

    EXPECT_EQ(f[0], 4.0f);
}

TEST(psroi_pooling_forward_gpu, basic_test1_avg) {

    int channels = 8;
    int width = 3;
    int height = 3;
    auto mode = pooling_mode::average;
    int pooled_width = 2;
    int pooled_height = 2;
    int group_sz = 2;

    std::vector<float> data(3 * 3 * 2 * 2 * 2);
    std::iota(data.begin(), data.end(), 1.f);

    float spatial_scale = 1.f / 3;
    std::vector<float> rois = {0, 0, 0, 8, 8};

    int num_rois = (int) rois.size() / CLDNN_ROI_VECTOR_SIZE;

    TestRunner<float> t(num_rois, channels, width, height, mode, pooled_width, pooled_height, spatial_scale, group_sz);

    memory output = t.Run(data, rois);

    EXPECT_EQ(output.get_layout().count(), 8u);

    auto f = output.pointer<float>();

    EXPECT_FLOAT_EQ(f[0], (1+2+4+5)/4.f);
    EXPECT_FLOAT_EQ(f[1], (11+12+14+15)/4.f);
    EXPECT_FLOAT_EQ(f[2], (22+23+25+26)/4.f);
    EXPECT_FLOAT_EQ(f[3], (32+33+35+36)/4.f);

    EXPECT_FLOAT_EQ(f[4], (37+38+40+41)/4.f);
    EXPECT_FLOAT_EQ(f[5], (47+48+50+51)/4.f);
    EXPECT_FLOAT_EQ(f[6], (58+59+61+62)/4.f);
    EXPECT_FLOAT_EQ(f[7], (68+69+71+72)/4.f);
}

//TODO: this is part of std in c++17, remove this!
template <typename T>
static inline T clamp(const T& v, const T& lower, const T& upper)
{
    return std::max(lower, std::min(v, upper));
}

class psroi_pooling_test : public tests::generic_test
{

public:
    
    psroi_pooling_test()
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
        const std::vector<int> test_rois_sizes = { 1, 20 };
        const int group_sz = 4;
        const int gss = group_sz * group_sz;


        for (cldnn::data_types data_type : test_data_types)
//        for (int batch_size : test_batch_sizes)
        for (int feature_size : { gss, 3 * gss })
        for (tensor input_size : test_input_sizes)
        for (int num_rois : test_rois_sizes)
        {
            test_params* tp = new test_params();

            tp->data_type = data_type;
            tp->input_layouts.emplace_back(tp->data_type, tp->fmt, cldnn::tensor(/*batch_size*/1, feature_size, input_size.spatial[0], input_size.spatial[1]));
            tp->input_layouts.emplace_back(tp->data_type, tp->fmt, cldnn::tensor(num_rois, 1, CLDNN_ROI_VECTOR_SIZE, 1));

            all_generic_params.push_back(tp);
        }

        return all_generic_params;
    }

    static std::vector<cldnn::primitive*> generate_layer_params()
    {
        float spatial_scale = 0.0625f;

        const int group_sz = 4;
        
        struct {
            int pooled_width;
            int pooled_height;
        } test_cases[] = {
            { 3, 3 },
            { 4, 4 },
        };

        for (auto mode : { pooling_mode::max, pooling_mode::average })
        for (size_t i = 0; i < ARRAY_SIZE(test_cases); i++)
        {
            std::string test_data_name = "input0"; // currently the framework assumes input0, input1,... naming
            std::string test_rois_name = "input1";
            all_layer_params.push_back(new roi_pooling(layer_name,
                        test_data_name,
                        test_rois_name,
                        mode,
                        test_cases[i].pooled_width,
                        test_cases[i].pooled_height,
                        spatial_scale,
                        group_sz,
                        { { 0, 0, 0, 0}, 0 }));
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
        return (format == cldnn_format_type::cldnn_format_bfyx);
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
            const auto chans = format::traits(p->fmt).order;

            res << "_" << "Input" << i;
            for (unsigned j = 0; j < p->input_layouts[i].size.sizes(p->fmt).size(); ++j)
            {
                res << chans[j] << p->input_layouts[i].size.sizes(p->fmt)[j];
            }
        }

        const auto layer = static_cast<cldnn::roi_pooling *>(v);
        res << (layer->mode == pooling_mode::max ? "_MAX" : "_AVG")
            << "_PooledW" << layer->pooled_width
            << "_PooledH" << layer->pooled_height
            << "_GroupSZ" << layer->group_sz
            << "_SpatialScaleInv" << (1/layer->spatial_scale);

        return res.str();
    }

private:

    template<typename Type>
    void prepare_input_for_test_typed(std::vector<cldnn::memory>& inputs) 
    {    
        auto bottom_rois = inputs[1].pointer<Type>();
        int num_rois = inputs[1].get_layout().size.batch[0];

        for (int n = 0; n < num_rois*5; n+=5) 
        {
            bottom_rois[n] = 0;
        }
    }

    template<typename Type>
    memory generate_reference_typed(const std::vector<cldnn::memory>& inputs) 
    {
        data_types dt = inputs[0].get_layout().data_type;

        const auto layer = static_cast<cldnn::roi_pooling *>(layer_params);

        const bool max_pool = layer->mode == pooling_mode::max;
        const int pooled_w = layer->pooled_width;
        const int pooled_h = layer->pooled_height;
        const int group_sz = layer->group_sz;
        const auto spatial_scale = static_cast<Type>(layer->spatial_scale);

        // Note: Strictly speaking these 2 asserts could be disabled, but this
        //       condition makes some sense otherwise a single output location
        //       will contain data from several channels...
        //TODO: do we want to keep this??
//        assert (pooled_w % group_sz == 0);
//        assert (pooled_h % group_sz == 0);

        auto data_ref = inputs[0].pointer<Type>();
        auto rois_ref = inputs[1].pointer<Type>();
        const Type * const src_data = data_ref.data();
        const Type * const src_rois = rois_ref.data();

        auto output = memory::allocate(engine, cldnn::layout(dt, cldnn::format::bfyx, get_output_layout()));
        auto dst_ref = output.pointer<Type>();
        Type * const dst_data = dst_ref.data();

//        const int data_dim_num = inputs[0].get_layout().size.spatial.size();
//        const int rois_dim_num = inputs[1].get_layout().size.spatial.size();
//        const int dst_dim_num = output.get_layout().size.spatial.size();

//        assert ((data_dim_num == 3 && rois_dim_num == 2 && out_dim_num == 4)/* ||
//                (data_dim_num == 4 && rois_dim_num == 3 && out_dim_num == 5)*/);

        const int data_w = inputs[0].get_layout().size.spatial[0];
        const int data_h = inputs[0].get_layout().size.spatial[1];
        const int data_c = inputs[0].get_layout().size.feature[0];
        const int data_b = inputs[0].get_layout().size.batch[0];  //data_dim_num > 3 ? inputs[0].get_layout().size.spatial[3] : 0;

        const int rois_w = inputs[1].get_layout().size.spatial[0];
        const int rois_r = inputs[1].get_layout().size.batch[0];
//        const int rois_b = inputs[1].get_layout().size.batch[0];  //rois_dim_num > 2 ? inputs[1].get_layout().size.spatial[2] : 0;
        const int rois_b = 1;

        const int dst_w = output.get_layout().size.spatial[0];
        const int dst_h = output.get_layout().size.spatial[1];
        const int dst_c = output.get_layout().size.feature[0];
        const int dst_r = output.get_layout().size.batch[0];
//        const int dst_b = output.get_layout().size.batch[0];  //out_dim_num > 4 ? output.get_layout().size.spatial[4] : 0;
        const int dst_b = 1;

        assert (rois_w == 5);   //TODO: legacy from Caffe: This should be 4 or 2*D
        assert (rois_b == data_b);

        assert (dst_w == pooled_w);
        assert (dst_h == pooled_h);
        assert (dst_c * group_sz * group_sz == data_c);
        assert (dst_r == rois_r);
        assert (dst_b == data_b);

        // Silence "unused-variable" errors in release where assert isn't a (void) cast
        (void) pooled_w;
        (void) pooled_h;
        (void) data_b;
        (void) rois_b;

        using coord_t = float;  //TODO: Type? Then if coords are >1024 we lose precision!
        using accum_t = float;  //TODO: Type? Again, lose precision in accumulation...

        for (int b = 0; b < dst_b; ++b)
        for (int r = 0; r < dst_r; ++r)
        for (int c = 0; c < dst_c; ++c)
        for (int y = 0; y < dst_h; ++y)
        for (int x = 0; x < dst_w; ++x)
        {
            const coord_t batch_idx = src_rois[0 + rois_w * (r + rois_r * b)];  //TODO: get rid of this
            const coord_t roi_x  = (coord_t)(round((float)src_rois[1 + rois_w * (r + rois_r * b)]) + 0) * (coord_t)spatial_scale;
            const coord_t roi_y  = (coord_t)(round((float)src_rois[2 + rois_w * (r + rois_r * b)]) + 0) * (coord_t)spatial_scale; 
            const coord_t roi_x1 = (coord_t)(round((float)src_rois[3 + rois_w * (r + rois_r * b)]) + 1) * (coord_t)spatial_scale; 
            const coord_t roi_y1 = (coord_t)(round((float)src_rois[4 + rois_w * (r + rois_r * b)]) + 1) * (coord_t)spatial_scale;

            const coord_t roi_w = std::max(roi_x1 - roi_x, .1f);
            const coord_t roi_h = std::max(roi_y1 - roi_y, .1f);

            const coord_t dx_begin = (x + 0) * (coord_t)(roi_w / dst_w);
            const coord_t dy_begin = (y + 0) * (coord_t)(roi_h / dst_h);
            const coord_t dx_after = (x + 1) * (coord_t)(roi_w / dst_w);
            const coord_t dy_after = (y + 1) * (coord_t)(roi_h / dst_h);

            const unsigned x_begin = (unsigned)clamp((double)floor(roi_x + dx_begin), 0., (double)data_w);
            const unsigned y_begin = (unsigned)clamp((double)floor(roi_y + dy_begin), 0., (double)data_h);
            const unsigned x_after = (unsigned)clamp((double)ceil(roi_x + dx_after), 0., (double)data_w);
            const unsigned y_after = (unsigned)clamp((double)ceil(roi_y + dy_after), 0., (double)data_h);

#if 0
            const coord_t group_bin_w = (coord_t)roi_w / dst_w;
            const coord_t group_bin_h = (coord_t)roi_h / dst_h;
            
            const unsigned group_x = clamp((int)(x * group_bin_w), 0, group_sz - 1);
            const unsigned group_y = clamp((int)(y * group_bin_h), 0, group_sz - 1);
#else
            const int group_x = x;
            const int group_y = y;
#endif

            const unsigned work_c = group_x + group_sz * (group_y + group_sz * c);

            const Type * const data =
                src_data +
                /* INPUT_PADDING_LOWER_SIZE_X + */
                data_w * ( /*INPUT_PADDING_LOWER_SIZE_Y + */ data_h * (work_c + data_c * b));

            accum_t res = max_pool && x_begin < x_after && y_begin < y_after ? std::numeric_limits<accum_t>::lowest() : (accum_t)0;

            for (unsigned yy = y_begin; yy < y_after; ++yy)
            for (unsigned xx = x_begin; xx < x_after; ++xx)
            {
                const Type val = data[xx + data_w * yy];

                res = max_pool ? std::max(res, (accum_t)val) : res + (accum_t)val;
            }

            if (!max_pool)
            {
                const unsigned area = (y_after - y_begin) * (x_after - x_begin);
                if (area) res /= area;
            }

            dst_data[x + dst_w * (y + dst_h * (c + dst_c * (r + dst_r * b)))] = res;
        }

        return output;
    }

private:
    cldnn::tensor get_output_layout() 
    {
        const auto layer = static_cast<cldnn::roi_pooling *>(layer_params);
        int gss = layer->group_sz * layer->group_sz;
        
        const int data_c = generic_params->input_layouts[0].size.feature[0];
        const int rois_r = generic_params->input_layouts[1].size.batch[0];

        assert (data_c % gss == 0);
        const int dst_c = data_c / gss;

        cldnn::tensor output_layout = { rois_r, dst_c, layer->pooled_width, layer->pooled_height };
        return output_layout;
    }

    static std::vector<tests::test_params*> all_generic_params;
    static std::vector<cldnn::primitive*> all_layer_params;
};

std::vector<cldnn::primitive*> psroi_pooling_test::all_layer_params = {};
std::vector<tests::test_params*> psroi_pooling_test::all_generic_params = {};

TEST_P(psroi_pooling_test, test_all) 
{
    run_single_test();
}

INSTANTIATE_TEST_CASE_P(PSROI_POOLING,
        psroi_pooling_test,
        ::testing::Combine(::testing::ValuesIn(psroi_pooling_test::generate_input_buffers_params()),
        ::testing::ValuesIn(psroi_pooling_test::generate_layer_params())),
        psroi_pooling_test::custom_param_name);
