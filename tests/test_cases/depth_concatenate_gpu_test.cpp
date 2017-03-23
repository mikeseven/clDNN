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
#include "api/primitives/depth_concatenate.hpp"
#include <api/topology.hpp>
#include <api/network.hpp>
#include <api/engine.hpp>
#include "test_utils/test_utils.h"

using namespace cldnn;
using namespace tests;

TEST(depth_concatenate_f32_gpu, test01) {
    //  Input count : 2
    //  Input1 : 2x 1x1 x 2
    //  Input2 : 2x 1x1 x 3
    //
    //  Input1:
    //  0.5  0.7  :f0
    //  0.2  0.4  :f1
    //
    //  Input2:
    //  1    0.1  :f0
    //  0.3 -0.5  :f1
    //  0   -0.2  :f2
    //
    //  Output:
    //  0.5  0.7  :f0
    //  0.2  0.4  :f1
    //  1    0.1  :f2
    //  0.3 -0.5  :f3
    //  0   -0.2  :f4
    //

    engine engine;
    auto input1 = memory::allocate(engine, {data_types::f32, tensor(format::yxfb, { 1,1,2,2 })});
    auto input2 = memory::allocate(engine, { data_types::f32, tensor(format::yxfb,{ 1,1,3,2 })});

    set_values(input1, { 0.5f, 0.7f, 0.2f, 0.4f });
    set_values(input2, { 1.0f, 0.1f, 0.3f, -0.5f, 0.0f, -0.2f });

    topology topology;
    topology.add(input_layout("input1", input1.get_layout()));
    topology.add(input_layout("input2", input2.get_layout()));
    topology.add(depth_concatenate("depth1", { "input1", "input2" }));

    network network(engine, topology);

    network.set_input_data("input1", input1);
    network.set_input_data("input2", input2);

    auto outputs = network.execute({});
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "depth1");

    auto output = outputs.at("depth1").get_memory();

    auto output_ptr = output.pointer<float>();
    EXPECT_FLOAT_EQ(0.5f, output_ptr[0]);
    EXPECT_FLOAT_EQ(0.7f, output_ptr[1]);
    EXPECT_FLOAT_EQ(0.2f, output_ptr[2]);
    EXPECT_FLOAT_EQ(0.4f, output_ptr[3]);
    EXPECT_FLOAT_EQ(1.0f, output_ptr[4]);
    EXPECT_FLOAT_EQ(0.1f, output_ptr[5]);
    EXPECT_FLOAT_EQ(0.3f, output_ptr[6]);
    EXPECT_FLOAT_EQ(-0.5f, output_ptr[7]);
    EXPECT_FLOAT_EQ(0.0f, output_ptr[8]);
    EXPECT_FLOAT_EQ(-0.2f, output_ptr[9]);
}

//////////////////////////////////////////////////////////////////////////////
//                                                                          //
//                      Exhaustive Negative Matrix tests                    //
//                                                                          //
//////////////////////////////////////////////////////////////////////////////

//TODO: this should be done using TEST_P or some equivallent construct
static network setup_depth_concatatenate_network(const std::vector<data_types> dts, const std::vector<tensor> ts)
{
    assert(dts.size() == ts.size());
    const size_t sz = ts.size();

    engine engine;
    topology topology;

    std::vector<std::string> input_names;
    input_names.resize(sz);

    for (size_t i = 0; i < sz; ++i)
    {
        auto input = memory::allocate(engine, { dts[i], ts[i] });

        input_names[i] = "input";
        input_names[i] += std::to_string(i);

        topology.add(input_layout(input_names[i], input.get_layout()));
    }
    //TODO: ask Uzi if something tests cases where there's missing input_names (nodes not present in the topology, etc.)
    topology.add(depth_concatenate("depth_concat_node", input_names));

    return network(engine, topology);
}

TEST(NegativeDepthConcatenateTest, DISABLED_TestAll) {
    auto d = data_types::f32;
    auto od = data_types::f16;

    auto f = format::bfyx;
    auto of = format::yxfb;

    std::vector<int> t { 1, 2, 3, 4 };
    std::vector<int> t0 { 7, 2, 3, 4 };
    std::vector<int> t1 { 1, 2, 7, 4 };
    std::vector<int> t2 { 1, 2, 3, 7 };

    //TODO: should be ASSERT_THROW(statement, exception_type) - but what exception type?
    ASSERT_ANY_THROW(setup_depth_concatatenate_network({ }, { }));

    ASSERT_ANY_THROW(setup_depth_concatatenate_network({ d, od }, { tensor(f, t), tensor(f, t) }));
    ASSERT_ANY_THROW(setup_depth_concatatenate_network({ d, d }, { tensor(f, t), tensor(of, t) }));
    ASSERT_ANY_THROW(setup_depth_concatatenate_network({ d, d }, { tensor(f, t), tensor(f, t0) }));
    ASSERT_ANY_THROW(setup_depth_concatatenate_network({ d, d }, { tensor(f, t), tensor(f, t1) }));
    ASSERT_ANY_THROW(setup_depth_concatatenate_network({ d, d }, { tensor(f, t), tensor(f, t2) }));

    ASSERT_ANY_THROW(setup_depth_concatatenate_network({ d, od, d }, { tensor(f, t), tensor(f, t), tensor(f, t) }));
    ASSERT_ANY_THROW(setup_depth_concatatenate_network({ d, d, od }, { tensor(f, t), tensor(f, t), tensor(f, t) }));
    ASSERT_ANY_THROW(setup_depth_concatatenate_network({ d, d, d }, { tensor(f, t), tensor(of, t), tensor(f, t) }));
    ASSERT_ANY_THROW(setup_depth_concatatenate_network({ d, d, d }, { tensor(f, t), tensor(f, t), tensor(of, t) }));
    ASSERT_ANY_THROW(setup_depth_concatatenate_network({ d, d, d }, { tensor(f, t), tensor(f, t0), tensor(f, t) }));
    ASSERT_ANY_THROW(setup_depth_concatatenate_network({ d, d, d }, { tensor(f, t), tensor(f, t1), tensor(f, t) }));
    ASSERT_ANY_THROW(setup_depth_concatatenate_network({ d, d, d }, { tensor(f, t), tensor(f, t2), tensor(f, t) }));
    ASSERT_ANY_THROW(setup_depth_concatatenate_network({ d, d, d }, { tensor(f, t), tensor(f, t), tensor(f, t0) }));
    ASSERT_ANY_THROW(setup_depth_concatatenate_network({ d, d, d }, { tensor(f, t), tensor(f, t), tensor(f, t1) }));
    ASSERT_ANY_THROW(setup_depth_concatatenate_network({ d, d, d }, { tensor(f, t), tensor(f, t), tensor(f, t2) }));
}

//////////////////////////////////////////////////////////////////////////////
//                                                                          //
//                      Exhaustive Positive Matrix tests                    //
//                                                                          //
//////////////////////////////////////////////////////////////////////////////

using namespace cldnn;

class depth_concatenate_test : public tests::generic_test
{

public:

    static void TearDownTestCase()
    {
        for (auto generic_params : all_generic_params)
        {
            delete generic_params;
        }

        for (auto layer_params : all_layer_params)
        {
            delete layer_params;
        }
    }

    virtual void print_params() override
    {
        std::cout
            << "Layer params:"
            << "\t{ data_type: " << (generic_params->data_type == data_types::f32 ? "f32" : "f16")
            << ", inputs: [";
        for (unsigned i = 0; i < generic_params->input_layouts.size(); ++i)
            std::cout
                << " { fmt: " << (generic_params->input_layouts[i].format == cldnn::format::bfyx ? "bfyx" : "???")
                << ", b: " << generic_params->input_layouts[i].sizes()[0]
                << ", f: " << generic_params->input_layouts[i].sizes()[1]
                << ", y: " << generic_params->input_layouts[i].sizes()[2]
                << ", x: " << generic_params->input_layouts[i].sizes()[3]
                << " },";
        std::cout << " ] }" << std::endl;
    }

    static std::vector<cldnn::primitive*> generate_specific_test_params(int i)
  {
        std::vector<cldnn::primitive*> all_layer_params;

        switch(i)
        {
            case 1 : all_layer_params.push_back(new depth_concatenate("depth_concatenate", {"input0"})); break;
            case 2 : all_layer_params.push_back(new depth_concatenate("depth_concatenate", {"input0", "input1"})); break;
            case 3 : all_layer_params.push_back(new depth_concatenate("depth_concatenate", {"input0", "input1", "input2"})); break;
          default: assert(0);
        }

        return all_layer_params;
    }

    static std::vector<tests::test_params*> generate_generic_test_params(int input_count)
    {
        std::vector<tests::test_params*> all_generic_params;

        for (cldnn::data_types dt : test_data_types)
        for (cldnn::format fmt : test_formats)
        for (int32_t b : test_batch_sizes)
        for (tensor & t : test_input_sizes)
        {
            const int w = t.spatial[0];
            const int h = t.spatial[1];

            switch(input_count)
            {
                case 1:
                    for(auto f0 : test_feature_sizes)
                    {
                        test_params * tp = new test_params();
                        tp->data_type = dt;

                        tp->input_layouts.push_back( cldnn::tensor( fmt, { b, f0, h, w }) );

                        all_generic_params.emplace_back(tp);
                    }
                    break;
                case 2:
                    for(auto f0 : test_feature_sizes)
                    for(auto f1 : test_feature_sizes)
                    {
                        test_params * tp = new test_params();
                        tp->data_type = dt;

                        tp->input_layouts.push_back( cldnn::tensor( fmt, { b, f0, h, w }) );
                      tp->input_layouts.push_back( cldnn::tensor( fmt, { b, f1, h, w }) );

                        all_generic_params.emplace_back(tp);
                    }
                    break;
                case 3:
                    for(auto f0 : test_feature_sizes)
                    for(auto f1 : test_feature_sizes)
                    for(auto f2 : test_feature_sizes)
                    {
                        test_params * tp = new test_params();
                        tp->data_type = dt;

                        tp->input_layouts.push_back( cldnn::tensor( fmt, { b, f0, h, w }) );
                        tp->input_layouts.push_back( cldnn::tensor( fmt, { b, f1, h, w }) );
                        tp->input_layouts.push_back( cldnn::tensor( fmt, { b, f2, h, w }) );

                        all_generic_params.emplace_back(tp);
                    }
                    break;
                default:
                    assert(0);
            }
        }

        return all_generic_params;
    }

    static std::vector<std::tuple<test_params*, cldnn::primitive*>> generate_all_test_params()
    {
        std::vector<std::tuple<test_params*, cldnn::primitive*>> res;

        for (int i = 1; i <= 3; ++i)
        {
            auto tpv = generate_generic_test_params(i); 
            auto pv = generate_specific_test_params(i);

            all_generic_params.insert(all_generic_params.end(), tpv.begin(), tpv.end());
            all_layer_params.insert(all_layer_params.end(), pv.begin(), pv.end());

            for (auto & tp : tpv)
            for (auto & p: pv)
                res.emplace_back(tp, p);
        }

        return res;
    }

    virtual bool is_format_supported(cldnn::format format) override
    {
        return format == cldnn_format_type::cldnn_format_bfyx;
    }

    template<typename Type>
    memory generate_reference_typed(const std::vector<memory> & inputs)
    {
        assert(!inputs.empty());

        const int in_b = inputs[0].get_layout().size.transform(cldnn::format::bfyx, 0).sizes()[0];
        const int in_h = inputs[0].get_layout().size.transform(cldnn::format::bfyx, 0).sizes()[2];
        const int in_w = inputs[0].get_layout().size.transform(cldnn::format::bfyx, 0).sizes()[3];

        int out_f = 0;

        for (const memory & input : inputs)
        {
            assert(input.get_layout().size.transform(cldnn::format::bfyx, 0).sizes()[0] == in_b);
            assert(input.get_layout().size.transform(cldnn::format::bfyx, 0).sizes()[2] == in_h);
            assert(input.get_layout().size.transform(cldnn::format::bfyx, 0).sizes()[3] == in_w);

            out_f += input.get_layout().size.transform(cldnn::format::bfyx, 0).sizes()[1];

            assert(input.get_layout().data_type == inputs[0].get_layout().data_type);
            assert(input.get_layout().size.format == inputs[0].get_layout().size.format);
        }

        //Output is bfyx
        auto output = memory::allocate(engine, cldnn::layout(inputs[0].get_layout().data_type, tensor(cldnn::format::bfyx, { in_b, out_f, in_h, in_w })));
        Type * const out_mem = output.pointer<Type>().data();

        int out_f_off = 0;
        for (const memory & input : inputs)
        {
            const int in_f = input.get_layout().size.transform(cldnn::format::bfyx, 0).sizes()[1];
            const Type * const in_mem = input.pointer<Type>().data();

            for (int n = 0; n < in_b; ++n)
            for (int f = 0; f < in_f; ++f)
            for (int y = 0; y < in_h; ++y)
            for (int x = 0; x < in_w; ++x)
            {
                const size_t in_idx = get_linear_index(input.get_layout(), n, f, y, x);
                const size_t out_idx = get_linear_index(output.get_layout(), n, out_f_off + f, y, x);

                out_mem[out_idx] = in_mem[in_idx];
            }

            out_f_off += in_f;
        }

        return output;
    }

    virtual memory generate_reference(const std::vector<memory> & inputs) override
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

    static std::string custom_param_name(const ::testing::TestParamInfo<std::tuple<test_params*, cldnn::primitive*>>& info)
    {
        std::stringstream res;

        const auto & generic_params = std::get<0>(info.param);

        res << info.index
            << "_DT" << (generic_params->data_type == data_types::f32 ? "f32" : "f16");

        for (unsigned i = 0; i < generic_params->input_layouts.size(); ++i)
            res << "_" << i << "InputFMT" << (generic_params->input_layouts[i].format == cldnn::format::bfyx ? "bfyx" : "other")
                << "_" << i << "InputDims" << generic_params->input_layouts[i].sizes()[0]
                << "x" << generic_params->input_layouts[i].sizes()[1]
                << "x" << generic_params->input_layouts[i].sizes()[2]
                << "x" << generic_params->input_layouts[i].sizes()[3];

        return res.str();
    }

private:

    static std::vector<tests::test_params*> all_generic_params;
    static std::vector<cldnn::primitive*> all_layer_params;

};

std::vector<cldnn::primitive*> depth_concatenate_test::all_layer_params = {};
std::vector<tests::test_params*> depth_concatenate_test::all_generic_params = {};

TEST_P(depth_concatenate_test, DISABLED_TestAll)
{
    run_single_test();
}
 
INSTANTIATE_TEST_CASE_P(DEPTHCONCATENATE,
    depth_concatenate_test,
    ::testing::ValuesIn(depth_concatenate_test::generate_all_test_params()),
    depth_concatenate_test::custom_param_name);

