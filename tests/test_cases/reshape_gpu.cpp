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
#include <gtest/gtest.h>
#include <api/CPP/topology.hpp>
#include <api/CPP/network.hpp>
#include <api/CPP/engine.hpp>

#include <api/CPP/data.hpp>
#include <api/CPP/reshape.hpp>
#include <api/CPP/input_layout.hpp>

#include "test_utils/test_utils.h"

using namespace cldnn;
using namespace tests;
using namespace testing;

template <class ElemType>
void generic_reshape_test(tensor const& input_size, tensor const& reshape_size, bool in_place, padding const& input_padd = padding(), padding const& output_padd = padding())
{
    engine engine;

    //allocate input memory
    //auto padded_input_size = input_size.add(input_padd.lower_size()).add(input_padd.upper_size());
    auto input = memory::allocate(engine, { std::is_same<ElemType, FLOAT16>::value ? data_types::f16 : data_types::f32, input_size });
    
    {
        auto input_ptr = input.cldnn::memory::pointer<ElemType>();
        auto input_itr = input_ptr.begin();

        auto elements = input_size.count();

        int value = 1;
        for (size_t i = 0; i < elements; ++i)
            *input_itr++ = (ElemType)value++;
    }

    topology tpl;
    std::string reshape_input = "input";

    tpl.add(input_layout("input", input.get_layout()));
    if (input_padd)
    {
        tpl.add(reorder("reorder", "input", input.get_layout(), "", padding(), input_padd));
        reshape_input = "reorder";
    }
    tpl.add(reshape("reshape", reshape_input, reshape_size, output_padd));

    //to check whether reshape has actually reused memory from previous primitive, we cannot compare it with 'input' allocate above since
    //input_layout creates its own buffer and copies data during 'network.set_inut_data' call
    //therefore we need to extract output from 'input' primitive as well and check if output from 'reshape' refers to the same underlayin cl::Buffer

    build_options bo;
    //ask for outputs from 'input' and 'reshape'
    bo.set_option(build_option::outputs({ "input", "reshape" }));

    network net(engine, tpl, bo);
    net.set_input_data("input", input);
    auto outputs = net.execute();

    ASSERT_TRUE(outputs.size() == 2 && outputs.count("reshape") == 1 && outputs.count("input") == 1);
    //extract underlaying memory object for both primitives 'input' and 'reshape'
    auto net_input = outputs.at("input").get_memory();
    auto output = outputs.at("reshape").get_memory();

    EXPECT_TRUE(output.get_layout().data_type == input.get_layout().data_type);     //reshape should not change data_type
    EXPECT_TRUE(output.get_layout().size.format == input.get_layout().size.format); //reshape should not change format

    //output size should be equal to requested plus output padding
    ASSERT_TRUE(output.get_layout().size == reshape_size.add(output_padd.lower_size()).add(output_padd.upper_size()));

    if (in_place)
        EXPECT_TRUE(output.is_the_same_buffer(net_input)); //if reshape should operate in place both memories should refer to the same underlaying cl::Buffer
    else
        EXPECT_TRUE(!output.is_the_same_buffer(net_input)); //otherwise they should not

    {
        auto output_ptr = output.pointer<const ElemType>();
        auto output_itr = output_ptr.begin();

        auto sizes = reshape_size.sizes();
        auto lower = output_padd.lower_size().transform(reshape_size.format, 0).sizes();
        auto upper = output_padd.upper_size().transform(reshape_size.format, 0).sizes();
        auto offsets = sizes;
        int32_t accum = 1;
        for (size_t i = 1; i <= sizes.size(); ++i)
        {
            offsets[sizes.size() - i] = accum;
            accum *= lower[sizes.size() - i] + sizes[sizes.size() - i] + upper[sizes.size() - i];
        }

        int value = 1;

        output_itr += lower[0] * offsets[0];
        for (int d1 = 0; d1 < sizes[0]; ++d1)
        {
            output_itr += lower[1] * offsets[1];
            for (int d2 = 0; d2 < sizes[1]; ++d2)
            {
                output_itr += lower[2] * offsets[2];
                for (int d3 = 0; d3 < sizes[2]; ++d3)
                {
                    output_itr += lower[3] * offsets[3];
                    for (int d4 = 0; d4 < sizes[3]; ++d4)
                    {
                        auto& output_value = *output_itr;
                        ++output_itr;
                        EXPECT_FLOAT_EQ(output_value, (ElemType)value);
                        ++value;
                    }

                    output_itr += upper[3] * offsets[3];
                }

                output_itr += upper[2] * offsets[2];
            }

            output_itr += upper[1] * offsets[1];
        }
    }
}

TEST(reshape_gpu_f32, basic_2dim_in_place)
{
    generic_reshape_test<float>(
        tensor(format::bfyx, { 1, 1, 2, 2 }),
        tensor(format::bfyx, { 1, 1, 4, 1 }),
        true);
}

TEST(reshape_gpu_f16, basic_2dim_in_place)
{
    generic_reshape_test<FLOAT16>(
        tensor(format::bfyx, { 1, 1, 2, 2 }),
        tensor(format::bfyx, { 1, 1, 1, 4 }),
        true);
}

TEST(reshape_gpu_f32, basic_4dim_in_place)
{
    generic_reshape_test<float>(
        tensor(format::yxfb, { 2, 4, 9, 9 }),
        tensor(format::yxfb, { 3, 4, 2, 27 }),
        true);
}

TEST(reshape_gpu_f16, basic_4dim_in_place)
{
    generic_reshape_test<FLOAT16>(
        tensor(format::yxfb, { 2, 4, 9, 9 }),
        tensor(format::yxfb, { 27, 2, 4, 3 }),
        true);
}

TEST(reshpape_gpu_f32, basic_2dim_output_padd)
{
    generic_reshape_test<float>(
        tensor(format::byxf, { 1, 4, 2, 1 }),
        tensor(format::byxf, { 1, 8, 1, 1 }),
        false,
        padding(),
        padding({ format::bfyx, { 0,0,1,1 } })
        );
}

TEST(reshape_gpu_f16, basic_2dim_output_padd)
{
    generic_reshape_test<FLOAT16>(
        tensor(format::byxf, { 1, 3, 4, 1 }),
        tensor(format::byxf, { 1, 2, 6, 1 }),
        false,
        padding(),
        padding({ format::bfyx, { 0,0,2,2 } })
        );
}

TEST(reshape_gpu_f32, basic_2dim_input_padd)
{
    generic_reshape_test<float>(
        tensor(format::fyxb, { 1, 2, 5, 1 }),
        tensor(format::fyxb, { 1, 5, 2, 1 }),
        false,
        padding({ format::bfyx, { 0,0,3,2 }, { 0,0,1,4 } })
        );
}

TEST(reshape_gpu_f16, basic_2dim_input_padd)
{
    generic_reshape_test<FLOAT16>(
        tensor(format::fyxb, { 1, 3, 3, 1 }),
        tensor(format::fyxb, { 1, 1, 9, 1 }),
        false,
        padding({ format::bfyx, { 0,0,4,1 }, { 0,0,2,3 } })
        );
}

TEST(reshape_gpu_f32, basic_2dim_input_output_padd)
{
    generic_reshape_test<float>(
        tensor(format::byxf, { 1, 5, 7, 1 }),
        tensor(format::byxf, { 1, 7, 5, 1 }),
        false,
        padding({ format::bfyx, { 0,0,4,4 }, { 0,0,1,1 } }),
        padding({ format::bfyx, { 0,0,0,0 }, { 0,0,3,0 } })
        );
}

TEST(reshape_gpu_f16, basic_2dim_input_output_padd)
{
    generic_reshape_test<FLOAT16>(
        tensor(format::byxf, { 1, 6, 6, 1 }),
        tensor(format::byxf, { 1, 3, 12, 1 }),
        false,
        padding({ format::bfyx, { 0,0,1,1 }, { 0,0,0,0 } }),
        padding({ format::bfyx, { 0,0,2,1 }, { 0,0,1,2 } })
        );
}

TEST(reshpape_gpu_f32, basic_4dim_output_padd)
{
    generic_reshape_test<float>(
        tensor(format::bfyx, { 2, 5, 7, 3 }),
        tensor(format::bfyx, { 1, 14, 15, 1 }),
        false,
        padding(),
        padding({ format::bfyx,{ 1,0,0,1 },{ 0,2,3,0 } })
        );
}

TEST(reshape_gpu_f16, basic_4dim_output_padd)
{
    generic_reshape_test<FLOAT16>(
        tensor(format::bfyx, { 5, 4, 2, 2 }),
        tensor(format::bfyx, { 40, 2, 1, 1 }),
        false,
        padding(),
        padding({ format::bfyx,{ 0,2,0,1 },{ 0,2,3,0 } })
        );
}

TEST(reshape_gpu_f32, basic_4dim_input_padd)
{
    generic_reshape_test<float>(
        tensor(format::yxfb, { 3, 3, 128, 8 }),
        tensor(format::yxfb, { 8, 9, 8, 16 }),
        false,
        padding({ format::bfyx, { 0,1,3,3}, { 0,1,1,1 } })
        );
}

TEST(reshape_gpu_f16, basic_4dim_input_padd)
{
    generic_reshape_test<FLOAT16>(
        tensor(format::yxfb, { 8, 8, 32, 2 }),
        tensor(format::yxfb, { 1, 4, 128, 8 }),
        false,
        padding({ format::bfyx, { 2,2,1,0 }, { 1,2,2,0 } })
        );
}

TEST(reshape_gpu_f32, basic_4dim_input_output_padd)
{
    generic_reshape_test<float>(
        tensor(format::fyxb, { 1024, 25, 25, 8 }),
        tensor(format::fyxb, { 64, 100, 100, 8 }),
        false,
        padding({ format::bfyx, { 2,0,2,1 }, { 0,1,4,0 } }),
        padding({ format::bfyx, { 1,2,3,4 }, { 0,4,1,1 } })
        );
}

TEST(reshape_gpu_f16, basic_4dim_input_output_padd)
{
    generic_reshape_test<FLOAT16>(
        tensor(format::byxf, { 32, 227, 227, 3 }),
        tensor(format::byxf, { 8, 227, 227, 12 }),
        false,
        padding({ format::bfyx, { 0,1,4,4 }, { 0,1,1,1 } }),
        padding({ format::bfyx, { 0,29,29,0 }, { 0,0,0,0 } })
        );
}