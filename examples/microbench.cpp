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

#include "common/common_tools.h"
#include "file.h"


#include <string>
#include <api/CPP/input_layout.hpp>
#include <api/CPP/convolution.hpp>
#include <api/CPP/data.hpp>

using namespace cldnn;


typedef enum
{
    zero = 0,
    one,
    zero_to_nine, 
} filler_type;

template<typename T>
void fill_memory(const memory& memory, filler_type fill)
{

    auto mem_ptr = memory.pointer<T>();
    float val = (fill == filler_type::zero) ? 0.0f : 1.0f;
    for (auto& it : mem_ptr)
    {
        if (fill == zero_to_nine)
            val += fmod(val + 1.0f, 10.0f);
        it = T(val);
    }
}

void test_conv(
    const engine& engine,
    topology& topology,
    const primitive_id& id,
    int32_t batch_size,
    int32_t in_w,
    int32_t in_h,
    int32_t in_feature,
    const padding& offset,
    const tensor& stride,
    int32_t out_feature,
    int32_t kernel_w,
    int32_t kernel_h,
    bool use_half)
{
    auto data_type = use_half ? data_types::f16 : data_types::f32;
    auto input_size = tensor(format::bfyx, { batch_size, in_feature, in_h, in_w });

    auto weight_size = tensor(format::yxio, { kernel_h, kernel_w, in_feature, out_feature });
    auto bias_size = tensor(format::bfyx, { 1, 1, 1, out_feature });

    auto input = memory::allocate(engine, { data_type, input_size });
    auto weights = memory::allocate(engine, { data_type, weight_size });
    auto bias = memory::allocate(engine, { data_type, bias_size });

    // to get different type of weights, biases and input values changed here. Feel free to add new type of fillers
    auto fill_with = filler_type::zero;
    if (!use_half)
    {
        fill_memory<float>(input, fill_with);
        fill_memory<float>(weights, fill_with);
        fill_memory<float>(bias, fill_with);
    }
    else
    {
        fill_memory<half_t>(input, fill_with);
        fill_memory<half_t>(weights, fill_with);
        fill_memory<half_t>(bias, fill_with);
    }
    auto conv = convolution(
        id,
        id + "_input",
        { id + "_weights" },
        { id + "_bias" },
        offset,
        stride,
		{ format::yx,{ 1,1 } },
        true);

    topology.add(data(id + "_input", input), data(id + "_weights", weights), data(id + "_bias", bias), conv);
}
cldnn::topology build_microbench(const std::string&, const cldnn::engine& engine, cldnn::layout& input_layout, int32_t batch_size)
{
    topology topology;
    bool use_half = input_layout.data_type == data_types::f16 ? true : false;
  
    //not used but needs to be initialized
    input_layout.size = { format::byxf,{ 1, 1, 1, 1 } };

    test_conv(
        engine,
        topology,
        "conv1_7x7_s2",
        batch_size,
        224, 224, 3, // input: x,y,f
        { format::bfyx, { 0, 0, -3, -3 } }, // padding
        { format::bfyx, { 1, 1, 2, 2 } }, // stride
        64, // output feature maps num
        7, 7, // kernel size
        use_half
    );

    test_conv(
        engine,
        topology,
        "conv2_3x3_reduce",
        batch_size,
        56, 56, 64, // input: x,y,f
        { format::bfyx, { 0, 0, 0, 0 } }, // padding
        { format::bfyx, { 1, 1, 2, 2 } }, // stride
        64, // output feature maps num
        1, 1, // kernel size
        use_half
    );

    test_conv(
        engine,
        topology,
        "conv2_3x3",
        batch_size,
        56, 56, 384, // input: x,y,f
        { format::bfyx, { 0, 0, 0, 0 } }, // padding
        { format::bfyx, { 1, 1, 2, 2 } }, // stride
        384, // output feature maps num
        3, 3, // kernel size
        use_half
    );

    test_conv(
        engine,
        topology,
        "inception_3a_1x1",
        batch_size,
        28, 28, 192, // input: x,y,f
        { format::bfyx, { 0, 0, 0, 0 } }, // padding
        { format::bfyx, { 1, 1, 1, 1 } }, // stride
        64, // output feature maps num
        1, 1, // kernel size
        use_half
    );

    test_conv(
        engine,
        topology,
        "inception_3a_3x3_reduce",
        batch_size,
        28, 28, 64, // input: x,y,f
        { format::bfyx, { 0, 0, 0, 0 } }, // padding
        { format::bfyx, { 1, 1, 1, 1 } }, // stride
        64, // output feature maps num
        1, 1, // kernel size
        use_half
    );

    test_conv(
        engine,
        topology,
        "inception_3a_3x3",
        batch_size,
        28, 28, 192, // input: x,y,f
        { format::bfyx, { 0, 0, -1, -1 } }, // padding
        { format::bfyx, { 1, 1, 1, 1 } }, // stride
        64, // output feature maps num
        3, 3, // kernel size
        use_half
    );

    test_conv(
        engine,
        topology,
        "inception_3a_double3x3_reduce",
        batch_size,
        28, 28, 192, // input: x,y,f
        { format::bfyx, { 0, 0, 0, 0 } }, // padding
        { format::bfyx, { 1, 1, 1, 1 } }, // stride
        64, // output feature maps num
        1, 1, // kernel size
        use_half
    );

    test_conv(
        engine,
        topology,
        "inception_3a_double3x3a",
        batch_size,
        28, 28, 64, // input: x,y,f
        { format::bfyx, { 0, 0, -1, -1 } }, // padding
        { format::bfyx, { 1, 1, 1, 1 } }, // stride
        96, // output feature maps num
        3, 3, // kernel size
        use_half
    );

    test_conv(
        engine,
        topology,
        "inception_3a_double3x3b",
        batch_size,
        28, 28, 96, // input: x,y,f
        { format::bfyx, { 0, 0, -1, -1 } }, // padding
        { format::bfyx, { 1, 1, 1, 1 } }, // stride
        96, // output feature maps num
        3, 3, // kernel size
        use_half
    );

    test_conv(
        engine,
        topology,
        "inception_3b_1x1",
        batch_size,
        28, 28, 256, // input: x,y,f
        { format::bfyx, { 0, 0, 0, 0 } }, // padding
        { format::bfyx, { 1, 1, 1, 1 } }, // stride
        64, // output feature maps num
        1, 1, // kernel size
        use_half
    );

    test_conv(
        engine,
        topology,
        "inception_3b_3x3_reduce",
        batch_size,
        28, 28, 64, // input: x,y,f
        { format::bfyx, { 0, 0, 0, 0 } }, // padding
        { format::bfyx, { 1, 1, 1, 1 } }, // stride
        64, // output feature maps num
        1, 1, // kernel size
        use_half
    );

    test_conv(
        engine,
        topology,
        "inception_3b_3x3",
        batch_size,
        28, 28, 192, // input: x,y,f
        { format::bfyx, { 0, 0, -1, -1 } }, // padding
        { format::bfyx, { 1, 1, 1, 1 } }, // stride
        64, // output feature maps num
        3, 3, // kernel size
        use_half
    );


    test_conv(
        engine,
        topology,
        "inception_3b_double3x3_reduce",
        batch_size,
        28, 28, 192, // input: x,y,f
        { format::bfyx, { 0, 0, 0, 0 } }, // padding
        { format::bfyx, { 1, 1, 1, 1 } }, // stride
        64, // output feature maps num
        1, 1, // kernel size
        use_half
    );

    test_conv(
        engine,
        topology,
        "inception_3b_double3x3a",
        batch_size,
        28, 28, 64, // input: x,y,f
        { format::bfyx, { 0, 0, -1, -1 } }, // padding
        { format::bfyx, { 1, 1, 1, 1 } }, // stride
        96, // output feature maps num
        3, 3, // kernel size
        use_half
    );

    test_conv(
        engine,
        topology,
        "inception_3b_double3x3b",
        batch_size,
        28, 28, 96, // input: x,y,f
        { format::bfyx, { 0, 0, -1, -1 } }, // padding
        { format::bfyx, { 1, 1, 1, 1 } }, // stride
        96, // output feature maps num
        3, 3, // kernel size
        use_half
    );

    //not used but needs to be in the topology
    auto output = data("output", memory::allocate(engine, { input_layout.data_type, tensor(format::yxfb,{ 1, 1, 1, 1 }) }));
    topology.add(output);

    return topology;
}
