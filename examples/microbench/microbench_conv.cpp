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

#include "common_tools.h"
#include "file.h"
#include <cmath>
#include <string>
#include <api/CPP/input_layout.hpp>
#include <api/CPP/convolution.hpp>
#include <api/CPP/data.hpp>
#include <api/CPP/input_layout.hpp>

using namespace cldnn;




void test_conv(
    const engine& engine,
    topology& topology,
    std::map<primitive_id, layout>& inputs,
    const primitive_id& id,
    int32_t batch_size,
    int32_t in_w,
    int32_t in_h,
    int32_t in_feature,
    const tensor& offset,
    const tensor& stride,
    int32_t out_feature,
    int32_t kernel_w,
    int32_t kernel_h,
    bool use_half)
{
    auto data_type = use_half ? data_types::f16 : data_types::f32;
    auto input_size = tensor(batch_size, in_feature, in_w, in_h);

    auto weight_size = tensor(out_feature, in_feature, kernel_w, kernel_h);
    auto bias_size = tensor(1, 1, out_feature, 1);

    primitive_id input_id = id + "_input";
    layout input_lay{ data_type, format::bfyx, input_size };

    primitive_id weights_id = id + "_weights";
    layout weights_lay{ data_type, format::yxfb, weight_size };

    primitive_id bias_id = id + "_bias";
    layout bias_lay{ data_type, format::bfyx, bias_size };

    inputs.insert({ input_id, input_lay });
    inputs.insert({ weights_id, weights_lay });
    inputs.insert({ bias_id, bias_lay });

    auto conv = convolution(
        id,
        input_id,
        { weights_id},
        { bias_id },
        stride,
        offset,
        { 1,1,1,1 },
        true);

    topology.add(input_layout(input_id, input_lay), input_layout(weights_id, weights_lay), input_layout(bias_id, bias_lay), conv);
}

cldnn::topology build_microbench_conv(const std::string&, const cldnn::engine& engine, std::map<primitive_id, cldnn::layout>& inputs, int32_t batch_size)
{
    topology topology;
    bool use_half = inputs.at("input_layout").data_type == data_types::f16 ? true : false;

    //not used but needs to be initialized
    inputs.at("input_layout").size = { 1, 1, 1, 1 };

    test_conv(
        engine,
        topology,
        inputs,
        "conv1_7x7_s2",
        batch_size,
        224, 224, 3, // input: x,y,f
        { 0, 0, -3, -3 }, //input offset
        { 1, 1, 2, 2 }, // stride
        64, // output feature maps num
        7, 7, // kernel size
        use_half
    );

    test_conv(
        engine,
        topology,
        inputs,
        "conv2_3x3_reduce",
        batch_size,
        56, 56, 64, // input: x,y,f
        { 0, 0, 0, 0 }, //input offset
        { 1, 1, 2, 2 }, // stride
        64, // output feature maps num
        1, 1, // kernel size
        use_half
    );

    test_conv(
        engine,
        topology,
        inputs,
        "conv2_3x3",
        batch_size,
        56, 56, 384, // input: x,y,f
        { 0, 0, 0, 0 }, //input offset
        { 1, 1, 2, 2 }, // stride
        384, // output feature maps num
        3, 3, // kernel size
        use_half
    );

    test_conv(
        engine,
        topology,
        inputs,
        "inception_3a_1x1",
        batch_size,
        28, 28, 192, // input: x,y,f
        { 0, 0, 0, 0 }, //input offset
        { 1, 1, 1, 1 }, // stride
        64, // output feature maps num
        1, 1, // kernel size
        use_half
    );

    test_conv(
        engine,
        topology,
        inputs,
        "inception_3a_3x3_reduce",
        batch_size,
        28, 28, 64, // input: x,y,f
        { 0, 0, 0, 0 }, //input offset
        { 1, 1, 1, 1 }, // stride
        64, // output feature maps num
        1, 1, // kernel size
        use_half
    );

    test_conv(
        engine,
        topology,
        inputs,
        "inception_3a_3x3",
        batch_size,
        28, 28, 192, // input: x,y,f
        { 0, 0, -1, -1 }, //input offset
        { 1, 1, 1, 1 }, // stride
        64, // output feature maps num
        3, 3, // kernel size
        use_half
    );

    test_conv(
        engine,
        topology,
        inputs,
        "inception_3a_double3x3_reduce",
        batch_size,
        28, 28, 192, // input: x,y,f
        { 0, 0, 0, 0 }, //input offset
        { 1, 1, 1, 1 }, // stride
        64, // output feature maps num
        1, 1, // kernel size
        use_half
    );

    test_conv(
        engine,
        topology,
        inputs,
        "inception_3a_double3x3a",
        batch_size,
        28, 28, 64, // input: x,y,f
        { 0, 0, -1, -1 }, //input offset
        { 1, 1, 1, 1 }, // stride
        96, // output feature maps num
        3, 3, // kernel size
        use_half
    );

    test_conv(
        engine,
        topology,
        inputs,
        "inception_3a_double3x3b",
        batch_size,
        28, 28, 96, // input: x,y,f
        { 0, 0, -1, -1 }, //input offset
        { 1, 1, 1, 1 }, // stride
        96, // output feature maps num
        3, 3, // kernel size
        use_half
    );

    test_conv(
        engine,
        topology,
        inputs,
        "inception_3b_1x1",
        batch_size,
        28, 28, 256, // input: x,y,f
        { 0, 0, 0, 0 }, //input offset
        { 1, 1, 1, 1 }, // stride
        64, // output feature maps num
        1, 1, // kernel size
        use_half
    );

    test_conv(
        engine,
        topology,
        inputs,
        "inception_3b_3x3_reduce",
        batch_size,
        28, 28, 64, // input: x,y,f
        { 0, 0, 0, 0 }, //input offset
        { 1, 1, 1, 1 }, // stride
        64, // output feature maps num
        1, 1, // kernel size
        use_half
    );

    test_conv(
        engine,
        topology,
        inputs,
        "inception_3b_3x3",
        batch_size,
        28, 28, 192, // input: x,y,f
        { 0, 0, -1, -1 }, //input offset
        { 1, 1, 1, 1 }, // stride
        64, // output feature maps num
        3, 3, // kernel size
        use_half
    );


    test_conv(
        engine,
        topology,
        inputs,
        "inception_3b_double3x3_reduce",
        batch_size,
        28, 28, 192, // input: x,y,f
        { 0, 0, 0, 0 }, //input offset
        { 1, 1, 1, 1 }, // stride
        64, // output feature maps num
        1, 1, // kernel size
        use_half
    );

    test_conv(
        engine,
        topology,
        inputs,
        "inception_3b_double3x3a",
        batch_size,
        28, 28, 64, // input: x,y,f
        { 0, 0, -1, -1 }, //input offset
        { 1, 1, 1, 1 }, // stride
        96, // output feature maps num
        3, 3, // kernel size
        use_half
    );

    test_conv(
        engine,
        topology,
        inputs,
        "inception_3b_double3x3b",
        batch_size,
        28, 28, 96, // input: x,y,f
        { 0, 0, -1, -1 }, //input offset
        { 1, 1, 1, 1 }, // stride
        96, // output feature maps num
        3, 3, // kernel size
        use_half
    );

    return topology;
}
