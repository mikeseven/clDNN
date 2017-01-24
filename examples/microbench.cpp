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
#include <api/primitives/input_layout.hpp>
#include <api/primitives/convolution.hpp>
#include <api/primitives/data.hpp>

using namespace cldnn;

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
    auto input_size = tensor(format::yxfb, { in_h, in_w, in_feature, batch_size });
    auto weight_size = tensor(format::yxio, { kernel_h, kernel_w, in_feature, out_feature });
    auto bias_size = tensor(format::x, { out_feature });

    auto input =   data(id + "_input", memory::allocate(engine, { data_type, input_size }));
    auto weights = data(id + "_weights", memory::allocate(engine, { data_type, weight_size }));
    auto bias =    data(id + "_bias", memory::allocate(engine, { data_type, bias_size }));

    auto conv = convolution(
        id,
        input,
        { weights },
        { bias },
        offset,
        stride,
        true);

    topology.add(input, weights, bias, conv);
}

cldnn::topology build_microbench(const std::string& weights_dir, const cldnn::engine& engine, cldnn::layout& input_layout, int32_t batch_size)
{
    topology topology;
    bool use_half = input_layout.data_type == data_types::f16 ? true : false;

    test_conv(
        engine,
        topology,
        "conv1_7x7_s2",
        batch_size,
        224, 224, 3, // input: x,y,f
        { format::yx, { -3, -3 } }, // padding
        { format::yx, { 2, 2 } }, // stride
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
        { format::yx, { 0, 0 } }, // padding
        { format::yx,{ 2, 2 } }, // stride
        64, // output feature maps num
        1, 1, // kernel size
        use_half
    );

    test_conv(
        engine,
        topology,
        "conv2_3x3",
        batch_size,
        56, 56, 64, // input: x,y,f
        { format::yx, { -1, -1 } }, // padding
        { format::yx,{ 1, 1 } }, // stride
        192, // output feature maps num
        3, 3, // kernel size
        use_half
    );

    test_conv(
        engine,
        topology,
        "inception_3a_1x1",
        batch_size,
        28, 28, 192, // input: x,y,f
        { format::yx,{ 0, 0 } }, // padding
        { format::yx,{ 1, 1 } }, // stride
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
        { format::yx,{ 0, 0 } }, // padding
        { format::yx,{ 1, 1 } }, // stride
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
        { format::yx,{ -1, -1 } }, // padding
        { format::yx,{ 1, 1 } }, // stride
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
        { format::yx,{ 0, 0 } }, // padding
        { format::yx,{ 1, 1 } }, // stride
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
        { format::yx,{ -1, -1 } }, // padding
        { format::yx,{ 1, 1 } }, // stride
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
        { format::yx,{ -1, -1 } }, // padding
        { format::yx,{ 1, 1 } }, // stride
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
        { format::yx,{ 0, 0 } }, // padding
        { format::yx,{ 1, 1 } }, // stride
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
        { format::yx,{ 0, 0 } }, // padding
        { format::yx,{ 1, 1 } }, // stride
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
        { format::yx,{ -1, -1 } }, // padding
        { format::yx,{ 1, 1 } }, // stride
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
        { format::yx,{ 0, 0 } }, // padding
        { format::yx,{ 1, 1 } }, // stride
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
        { format::yx,{ -1, -1 } }, // padding
        { format::yx,{ 1, 1 } }, // stride
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
        { format::yx,{ -1, -1 } }, // padding
        { format::yx,{ 1, 1 } }, // stride
        96, // output feature maps num
        3, 3, // kernel size
        use_half
    );

    return topology;
}
