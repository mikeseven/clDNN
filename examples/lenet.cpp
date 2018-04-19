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
#include <api/CPP/reorder.hpp>
#include <api/CPP/convolution.hpp>
#include <api/CPP/pooling.hpp>
#include <api/CPP/fully_connected.hpp>
#include <api/CPP/softmax.hpp>
#include <api/CPP/scale.hpp>

using namespace cldnn;

// Building lenet network with loading weights & biases from file
cldnn::topology build_lenet(const std::string& weights_dir, const cldnn::engine& engine, cldnn::layout& input_layout, int32_t batch_size)
{
    // [224x224x3xB] convolution->relu->pooling->lrn [1000xB]
    input_layout.size = { batch_size, 1, 28, 28 };
    auto input = cldnn::input_layout("input", input_layout);

    auto reordered_input = reorder(
        "reorder",
        input,
        { input_layout.data_type, input_layout.format, input_layout.size });

    auto scale_val = memory::allocate(engine, { input_layout.data_type, format::bfyx,{ 1, 1, 1, 1 } });
    auto scale_factor = cldnn::data("scale_factor_val", scale_val);
    auto ptr = scale_val.pointer<float>();
    ptr[0] = 0.00390625f;

    // scale input data
    auto scale_input = scale(
        "scale_input",
        reordered_input,
        scale_factor
        );

    auto conv1_w = file::create({ engine, join_path(weights_dir, "conv1_weights.nnd")});
    auto conv1_b = file::create({ engine, join_path(weights_dir, "conv1_bias.nnd")});
    auto conv1 = convolution("conv1",
        scale_input,
        { conv1_w },
        { conv1_b },
        { 1, 1, 1, 1 },
        { 0, 0, 0, 0 },
        { 1, 1, 1, 1 },
        false);

    auto pool1 = pooling("pool1",
        conv1,
        pooling_mode::max,
        { 1,1,2,2 }, // kernel
        { 1,1,2,2 } // strd
    );

    auto conv2_w = file::create({ engine, join_path(weights_dir, "conv2_weights.nnd")});
    auto conv2_b = file::create({ engine, join_path(weights_dir, "conv2_bias.nnd") });
    auto conv2 = convolution("conv2",
        pool1,
        { conv2_w },
        { conv2_b },
        { 1, 1, 1, 1 },
        { 0, 0, 0, 0 },
        { 1, 1, 1, 1 },
        false);

    auto pool2 = pooling("pool2",
        conv2,
        pooling_mode::max,
        { 1,1,2,2 }, // kernel
        { 1,1,2,2 } // strd
    );

    auto ip1_w = file::create({ engine, join_path(weights_dir, "ip1_weights.nnd")});
    auto ip1_b = file::create({ engine, join_path(weights_dir, "ip1_bias.nnd") });
    auto ip1 = fully_connected("ip1",
        pool2,
        ip1_w,
        ip1_b,
        true,
        0
    );

    auto ip2_w = file::create({ engine, join_path(weights_dir, "ip2_weights.nnd")});
    auto ip2_b = file::create({ engine, join_path(weights_dir, "ip2_bias.nnd") });
    auto ip2 = fully_connected("ip2",
        ip1,
        ip2_w,
        ip2_b,
        false
    );

    auto softmax = cldnn::softmax(
        "output",
        ip2);

    return topology(
        input, reordered_input,
        scale_factor, scale_input,
        conv1, conv1_w, conv1_b, pool1,
        conv2, conv2_w, conv2_b, pool2,
        ip1, ip1_w, ip1_b,
        ip2, ip2_w, ip2_b,
        softmax
        );
}
