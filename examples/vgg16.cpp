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
#include <string>
#include <api/primitives/input_layout.hpp>
#include <api/primitives/reorder.hpp>
#include <api/primitives/convolution.hpp>
#include <api/primitives/pooling.hpp>
#include <api/primitives/normalization.hpp>
#include <api/primitives/fully_connected.hpp>
#include <api/primitives/softmax.hpp>

using namespace cldnn;

// Building vgg16 network with loading weights & biases from file
cldnn::topology build_vgg16(const std::string& weights_dir, weights_optimizer& wo, cldnn::layout& input_layout, int32_t batch_size, bool use_bfyx)
{
    // [224x224x3xB] convolution->relu->pooling->lrn [1000xB]
    input_layout.size = { format::byxf,{ batch_size, 224, 224, 3 } };
    auto input = cldnn::input_layout("input", input_layout);

    use_bfyx = true;

    auto mem_format = use_bfyx ? format::bfyx : format::yxfb;

    // create conversion to yxfb format and subtract mean values
    tensor reorder_size = input_layout.size.transform(mem_format, 1);
    auto reorder_mean = wo.create_weights_from_file(join_path(weights_dir, "imagenet_mean.nnd"), file::mean);
    auto reordered_input = reorder(
        "reorder",
        input,
        { input_layout.data_type, reorder_size },
        reorder_mean);

    auto conv1_1 = convolution("conv1_1",
        reordered_input,
        { wo.create_weights_from_file(join_path(weights_dir, "conv1_1_weights.nnd"), file::convolution) },
        { wo.create_weights_from_file(join_path(weights_dir, "conv1_1_bias.nnd"), file::bias) },
        { format::yx, { -1, -1 } },
        { format::yx, { 1, 1 } },
        true); // negative slope for RELU

    auto conv1_2 = convolution("conv1_2",
            conv1_1,
        { wo.create_weights_from_file(join_path(weights_dir, "conv1_2_weights.nnd"), file::convolution) },
        { wo.create_weights_from_file(join_path(weights_dir, "conv1_2_bias.nnd"),  file::bias) },
        { format::yx, { -1, -1 } },
        { format::yx, { 1, 1 } },
        true); // negative slope for RELU


    auto pool1 = pooling("pool1",
        conv1_2,
        pooling_mode::max,
        { format::yx, { 2,2 } }, // strd
        { format::yx, { 2,2 } } // kernel
    );

    auto conv2_1 = convolution("conv2_1",
        pool1,
        { wo.create_weights_from_file(join_path(weights_dir, "conv2_1_weights.nnd"), file::convolution) },
        { wo.create_weights_from_file(join_path(weights_dir, "conv2_1_bias.nnd"),  file::bias) },
        { format::yx, { -1, -1 } },
        { format::yx, { 1, 1 } },
        true); // negative slope for RELU

    auto conv2_2 = convolution("conv2_2",
        conv2_1,
        { wo.create_weights_from_file(join_path(weights_dir, "conv2_2_weights.nnd"), file::convolution) },
        { wo.create_weights_from_file(join_path(weights_dir, "conv2_2_bias.nnd"),  file::bias) },
        { format::yx, { -1, -1 } },
        { format::yx, { 1, 1 } },
        true); // negative slope for RELU

    auto pool2 = pooling("pool2",
        conv2_2,
        pooling_mode::max,
        { format::yx, { 2,2 } }, // strd
        { format::yx, { 2,2 } } // kernel
    );

    auto conv3_1 = convolution("conv3_1",
        pool2,
        { wo.create_weights_from_file(join_path(weights_dir, "conv3_1_weights.nnd"), file::convolution) },
        { wo.create_weights_from_file(join_path(weights_dir, "conv3_1_bias.nnd"),  file::bias) },
        { format::yx, { -1, -1 } },
        { format::yx, { 1, 1 } },
        true); // negative slope for RELU

    auto conv3_2 = convolution("conv3_2",
        conv3_1,
        { wo.create_weights_from_file(join_path(weights_dir, "conv3_2_weights.nnd"), file::convolution) },
        { wo.create_weights_from_file(join_path(weights_dir, "conv3_2_bias.nnd"),  file::bias) },
        { format::yx, { -1, -1 } },
        { format::yx, { 1, 1 } },
        true); // negative slope for RELU

    auto conv3_3 = convolution("conv3_3",
        conv3_2,
        { wo.create_weights_from_file(join_path(weights_dir, "conv3_3_weights.nnd"), file::convolution) },
        { wo.create_weights_from_file(join_path(weights_dir, "conv3_3_bias.nnd"),  file::bias) },
        { format::yx, { -1, -1 } },
        { format::yx, { 1, 1 } },
        true); // negative slope for RELU

    auto pool3 = pooling("pool3",
        conv3_3,
        pooling_mode::max,
        { format::yx, { 2,2 } }, // strd
        { format::yx, { 2,2 } } // kernel
    );

    auto conv4_1 = convolution("conv4_1",
        pool3,
        { wo.create_weights_from_file(join_path(weights_dir, "conv4_1_weights.nnd"), file::convolution) },
        { wo.create_weights_from_file(join_path(weights_dir, "conv4_1_bias.nnd"),  file::bias) },
        { format::yx, { -1, -1 } },
        { format::yx, { 1, 1 } },
        true); // negative slope for RELU

    auto conv4_2 = convolution("conv4_2",
        conv4_1,
        { wo.create_weights_from_file(join_path(weights_dir, "conv4_2_weights.nnd"), file::convolution) },
        { wo.create_weights_from_file(join_path(weights_dir, "conv4_2_bias.nnd"),  file::bias) },
        { format::yx, { -1, -1 } },
        { format::yx, { 1, 1 } },
        true); // negative slope for RELU

    auto conv4_3 = convolution("conv4_3",
        conv4_2,
        { wo.create_weights_from_file(join_path(weights_dir, "conv4_3_weights.nnd"), file::convolution) },
        { wo.create_weights_from_file(join_path(weights_dir, "conv4_3_bias.nnd"),  file::bias) },
        { format::yx, { -1, -1 } },
        { format::yx, { 1, 1 } },
        true); // negative slope for RELU

    auto pool4 = pooling("pool4",
        conv4_3,
        pooling_mode::max,
        { format::yx, { 2,2 } }, // strd
        { format::yx, { 2,2 } } // kernel
    );

    auto conv5_1 = convolution("conv5_1",
        pool4,
        { wo.create_weights_from_file(join_path(weights_dir, "conv5_1_weights.nnd"), file::convolution) },
        { wo.create_weights_from_file(join_path(weights_dir, "conv5_1_bias.nnd"),  file::bias) },
        { format::yx, { -1, -1 } },
        { format::yx, { 1,1 } },
        true); // negative slope for RELU

    auto conv5_2 = convolution("conv5_2",
        conv5_1,
        { wo.create_weights_from_file(join_path(weights_dir, "conv5_2_weights.nnd"), file::convolution) },
        { wo.create_weights_from_file(join_path(weights_dir, "conv5_2_bias.nnd"),  file::bias) },
        { format::yx, { -1, -1 } },
        { format::yx, { 1,1 } },
        true); // negative slope for RELU

    auto conv5_3 = convolution("conv5_3",
        conv5_2,
        { wo.create_weights_from_file(join_path(weights_dir, "conv5_3_weights.nnd"), file::convolution) },
        { wo.create_weights_from_file(join_path(weights_dir, "conv5_3_bias.nnd"),  file::bias) },
        { format::yx, { -1, -1 } },
        { format::yx, { 1,1 } },
        true); // negative slope for RELU

    auto pool5 = pooling("pool5",
        conv5_3,
        pooling_mode::max,
        { format::yx, { 2,2 } }, // strd
        { format::yx, { 2,2 } } // kernel
    );

    auto fc6 = fully_connected("fc6",
        pool5,
        wo.create_weights_from_file(join_path(weights_dir, "fc6_weights.nnd"), file::fully_connected),
        wo.create_weights_from_file(join_path(weights_dir, "fc6_bias.nnd"),  file::bias),
        true,
        0
    );

    auto fc7 = fully_connected("fc7",
        fc6,
        wo.create_weights_from_file(join_path(weights_dir, "fc7_weights.nnd"), file::fully_connected),
        wo.create_weights_from_file(join_path(weights_dir, "fc7_bias.nnd"),  file::bias),
        true,
        0
    );

    auto fc8 = fully_connected("fc8",
        fc7,
        wo.create_weights_from_file(join_path(weights_dir, "fc8_weights.nnd"), file::fully_connected),
        wo.create_weights_from_file(join_path(weights_dir, "fc8_bias.nnd"),  file::bias),
        true,
        0
    );

    auto softmax = cldnn::softmax(
        "output",
        fc8);

    return topology {
        input,
        reordered_input,
        conv1_1,
        conv1_2,
        pool1,
        conv2_1,
        conv2_2,
        pool2,
        conv3_1,
        conv3_2,
        conv3_3,
        pool3,
        conv4_1,
        conv4_2,
        conv4_3,
        pool4,
        conv5_1,
        conv5_2,
        conv5_3,
        pool5,
        fc6,
        fc7,
        fc8,
        softmax
    };
}
