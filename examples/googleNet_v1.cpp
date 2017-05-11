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
#include <api/CPP/lrn.hpp>
#include <api/CPP/fully_connected.hpp>
#include <api/CPP/depth_concatenate.hpp>
#include <api/CPP/softmax.hpp>

using namespace cldnn;

// Building GoogLeNet v1 network with loading weights & biases from file
cldnn::topology build_googlenetv1(const std::string& weights_dir, const cldnn::engine& engine, cldnn::layout& input_layout, int32_t batch_size)
{
    // [224x224x3xB] convolution->relu->pooling->lrn [1000xB]
    input_layout.size = { batch_size, 3, 224, 224 };
    auto input = cldnn::input_layout("input", input_layout);

    // subtract mean values
    auto reordered_input = reorder(
        "reorder",
        input,
        { input_layout.data_type, input_layout.format, input_layout.size },
        std::vector<float>{ 104.0f, 117.0f, 123.0f });

    auto conv1_7x7_s2_w = file::create({ engine, join_path(weights_dir, "conv1_7x7_s2_weights.nnd")});
    auto conv1_7x7_s2_b = file::create({ engine, join_path(weights_dir, "conv1_7x7_s2_bias.nnd")});
    auto conv1_7x7_s2 = convolution("conv1_7x7_s2",
        reordered_input,
        { conv1_7x7_s2_w },
        { conv1_7x7_s2_b },
        { 1, 1, 2, 2 },
        { 0, 0, -3, -3 },
		{ 1, 1, 1, 1 },
        true);

    auto pool1_3x3_s2 = pooling("pool1_3x3_s2",
        conv1_7x7_s2,
        pooling_mode::max,
        { 1,1,3,3 },  // kernel
        { 1,1,2,2 }); // strd

    auto pool1_norm1 = lrn("pool1_norm1",
        pool1_3x3_s2,
        5,
        1.0f,
        0.0001f,
        0.75f,
        cldnn_lrn_norm_region_across_channel);

    auto conv2_3x3_reduce_w = file::create({ engine, join_path(weights_dir, "conv2_3x3_reduce_weights.nnd")});
    auto conv2_3x3_reduce_b = file::create({ engine, join_path(weights_dir, "conv2_3x3_reduce_bias.nnd")});
    auto conv2_3x3_reduce = convolution("conv2_3x3_reduce",
        pool1_norm1,
        { conv2_3x3_reduce_w },
        { conv2_3x3_reduce_b },
        { 1, 1, 1, 1 },
        { 0, 0, 0, 0 },
		{ 1, 1, 1, 1 },
        true);

    auto conv2_3x3_w = file::create({ engine, join_path(weights_dir, "conv2_3x3_weights.nnd")});
    auto conv2_3x3_b = file::create({ engine, join_path(weights_dir, "conv2_3x3_bias.nnd")});
    auto conv2_3x3 = convolution("conv2_3x3",
        conv2_3x3_reduce,
        { conv2_3x3_w },
        { conv2_3x3_b },
        { 1, 1, 1, 1 },
        { 0, 0, -1, -1 },
		{ 1, 1, 1, 1 },
        true);

    auto conv2_norm2 = lrn("conv2_norm2",
        conv2_3x3,
        5,
        1.0f,
        0.0001f,
        0.75f,
        cldnn_lrn_norm_region_across_channel);

    auto pool2_3x3_s2 = pooling("pool2_3x3_s2",
        conv2_norm2,
        pooling_mode::max,
        { 1,1,3,3 }, // kernel
        { 1,1,2,2 }); // strd

    // ----------- END OF PIPE -------------
    // Inception 1
    // 1st branch
    auto inception_3a_1x1_w = file::create({ engine, join_path(weights_dir, "inception_3a_1x1_weights.nnd")});
    auto inception_3a_1x1_b = file::create({ engine, join_path(weights_dir, "inception_3a_1x1_bias.nnd")});
    auto inception_3a_1x1 = convolution("inception_3a_1x1",
        pool2_3x3_s2,
        { inception_3a_1x1_w },
        { inception_3a_1x1_b },
        { 1, 1, 1, 1 },
        { 0, 0, 0, 0 },
		{ 1, 1, 1, 1 },
        true);

    // 2nd branch

    auto inception_3a_3x3_reduce_w = file::create({ engine, join_path(weights_dir, "inception_3a_3x3_reduce_weights.nnd")});
    auto inception_3a_3x3_reduce_b = file::create({ engine, join_path(weights_dir, "inception_3a_3x3_reduce_bias.nnd") });
    auto inception_3a_3x3_reduce = convolution("inception_3a_3x3_reduce",
        pool2_3x3_s2,
        { inception_3a_3x3_reduce_w },
        { inception_3a_3x3_reduce_b },
        { 1, 1, 1, 1 },
        { 0, 0, 0, 0 },
		{ 1, 1, 1, 1 },
        true);


    auto inception_3a_3x3_w = file::create({ engine, join_path(weights_dir, "inception_3a_3x3_weights.nnd")});
    auto inception_3a_3x3_b = file::create({ engine, join_path(weights_dir, "inception_3a_3x3_bias.nnd") });
    auto inception_3a_3x3 = convolution("inception_3a_3x3",
        inception_3a_3x3_reduce,
        { inception_3a_3x3_w },
        { inception_3a_3x3_b },
        { 1, 1, 1, 1 },
        { 0, 0, -1, -1 },
		{ 1, 1, 1, 1 },
        true);

    // 3rd branch

    auto inception_3a_5x5_reduce_w = file::create({ engine, join_path(weights_dir, "inception_3a_5x5_reduce_weights.nnd")});
    auto inception_3a_5x5_reduce_b = file::create({ engine, join_path(weights_dir, "inception_3a_5x5_reduce_bias.nnd") });
    auto inception_3a_5x5_reduce = convolution("inception_3a_5x5_reduce",
        pool2_3x3_s2,
        { inception_3a_5x5_reduce_w },
        { inception_3a_5x5_reduce_b },
        { 1, 1, 1, 1 },
        { 0, 0, 0, 0 },
		{ 1, 1, 1, 1 },
        true);


    auto inception_3a_5x5_w = file::create({ engine, join_path(weights_dir, "inception_3a_5x5_weights.nnd")});
    auto inception_3a_5x5_b = file::create({ engine, join_path(weights_dir, "inception_3a_5x5_bias.nnd") });
    auto inception_3a_5x5 = convolution("inception_3a_5x5",
        inception_3a_5x5_reduce,
        { inception_3a_5x5_w },
        { inception_3a_5x5_b },
        { 1, 1, 1, 1 },
        { 0, 0, -2, -2 },
		{ 1, 1, 1, 1 },
        true);

    // 4th branch

    auto inception_3a_pool = pooling("inception_3a_pool",
        pool2_3x3_s2,
        pooling_mode::max,
        { 1, 1, 3, 3 }, // kernel
        { 1, 1, 1, 1 }, // strd
        { 0, 0, -1, -1 } //input offset
    );

    auto inception_3a_pool_proj_w = file::create({ engine, join_path(weights_dir, "inception_3a_pool_proj_weights.nnd")});
    auto inception_3a_pool_proj_b = file::create({ engine, join_path(weights_dir, "inception_3a_pool_proj_bias.nnd") });
    auto inception_3a_pool_proj = convolution("inception_3a_pool_proj",
        inception_3a_pool,
        { inception_3a_pool_proj_w },
        { inception_3a_pool_proj_b },
        { 1, 1, 1, 1 },
        { 0, 0, 0, 0 },

		{ 1, 1, 1, 1 },
        true);

    auto inception_3a_output = depth_concatenate("inception_3a_output",
        {
            inception_3a_1x1,
            inception_3a_3x3,
            inception_3a_5x5,
            inception_3a_pool_proj
        });
    // End of 1st nception
    
    // --------------------- 2nd inception ---------------------------------

    // 1st branch
    auto inception_3b_1x1_w = file::create({ engine, join_path(weights_dir, "inception_3b_1x1_weights.nnd")});
    auto inception_3b_1x1_b = file::create({ engine, join_path(weights_dir, "inception_3b_1x1_bias.nnd") });
    auto inception_3b_1x1 = convolution("inception_3b_1x1",
        inception_3a_output,
        { inception_3b_1x1_w },
        { inception_3b_1x1_b },
        { 1, 1, 1, 1 },
        { 0, 0, 0, 0 },
		{ 1, 1, 1, 1 },
        true);

    // 2nd branch

    auto inception_3b_3x3_reduce_w = file::create({ engine, join_path(weights_dir, "inception_3b_3x3_reduce_weights.nnd")});
    auto inception_3b_3x3_reduce_b = file::create({ engine, join_path(weights_dir, "inception_3b_3x3_reduce_bias.nnd") });
    auto inception_3b_3x3_reduce = convolution("inception_3b_3x3_reduce",
        inception_3a_output,
        { inception_3b_3x3_reduce_w },
        { inception_3b_3x3_reduce_b },
        { 1, 1, 1, 1 },
        { 0, 0, 0, 0 },
		{ 1, 1, 1, 1 },
        true);

    auto inception_3b_3x3_w = file::create({ engine, join_path(weights_dir, "inception_3b_3x3_weights.nnd")});
    auto inception_3b_3x3_b = file::create({ engine, join_path(weights_dir, "inception_3b_3x3_bias.nnd") });
    auto inception_3b_3x3 = convolution("inception_3b_3x3",
        inception_3b_3x3_reduce,
        { inception_3b_3x3_w },
        { inception_3b_3x3_b },
        { 1, 1, 1, 1 },
        { 0, 0, -1, -1 },
		{ 1, 1, 1, 1 },
        true);

    // 3rd branch

    auto inception_3b_5x5_reduce_w = file::create({ engine, join_path(weights_dir, "inception_3b_5x5_reduce_weights.nnd")});
    auto inception_3b_5x5_reduce_b = file::create({ engine, join_path(weights_dir, "inception_3b_5x5_reduce_bias.nnd") });
    auto inception_3b_5x5_reduce = convolution("inception_3b_5x5_reduce",
        inception_3a_output,
        { inception_3b_5x5_reduce_w },
        { inception_3b_5x5_reduce_b },
        { 1, 1, 1, 1 },
        { 0, 0, 0, 0 },
		{ 1, 1, 1, 1 },
        true);

    auto inception_3b_5x5_w = file::create({ engine, join_path(weights_dir, "inception_3b_5x5_weights.nnd")});
    auto inception_3b_5x5_b = file::create({ engine, join_path(weights_dir, "inception_3b_5x5_bias.nnd") });
    auto inception_3b_5x5 = convolution("inception_3b_5x5",
        inception_3b_5x5_reduce,
        { inception_3b_5x5_w },
        { inception_3b_5x5_b },
        { 1, 1, 1, 1 },
        { 0, 0, -2, -2 },
		{ 1, 1, 1, 1 },
        true);

    // 4th branch

    auto inception_3b_pool = pooling("inception_3b_pool",
        inception_3a_output,
        pooling_mode::max,
        { 1, 1, 3, 3 }, // kernel
        { 1, 1, 1, 1 }, // strd
        { 0, 0, -1, -1 } //input offset
    );

    auto inception_3b_pool_proj_w = file::create({ engine, join_path(weights_dir, "inception_3b_pool_proj_weights.nnd")});
    auto inception_3b_pool_proj_b = file::create({ engine, join_path(weights_dir, "inception_3b_pool_proj_bias.nnd") });
    auto inception_3b_pool_proj = convolution("inception_3b_pool_proj",
        inception_3b_pool,
        { inception_3b_pool_proj_w },
        { inception_3b_pool_proj_b },
        { 1, 1, 1, 1 },
        { 0, 0, 0, 0 },
		{ 1, 1, 1, 1 },
        true);

    auto inception_3b_output = depth_concatenate("inception_3b_output",
        {
            inception_3b_1x1,
            inception_3b_3x3,
            inception_3b_5x5,
            inception_3b_pool_proj
        });

    // End of 2nd inception

    auto pool3_3x3_s2 = pooling("pool3_3x3_s2",
        inception_3b_output,
        pooling_mode::max,
        { 1, 1, 3, 3 }, // kernel
        { 1, 1, 2, 2 }  // strd
    );

    // --------------------- 3rd inception ---------------------------------
    // 1st branch
    auto inception_4a_1x1_w = file::create({ engine, join_path(weights_dir, "inception_4a_1x1_weights.nnd")});
    auto inception_4a_1x1_b = file::create({ engine, join_path(weights_dir, "inception_4a_1x1_bias.nnd") });
    auto inception_4a_1x1 = convolution("inception_4a_1x1",
        pool3_3x3_s2,
        { inception_4a_1x1_w },
        { inception_4a_1x1_b },
        { 1, 1, 1, 1 },
        { 0, 0, 0, 0 },
		{ 1, 1, 1, 1 },
        true);

    // 2nd branch

    auto inception_4a_3x3_reduce_w = file::create({ engine, join_path(weights_dir, "inception_4a_3x3_reduce_weights.nnd")});
    auto inception_4a_3x3_reduce_b = file::create({ engine, join_path(weights_dir, "inception_4a_3x3_reduce_bias.nnd") });
    auto inception_4a_3x3_reduce = convolution("inception_4a_3x3_reduce",
        pool3_3x3_s2,
        { inception_4a_3x3_reduce_w },
        { inception_4a_3x3_reduce_b },
        { 1, 1, 1, 1 },
        { 0, 0, 0, 0 },
		{ 1, 1, 1, 1 },
        true);

    auto inception_4a_3x3_w = file::create({ engine, join_path(weights_dir, "inception_4a_3x3_weights.nnd")});
    auto inception_4a_3x3_b = file::create({ engine, join_path(weights_dir, "inception_4a_3x3_bias.nnd") });
    auto inception_4a_3x3 = convolution("inception_4a_3x3",
        inception_4a_3x3_reduce,
        { inception_4a_3x3_w },
        { inception_4a_3x3_b },
        { 1, 1, 1, 1 },
        { 0, 0, -1, -1 },
		{ 1, 1, 1, 1 },
        true);

    // 3rd branch

    auto inception_4a_5x5_reduce_w = file::create({ engine, join_path(weights_dir, "inception_4a_5x5_reduce_weights.nnd")});
    auto inception_4a_5x5_reduce_b = file::create({ engine, join_path(weights_dir, "inception_4a_5x5_reduce_bias.nnd") });
    auto inception_4a_5x5_reduce = convolution("inception_4a_5x5_reduce",
        pool3_3x3_s2,
        { inception_4a_5x5_reduce_w },
        { inception_4a_5x5_reduce_b },
        { 1, 1, 1, 1 },
        { 0, 0, 0, 0 },
		{ 1, 1, 1, 1 },
        true);

    auto inception_4a_5x5_w = file::create({ engine, join_path(weights_dir, "inception_4a_5x5_weights.nnd")});
    auto inception_4a_5x5_b = file::create({ engine, join_path(weights_dir, "inception_4a_5x5_bias.nnd") });
    auto inception_4a_5x5 = convolution("inception_4a_5x5",
        inception_4a_5x5_reduce,
        { inception_4a_5x5_w },
        { inception_4a_5x5_b },
        { 1, 1, 1, 1 },
        { 0, 0, -2, -2 },
		{ 1, 1, 1, 1 },
        true);

    // 4th branch

    auto inception_4a_pool = pooling("inception_4a_pool",
        pool3_3x3_s2,
        pooling_mode::max,
        { 1, 1, 3, 3 }, // kernel
        { 1, 1, 1, 1 }, // strd
        { 0,0,-1,-1 } //input offset
    );

    auto inception_4a_pool_proj_w = file::create({ engine, join_path(weights_dir, "inception_4a_pool_proj_weights.nnd")});
    auto inception_4a_pool_proj_b = file::create({ engine, join_path(weights_dir, "inception_4a_pool_proj_bias.nnd") });
    auto inception_4a_pool_proj = convolution("inception_4a_pool_proj",
        inception_4a_pool,
        { inception_4a_pool_proj_w },
        { inception_4a_pool_proj_b },
        { 1, 1, 1, 1 },
        { 0, 0, 0, 0 },
		{ 1, 1, 1, 1 },
        true);

    auto inception_4a_output = depth_concatenate("inception_4a_output",
    {
            inception_4a_1x1,
            inception_4a_3x3,
            inception_4a_5x5,
            inception_4a_pool_proj
    });
    // End of 3rd inception

    // --------------------- 4th inception ---------------------------------
    // 1st branch
    auto inception_4b_1x1_w = file::create({ engine, join_path(weights_dir, "inception_4b_1x1_weights.nnd")});
    auto inception_4b_1x1_b = file::create({ engine, join_path(weights_dir, "inception_4b_1x1_bias.nnd") });
    auto inception_4b_1x1 = convolution("inception_4b_1x1",
        inception_4a_output,
        { inception_4b_1x1_w },
        { inception_4b_1x1_b },
        { 1, 1, 1, 1 },
        { 0, 0, 0, 0 },
		{ 1, 1, 1, 1 },
        true);

    // 2nd branch

    auto inception_4b_3x3_reduce_w = file::create({ engine, join_path(weights_dir, "inception_4b_3x3_reduce_weights.nnd")});
    auto inception_4b_3x3_reduce_b = file::create({ engine, join_path(weights_dir, "inception_4b_3x3_reduce_bias.nnd") });
    auto inception_4b_3x3_reduce = convolution("inception_4b_3x3_reduce",
        inception_4a_output,
        { inception_4b_3x3_reduce_w },
        { inception_4b_3x3_reduce_b },
        { 1, 1, 1, 1 },
        { 0, 0, 0, 0 },
		{ 1, 1, 1, 1 },
        true);

    auto inception_4b_3x3_w = file::create({ engine, join_path(weights_dir, "inception_4b_3x3_weights.nnd")});
    auto inception_4b_3x3_b = file::create({ engine, join_path(weights_dir, "inception_4b_3x3_bias.nnd") });
    auto inception_4b_3x3 = convolution("inception_4b_3x3",
        inception_4b_3x3_reduce,
        { inception_4b_3x3_w },
        { inception_4b_3x3_b },
        { 1, 1, 1, 1 },
        { 0, 0, -1, -1 },
		{ 1, 1, 1, 1 },
        true);

    // 3rd branch

    auto inception_4b_5x5_reduce_w = file::create({ engine, join_path(weights_dir, "inception_4b_5x5_reduce_weights.nnd")});
    auto inception_4b_5x5_reduce_b = file::create({ engine, join_path(weights_dir, "inception_4b_5x5_reduce_bias.nnd") });
    auto inception_4b_5x5_reduce = convolution("inception_4b_5x5_reduce",
        inception_4a_output,
        { inception_4b_5x5_reduce_w },
        { inception_4b_5x5_reduce_b },
        { 1, 1, 1, 1 },
        { 0, 0, 0, 0 },
		{ 1, 1, 1, 1 },
        true);

    auto inception_4b_5x5_w = file::create({ engine, join_path(weights_dir, "inception_4b_5x5_weights.nnd")});
    auto inception_4b_5x5_b = file::create({ engine, join_path(weights_dir, "inception_4b_5x5_bias.nnd") });
    auto inception_4b_5x5 = convolution("inception_4b_5x5",
        inception_4b_5x5_reduce,
        { inception_4b_5x5_w },
        { inception_4b_5x5_b },
        { 1, 1, 1, 1 },
        { 0, 0, -2, -2 },
		{ 1, 1, 1, 1 },
        true);

    // 4th branch

    auto inception_4b_pool = pooling("inception_4b_pool",
        inception_4a_output,
        pooling_mode::max,
        { 1, 1, 3, 3 }, // kernel
        { 1, 1, 1, 1 }, // strd
        { 0,0,-1,-1 } //input offset
    );

    auto inception_4b_pool_proj_w = file::create({ engine, join_path(weights_dir, "inception_4b_pool_proj_weights.nnd")});
    auto inception_4b_pool_proj_b = file::create({ engine, join_path(weights_dir, "inception_4b_pool_proj_bias.nnd") });
    auto inception_4b_pool_proj = convolution("inception_4b_pool_proj",
        inception_4b_pool,
        { inception_4b_pool_proj_w },
        { inception_4b_pool_proj_b },
        { 1, 1, 1, 1 },
        { 0, 0, 0, 0 },
		{ 1, 1, 1, 1 },
        true);

    auto inception_4b_output = depth_concatenate("inception_4b_output",
    {
            inception_4b_1x1,
            inception_4b_3x3,
            inception_4b_5x5,
            inception_4b_pool_proj
    });

    // End of 4th inception
    
    // --------------------- 5th inception ---------------------------------
    // 1st branch
    auto inception_4c_1x1_w = file::create({ engine, join_path(weights_dir, "inception_4c_1x1_weights.nnd")});
    auto inception_4c_1x1_b = file::create({ engine, join_path(weights_dir, "inception_4c_1x1_bias.nnd") });
    auto inception_4c_1x1 = convolution("inception_4c_1x1",
        inception_4b_output,
        { inception_4c_1x1_w },
        { inception_4c_1x1_b },
        { 1, 1, 1, 1 },
        { 0, 0, 0, 0 },
		{ 1, 1, 1, 1 },
        true);

    // 2nd branch

    auto inception_4c_3x3_reduce_w = file::create({ engine, join_path(weights_dir, "inception_4c_3x3_reduce_weights.nnd")});
    auto inception_4c_3x3_reduce_b = file::create({ engine, join_path(weights_dir, "inception_4c_3x3_reduce_bias.nnd") });
    auto inception_4c_3x3_reduce = convolution("inception_4c_3x3_reduce",
        inception_4b_output,
        { inception_4c_3x3_reduce_w },
        { inception_4c_3x3_reduce_b },
        { 1, 1, 1, 1 },
        { 0, 0, 0, 0 },
		{ 1, 1, 1, 1 },
        true);

    auto inception_4c_3x3_w = file::create({ engine, join_path(weights_dir, "inception_4c_3x3_weights.nnd")});
    auto inception_4c_3x3_b = file::create({ engine, join_path(weights_dir, "inception_4c_3x3_bias.nnd") });
    auto inception_4c_3x3 = convolution("inception_4c_3x3",
        inception_4c_3x3_reduce,
        { inception_4c_3x3_w },
        { inception_4c_3x3_b },
        { 1, 1, 1, 1 },
        { 0, 0, -1, -1 },
		{ 1, 1, 1, 1 },
        true);

    // 3rd branch
    auto inception_4c_5x5_reduce_w = file::create({ engine, join_path(weights_dir, "inception_4c_5x5_reduce_weights.nnd")});
    auto inception_4c_5x5_reduce_b = file::create({ engine, join_path(weights_dir, "inception_4c_5x5_reduce_bias.nnd") });
    auto inception_4c_5x5_reduce = convolution("inception_4c_5x5_reduce",
        inception_4b_output,
        { inception_4c_5x5_reduce_w },
        { inception_4c_5x5_reduce_b },
        { 1, 1, 1, 1 },
        { 0, 0, 0, 0 },
		{ 1, 1, 1, 1 },
        true);

    auto inception_4c_5x5_w = file::create({ engine, join_path(weights_dir, "inception_4c_5x5_weights.nnd")});
    auto inception_4c_5x5_b = file::create({ engine, join_path(weights_dir, "inception_4c_5x5_bias.nnd") });
    auto inception_4c_5x5 = convolution("inception_4c_5x5",
        inception_4c_5x5_reduce,
        { inception_4c_5x5_w },
        { inception_4c_5x5_b },
        { 1, 1, 1, 1 },
        { 0, 0, -2, -2 },
		{ 1, 1, 1, 1 },
        true);

    // 4th branch

    auto inception_4c_pool = pooling("inception_4c_pool",
        inception_4b_output,
        pooling_mode::max,
        { 1, 1, 3, 3 }, // kernel
        { 1, 1, 1, 1 }, // strd
        { 0,0,-1,-1 } //input offset
    );

    auto inception_4c_pool_proj_w = file::create({ engine, join_path(weights_dir, "inception_4c_pool_proj_weights.nnd")});
    auto inception_4c_pool_proj_b = file::create({ engine, join_path(weights_dir, "inception_4c_pool_proj_bias.nnd") });
    auto inception_4c_pool_proj = convolution("inception_4c_pool_proj",
        inception_4c_pool,
        { inception_4c_pool_proj_w },
        { inception_4c_pool_proj_b },
        { 1, 1, 1, 1 },
        { 0, 0, 0, 0 },
		{ 1, 1, 1, 1 },
        true);

    auto inception_4c_output = depth_concatenate("inception_4c_output",
    {
            inception_4c_1x1,
            inception_4c_3x3,
            inception_4c_5x5,
            inception_4c_pool_proj
    });

    // --------------------- 6th inception ---------------------------------
    // 1st branch
    auto inception_4d_1x1_w = file::create({ engine, join_path(weights_dir, "inception_4d_1x1_weights.nnd")});
    auto inception_4d_1x1_b = file::create({ engine, join_path(weights_dir, "inception_4d_1x1_bias.nnd") });
    auto inception_4d_1x1 = convolution("inception_4d_1x1",
        inception_4c_output,
        { inception_4d_1x1_w },
        { inception_4d_1x1_b },
        { 1, 1, 1, 1 },
        { 0, 0, 0, 0 },
		{ 1, 1, 1, 1 },
        true);

    // 2nd branch

    auto inception_4d_3x3_reduce_w = file::create({ engine, join_path(weights_dir, "inception_4d_3x3_reduce_weights.nnd")});
    auto inception_4d_3x3_reduce_b = file::create({ engine, join_path(weights_dir, "inception_4d_3x3_reduce_bias.nnd") });
    auto inception_4d_3x3_reduce = convolution("inception_4d_3x3_reduce",
        inception_4c_output,
        { inception_4d_3x3_reduce_w },
        { inception_4d_3x3_reduce_b },
        { 1, 1, 1, 1 },
        { 0, 0, 0, 0 },
		{ 1, 1, 1, 1 },
        true);

    auto inception_4d_3x3_w = file::create({ engine, join_path(weights_dir, "inception_4d_3x3_weights.nnd")});
    auto inception_4d_3x3_b = file::create({ engine, join_path(weights_dir, "inception_4d_3x3_bias.nnd") });
    auto inception_4d_3x3 = convolution("inception_4d_3x3",
        inception_4d_3x3_reduce,
        { inception_4d_3x3_w },
        { inception_4d_3x3_b },
        { 1, 1, 1, 1 },
        { 0, 0, -1, -1 },
		{ 1, 1, 1, 1 },
        true);

    // 3rd branch
    auto inception_4d_5x5_reduce_w = file::create({ engine, join_path(weights_dir, "inception_4d_5x5_reduce_weights.nnd")});
    auto inception_4d_5x5_reduce_b = file::create({ engine, join_path(weights_dir, "inception_4d_5x5_reduce_bias.nnd") });
    auto inception_4d_5x5_reduce = convolution("inception_4d_5x5_reduce",
        inception_4c_output,
        { inception_4d_5x5_reduce_w },
        { inception_4d_5x5_reduce_b },
        { 1, 1, 1, 1 },
        { 0, 0, 0, 0 },
		{ 1, 1, 1, 1 },
        true);

    auto inception_4d_5x5_w = file::create({ engine, join_path(weights_dir, "inception_4d_5x5_weights.nnd")});
    auto inception_4d_5x5_b = file::create({ engine, join_path(weights_dir, "inception_4d_5x5_bias.nnd") });
    auto inception_4d_5x5 = convolution("inception_4d_5x5",
        inception_4d_5x5_reduce,
        { inception_4d_5x5_w },
        { inception_4d_5x5_b },
        { 1, 1, 1, 1 },
        { 0, 0, -2, -2 },
		{ 1, 1, 1, 1 },
        true);

    // 4th branch

    auto inception_4d_pool = pooling("inception_4d_pool",
        inception_4c_output,
        pooling_mode::max,
        { 1, 1, 3, 3 }, // kernel
        { 1, 1, 1, 1 }, // strd
        { 0,0,-1,-1 } //input offset
    );

    auto inception_4d_pool_proj_w = file::create({ engine, join_path(weights_dir, "inception_4d_pool_proj_weights.nnd")});
    auto inception_4d_pool_proj_b = file::create({ engine, join_path(weights_dir, "inception_4d_pool_proj_bias.nnd") });
    auto inception_4d_pool_proj = convolution("inception_4d_pool_proj",
        inception_4d_pool,
        { inception_4d_pool_proj_w },
        { inception_4d_pool_proj_b },
        { 1, 1, 1, 1 },
        { 0, 0, 0, 0 },
		{ 1, 1, 1, 1 },
        true);

    auto inception_4d_output = depth_concatenate("inception_4d_output",
    {
            inception_4d_1x1,
            inception_4d_3x3,
            inception_4d_5x5,
            inception_4d_pool_proj
    });

    // --------------------- 7th inception ---------------------------------
    // 1st branch
    auto inception_4e_1x1_w = file::create({ engine, join_path(weights_dir, "inception_4e_1x1_weights.nnd")});
    auto inception_4e_1x1_b = file::create({ engine, join_path(weights_dir, "inception_4e_1x1_bias.nnd") });
    auto inception_4e_1x1 = convolution("inception_4e_1x1",
        inception_4d_output,
        { inception_4e_1x1_w },
        { inception_4e_1x1_b },
        { 1, 1, 1, 1 },
        { 0, 0, 0, 0 },
		{ 1, 1, 1, 1 },
        true);

    // 2nd branch

    auto inception_4e_3x3_reduce_w = file::create({ engine, join_path(weights_dir, "inception_4e_3x3_reduce_weights.nnd")});
    auto inception_4e_3x3_reduce_b = file::create({ engine, join_path(weights_dir, "inception_4e_3x3_reduce_bias.nnd") });
    auto inception_4e_3x3_reduce = convolution("inception_4e_3x3_reduce",
        inception_4d_output,
        { inception_4e_3x3_reduce_w },
        { inception_4e_3x3_reduce_b },
        { 1, 1, 1, 1 },
        { 0, 0, 0, 0 },
		{ 1, 1, 1, 1 },
        true);

    auto inception_4e_3x3_w = file::create({ engine, join_path(weights_dir, "inception_4e_3x3_weights.nnd")});
    auto inception_4e_3x3_b = file::create({ engine, join_path(weights_dir, "inception_4e_3x3_bias.nnd") });
    auto inception_4e_3x3 = convolution("inception_4e_3x3",
        inception_4e_3x3_reduce,
        { inception_4e_3x3_w },
        { inception_4e_3x3_b },
        { 1, 1, 1, 1 },
        { 0, 0, -1, -1 },
		{ 1, 1, 1, 1 },
        true);

    // 3rd branch
    auto inception_4e_5x5_reduce_w = file::create({ engine, join_path(weights_dir, "inception_4e_5x5_reduce_weights.nnd")});
    auto inception_4e_5x5_reduce_b = file::create({ engine, join_path(weights_dir, "inception_4e_5x5_reduce_bias.nnd") });
    auto inception_4e_5x5_reduce = convolution("inception_4e_5x5_reduce",
        inception_4d_output,
        { inception_4e_5x5_reduce_w },
        { inception_4e_5x5_reduce_b },
        { 1, 1, 1, 1 },
        { 0, 0, 0, 0 },
		{ 1, 1, 1, 1 },
        true);

    auto inception_4e_5x5_w = file::create({ engine, join_path(weights_dir, "inception_4e_5x5_weights.nnd")});
    auto inception_4e_5x5_b = file::create({ engine, join_path(weights_dir, "inception_4e_5x5_bias.nnd") });
    auto inception_4e_5x5 = convolution("inception_4e_5x5",
        inception_4e_5x5_reduce,
        { inception_4e_5x5_w },
        { inception_4e_5x5_b },
        { 1, 1, 1, 1 },
        { 0, 0, -2, -2 },
		{ 1, 1, 1, 1 },
        true);

    // 4th branch

    auto inception_4e_pool = pooling("inception_4e_pool",
        inception_4d_output,
        pooling_mode::max,
        { 1, 1, 3, 3 }, // kernel
        { 1, 1, 1, 1 }, // strd
        { 0,0,-1,-1 } //input offset
    );

    auto inception_4e_pool_proj_w = file::create({ engine, join_path(weights_dir, "inception_4e_pool_proj_weights.nnd")});
    auto inception_4e_pool_proj_b = file::create({ engine, join_path(weights_dir, "inception_4e_pool_proj_bias.nnd") });
    auto inception_4e_pool_proj = convolution("inception_4e_pool_proj",
        inception_4e_pool,
        { inception_4e_pool_proj_w },
        { inception_4e_pool_proj_b },
        { 1, 1, 1, 1 },
        { 0, 0, 0, 0 },
		{ 1, 1, 1, 1 },
        true);

    auto inception_4e_output = depth_concatenate("inception_4e_output",
    {
            inception_4e_1x1,
            inception_4e_3x3,
            inception_4e_5x5,
            inception_4e_pool_proj
    });
   
    // ----------- End of 7th inception -------------------

    auto pool4_3x3_s2 = pooling("pool4_3x3_s2",
        inception_4e_output,
        pooling_mode::max,
        { 1, 1, 3, 3 }, // kernel
        { 1, 1, 2, 2 } // strd
    );

    // --------------------- 8th inception ---------------------------------
    // 1st branch
    auto inception_5a_1x1_w = file::create({ engine, join_path(weights_dir, "inception_5a_1x1_weights.nnd")});
    auto inception_5a_1x1_b = file::create({ engine, join_path(weights_dir, "inception_5a_1x1_bias.nnd") });
    auto inception_5a_1x1 = convolution("inception_5a_1x1",
        pool4_3x3_s2,
        { inception_5a_1x1_w },
        { inception_5a_1x1_b },
        { 1, 1, 1, 1 },
        { 0, 0, 0, 0 },
		{ 1, 1, 1, 1 },
        true);

    // 2nd branch

    auto inception_5a_3x3_reduce_w = file::create({ engine, join_path(weights_dir, "inception_5a_3x3_reduce_weights.nnd")});
    auto inception_5a_3x3_reduce_b = file::create({ engine, join_path(weights_dir, "inception_5a_3x3_reduce_bias.nnd") });
    auto inception_5a_3x3_reduce = convolution("inception_5a_3x3_reduce",
        pool4_3x3_s2,
        { inception_5a_3x3_reduce_w },
        { inception_5a_3x3_reduce_b },
        { 1, 1, 1, 1 },
        { 0, 0, 0, 0 },
		{ 1, 1, 1, 1 },
        true);

    auto inception_5a_3x3_w = file::create({ engine, join_path(weights_dir, "inception_5a_3x3_weights.nnd")});
    auto inception_5a_3x3_b = file::create({ engine, join_path(weights_dir, "inception_5a_3x3_bias.nnd") });
    auto inception_5a_3x3 = convolution("inception_5a_3x3",
        inception_5a_3x3_reduce,
        { inception_5a_3x3_w },
        { inception_5a_3x3_b },
        { 1, 1, 1, 1 },
        { 0, 0, -1, -1 },
		{ 1, 1, 1, 1 },
        true);

    // 3rd branch
    auto inception_5a_5x5_reduce_w = file::create({ engine, join_path(weights_dir, "inception_5a_5x5_reduce_weights.nnd")});
    auto inception_5a_5x5_reduce_b = file::create({ engine, join_path(weights_dir, "inception_5a_5x5_reduce_bias.nnd") });
    auto inception_5a_5x5_reduce = convolution("inception_5a_5x5_reduce",
        pool4_3x3_s2,
        { inception_5a_5x5_reduce_w },
        { inception_5a_5x5_reduce_b },
        { 1, 1, 1, 1 },
        { 0, 0, 0, 0 },
		{ 1, 1, 1, 1 },
        true);

    auto inception_5a_5x5_w = file::create({ engine, join_path(weights_dir, "inception_5a_5x5_weights.nnd")});
    auto inception_5a_5x5_b = file::create({ engine, join_path(weights_dir, "inception_5a_5x5_bias.nnd") });
    auto inception_5a_5x5 = convolution("inception_5a_5x5",
        inception_5a_5x5_reduce,
        { inception_5a_5x5_w },
        { inception_5a_5x5_b },
        { 1, 1, 1, 1 },
        { 0, 0, -2, -2 },
		{ 1, 1, 1, 1 },
        true);

    // 4th branch

    auto inception_5a_pool = pooling("inception_5a_pool",
        pool4_3x3_s2,
        pooling_mode::max,
        { 1, 1, 3, 3 }, // kernel
        { 1, 1, 1, 1 }, // strd
        { 0,0,-1,-1 } //input offset
    );

    auto inception_5a_pool_proj_w = file::create({ engine, join_path(weights_dir, "inception_5a_pool_proj_weights.nnd")});
    auto inception_5a_pool_proj_b = file::create({ engine, join_path(weights_dir, "inception_5a_pool_proj_bias.nnd") });
    auto inception_5a_pool_proj = convolution("inception_5a_pool_proj",
        inception_5a_pool,
        { inception_5a_pool_proj_w },
        { inception_5a_pool_proj_b },
        { 1, 1, 1, 1 },
        { 0, 0, 0, 0 },
		{ 1, 1, 1, 1 },
        true);

    auto inception_5a_output = depth_concatenate("inception_5a_output",
    {
            inception_5a_1x1,
            inception_5a_3x3,
            inception_5a_5x5,
            inception_5a_pool_proj
    });

    // --------------------- 8th inception ---------------------------------
    // 1st branch
    auto inception_5b_1x1_w = file::create({ engine, join_path(weights_dir, "inception_5b_1x1_weights.nnd")});
    auto inception_5b_1x1_b = file::create({ engine, join_path(weights_dir, "inception_5b_1x1_bias.nnd") });
    auto inception_5b_1x1 = convolution("inception_5b_1x1",
        inception_5a_output,
        { inception_5b_1x1_w },
        { inception_5b_1x1_b },
        { 1, 1, 1, 1 },
        { 0, 0, 0, 0 },
		{ 1, 1, 1, 1 },
        true);

    // 2nd branch

    auto inception_5b_3x3_reduce_w = file::create({ engine, join_path(weights_dir, "inception_5b_3x3_reduce_weights.nnd")});
    auto inception_5b_3x3_reduce_b = file::create({ engine, join_path(weights_dir, "inception_5b_3x3_reduce_bias.nnd") });
    auto inception_5b_3x3_reduce = convolution("inception_5b_3x3_reduce",
        inception_5a_output,
        { inception_5b_3x3_reduce_w },
        { inception_5b_3x3_reduce_b },
        { 1, 1, 1, 1 },
        { 0, 0, 0, 0 },
		{ 1, 1, 1, 1 },
        true);

    auto inception_5b_3x3_w = file::create({ engine, join_path(weights_dir, "inception_5b_3x3_weights.nnd")});
    auto inception_5b_3x3_b = file::create({ engine, join_path(weights_dir, "inception_5b_3x3_bias.nnd") });
    auto inception_5b_3x3 = convolution("inception_5b_3x3",
        inception_5b_3x3_reduce,
        { inception_5b_3x3_w },
        { inception_5b_3x3_b },
        { 1, 1, 1, 1 },
        { 0, 0, -1, -1 },
		{ 1, 1, 1, 1 },
        true);

    // 3rd branch
    auto inception_5b_5x5_reduce_w = file::create({ engine, join_path(weights_dir, "inception_5b_5x5_reduce_weights.nnd")});
    auto inception_5b_5x5_reduce_b = file::create({ engine, join_path(weights_dir, "inception_5b_5x5_reduce_bias.nnd") });
    auto inception_5b_5x5_reduce = convolution("inception_5b_5x5_reduce",
        inception_5a_output,
        { inception_5b_5x5_reduce_w },
        { inception_5b_5x5_reduce_b },
        { 1, 1, 1, 1 },
        { 0, 0, 0, 0 },
		{ 1, 1, 1, 1 },
        true);

    auto inception_5b_5x5_w = file::create({ engine, join_path(weights_dir, "inception_5b_5x5_weights.nnd")});
    auto inception_5b_5x5_b = file::create({ engine, join_path(weights_dir, "inception_5b_5x5_bias.nnd") });
    auto inception_5b_5x5 = convolution("inception_5b_5x5",
        inception_5b_5x5_reduce,
        { inception_5b_5x5_w },
        { inception_5b_5x5_b },
        { 1, 1, 1, 1 },
        { 0, 0, -2, -2 },
		{ 1, 1, 1, 1 },
        true);

    // 4th branch

    auto inception_5b_pool = pooling("inception_5b_pool",
        inception_5a_output,
        pooling_mode::max,
        { 1, 1, 3, 3 }, // kernel
        { 1, 1, 1, 1 }, // strd
        { 0,0,-1,-1 } //input offset
    );

    auto inception_5b_pool_proj_w = file::create({ engine, join_path(weights_dir, "inception_5b_pool_proj_weights.nnd")});
    auto inception_5b_pool_proj_b = file::create({ engine, join_path(weights_dir, "inception_5b_pool_proj_bias.nnd") });
    auto inception_5b_pool_proj = convolution("inception_5b_pool_proj",
        inception_5b_pool,
        { inception_5b_pool_proj_w },
        { inception_5b_pool_proj_b },
        { 1, 1, 1, 1 },
        { 0, 0, 0, 0 },
		{ 1, 1, 1, 1 },
        true);

    auto inception_5b_output = depth_concatenate("inception_5b_output",
    {
            inception_5b_1x1,
            inception_5b_3x3,
            inception_5b_5x5,
            inception_5b_pool_proj
    });

    // ------------------ ENDING PIPE --------------------------
    auto pool5_7x7_s1 = pooling("pool5_7x7_s1",
        inception_5b_output,
        pooling_mode::average,
        { 1, 1, 7, 7 }, // kernel
        { 1, 1, 1, 1 } // strd
    );


    auto loss3_classifier_w = file::create({ engine, join_path(weights_dir, "loss3_classifier_weights.nnd")});
    auto loss3_classifier_b = file::create({ engine, join_path(weights_dir, "loss3_classifier_bias.nnd")});
    auto loss3_classifier = fully_connected("loss3_classifier",
        pool5_7x7_s1,
        loss3_classifier_w,
        loss3_classifier_b,
        true,
        0
    );

    auto softmax = cldnn::softmax(
        "output",
        loss3_classifier);

    cldnn::topology topology{
        input,
        reordered_input,
        conv1_7x7_s2, conv1_7x7_s2_w, conv1_7x7_s2_b,
        pool1_3x3_s2,
        pool1_norm1,
        conv2_3x3_reduce, conv2_3x3_reduce_w, conv2_3x3_reduce_b,
        conv2_3x3, conv2_3x3_w, conv2_3x3_b,
        conv2_norm2,
        pool2_3x3_s2 };
    topology.add(
        inception_3a_1x1, inception_3a_1x1_w, inception_3a_1x1_b,
        inception_3a_3x3_reduce, inception_3a_3x3_reduce_w, inception_3a_3x3_reduce_b,
        inception_3a_3x3, inception_3a_3x3_w, inception_3a_3x3_b,
        inception_3a_5x5_reduce, inception_3a_5x5_reduce_w, inception_3a_5x5_reduce_b,
        inception_3a_5x5, inception_3a_5x5_w, inception_3a_5x5_b,
        inception_3a_pool,
        inception_3a_pool_proj, inception_3a_pool_proj_w, inception_3a_pool_proj_b,
        inception_3a_output);
    topology.add(
        inception_3b_1x1, inception_3b_1x1_w, inception_3b_1x1_b,
        inception_3b_3x3_reduce, inception_3b_3x3_reduce_w, inception_3b_3x3_reduce_b,
        inception_3b_3x3, inception_3b_3x3_w, inception_3b_3x3_b,
        inception_3b_5x5_reduce, inception_3b_5x5_reduce_w, inception_3b_5x5_reduce_b,
        inception_3b_5x5, inception_3b_5x5_w, inception_3b_5x5_b,
        inception_3b_pool,
        inception_3b_pool_proj, inception_3b_pool_proj_w, inception_3b_pool_proj_b,
        inception_3b_output,
        pool3_3x3_s2);
    topology.add(
        inception_4a_1x1, inception_4a_1x1_w, inception_4a_1x1_b,
        inception_4a_3x3_reduce, inception_4a_3x3_reduce_w, inception_4a_3x3_reduce_b,
        inception_4a_3x3, inception_4a_3x3_w, inception_4a_3x3_b,
        inception_4a_5x5_reduce, inception_4a_5x5_reduce_w, inception_4a_5x5_reduce_b,
        inception_4a_5x5, inception_4a_5x5_w, inception_4a_5x5_b,
        inception_4a_pool,
        inception_4a_pool_proj, inception_4a_pool_proj_w, inception_4a_pool_proj_b,
        inception_4a_output);
    topology.add(
        inception_4b_1x1, inception_4b_1x1_w, inception_4b_1x1_b,
        inception_4b_3x3_reduce, inception_4b_3x3_reduce_w, inception_4b_3x3_reduce_b,
        inception_4b_3x3, inception_4b_3x3_w, inception_4b_3x3_b,
        inception_4b_5x5_reduce, inception_4b_5x5_reduce_w, inception_4b_5x5_reduce_b,
        inception_4b_5x5, inception_4b_5x5_w, inception_4b_5x5_b,
        inception_4b_pool,
        inception_4b_pool_proj, inception_4b_pool_proj_w, inception_4b_pool_proj_b,
        inception_4b_output);
    topology.add(
        inception_4c_1x1, inception_4c_1x1_w, inception_4c_1x1_b,
        inception_4c_3x3_reduce, inception_4c_3x3_reduce_w, inception_4c_3x3_reduce_b,
        inception_4c_3x3, inception_4c_3x3_w, inception_4c_3x3_b,
        inception_4c_5x5_reduce, inception_4c_5x5_reduce_w, inception_4c_5x5_reduce_b,
        inception_4c_5x5, inception_4c_5x5_w, inception_4c_5x5_b,
        inception_4c_pool,
        inception_4c_pool_proj, inception_4c_pool_proj_w, inception_4c_pool_proj_b,
        inception_4c_output);
    topology.add(
        inception_4d_1x1, inception_4d_1x1_w, inception_4d_1x1_b,
        inception_4d_3x3_reduce, inception_4d_3x3_reduce_w, inception_4d_3x3_reduce_b,
        inception_4d_3x3, inception_4d_3x3_w, inception_4d_3x3_b,
        inception_4d_5x5_reduce, inception_4d_5x5_reduce_w, inception_4d_5x5_reduce_b,
        inception_4d_5x5, inception_4d_5x5_w, inception_4d_5x5_b,
        inception_4d_pool,
        inception_4d_pool_proj, inception_4d_pool_proj_w, inception_4d_pool_proj_b,
        inception_4d_output);
    topology.add(
        inception_4e_1x1, inception_4e_1x1_w, inception_4e_1x1_b,
        inception_4e_3x3_reduce, inception_4e_3x3_reduce_w, inception_4e_3x3_reduce_b,
        inception_4e_3x3, inception_4e_3x3_w, inception_4e_3x3_b,
        inception_4e_5x5_reduce, inception_4e_5x5_reduce_w, inception_4e_5x5_reduce_b,
        inception_4e_5x5, inception_4e_5x5_w, inception_4e_5x5_b,
        inception_4e_pool,
        inception_4e_pool_proj, inception_4e_pool_proj_w, inception_4e_pool_proj_b,
        inception_4e_output,
        pool4_3x3_s2);
    topology.add(
        inception_5a_1x1, inception_5a_1x1_w, inception_5a_1x1_b,
        inception_5a_3x3_reduce, inception_5a_3x3_reduce_w, inception_5a_3x3_reduce_b,
        inception_5a_3x3, inception_5a_3x3_w, inception_5a_3x3_b,
        inception_5a_5x5_reduce, inception_5a_5x5_reduce_w, inception_5a_5x5_reduce_b,
        inception_5a_5x5, inception_5a_5x5_w, inception_5a_5x5_b,
        inception_5a_pool,
        inception_5a_pool_proj, inception_5a_pool_proj_w, inception_5a_pool_proj_b,
        inception_5a_output);
    topology.add(
        inception_5b_1x1, inception_5b_1x1_w, inception_5b_1x1_b,
        inception_5b_3x3_reduce, inception_5b_3x3_reduce_w, inception_5b_3x3_reduce_b,
        inception_5b_3x3, inception_5b_3x3_w, inception_5b_3x3_b,
        inception_5b_5x5_reduce, inception_5b_5x5_reduce_w, inception_5b_5x5_reduce_b,
        inception_5b_5x5, inception_5b_5x5_w, inception_5b_5x5_b,
        inception_5b_pool,
        inception_5b_pool_proj, inception_5b_pool_proj_w, inception_5b_pool_proj_b,
        inception_5b_output,
        pool5_7x7_s1,
        loss3_classifier, loss3_classifier_w, loss3_classifier_b,
        softmax);
    return topology;
}
