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

// Building GoogLeNet v1 network with loading weights & biases from file
cldnn::topology build_googlenetv1(const std::string& weights_dir, weights_optimizer& wo, cldnn::layout& input_layout, int32_t batch_size, bool use_half)
{
    input_layout.data_type = use_half ? data_types::f16 : data_types::f32;
    // [224x224x3xB] convolution->relu->pooling->lrn [1000xB]
    input_layout.size = { format::byxf,{ batch_size, 224, 224, 3 } };
    auto input = cldnn::input_layout("input", input_layout);

    // create conversion to yxfb format and subtract mean values
    tensor reorder_size = input_layout.size.transform(format::yxfb, 1);
    auto reordered_input = reorder(
        "reorder",
        input,
        { input_layout.data_type, reorder_size },
        std::vector<float>{ 104.0f, 117.0f, 123.0f });

    wo.create_weights_from_file(join_path(weights_dir, "conv1_7x7_s2_weights.nnd"), file::convolution),
    wo.create_weights_from_file(join_path(weights_dir, "conv1_7x7_s2_bias.nnd"),  file::bias),
    auto conv1_7x7_s2 = convolution::create(
    {
        mem_format,
        {
            reordered_input,
            wo.create_weights_from_file(join_path(weights_dir, "conv1_7x7_s2_weights.nnd"), file::convolution),
            wo.create_weights_from_file(join_path(weights_dir, "conv1_7x7_s2_bias.nnd"),  file::bias),
        },
        { 0,{ -3, -3 }, 0 },
        { 1,{ 2, 2 }, 1 },
        padding::zero,
        1,
        true 
    });

    auto pool1_3x3_s2 = pooling::create(
    {
        pooling::mode::max,
        mem_format,
        conv1_7x7_s2,
        { 1,{ 2,2 },1 }, // strd
        { 1,{ 3,3 },1 }, // kernel
        padding::zero
    });

    auto pool1_norm1 = normalization::response::create(
    {
        mem_format,
        pool1_3x3_s2,
        5,
        padding::zero,
        1.0f,
        0.00002f,
        0.75f
    });

    auto conv2_3x3_reduce = convolution::create(
    {
        mem_format,
        {
            pool1_norm1,
            wo.create_weights_from_file(join_path(weights_dir, "conv2_3x3_reduce_weights.nnd"), file::convolution),
            wo.create_weights_from_file(join_path(weights_dir, "conv2_3x3_reduce_bias.nnd"),  file::bias),
        },
        { 0,{ 0, 0 }, 0 },
        { 1,{ 1, 1 }, 1 },
        padding::zero,
        1,
        true 
    });

    auto conv2_3x3 = convolution::create(
    {
        mem_format,
        {
            conv2_3x3_reduce,
            wo.create_weights_from_file(join_path(weights_dir, "conv2_3x3_weights.nnd"), file::convolution),
            wo.create_weights_from_file(join_path(weights_dir, "conv2_3x3_bias.nnd"),  file::bias),
        },
        { 0,{ -1, -1 }, 0 },
        { 1,{ 1, 1 }, 1 },
        padding::zero,
        1,
        true
    });

    auto conv2_norm2 = normalization::response::create(
    {
        mem_format,
        conv2_3x3,
        5,
        padding::zero,
        1.0f,
        0.00002f,
        0.75f
    });

    auto pool2_3x3_s2 = pooling::create(
    {
        pooling::mode::max,
        mem_format,
        conv2_norm2,
        { 1,{ 2,2 },1 }, // strd
        { 1,{ 3,3 },1 }, // kernel
        padding::zero
    });

    // ----------- END OF PIPE -------------
    // Inception 1
    // 1st branch
    auto inception_3a_1x1 = convolution::create(
    {
        mem_format,
        {
            pool2_3x3_s2,
            wo.create_weights_from_file(join_path(weights_dir, "inception_3a_1x1_weights.nnd"), file::convolution),
            wo.create_weights_from_file(join_path(weights_dir, "inception_3a_1x1_bias.nnd"),  file::bias),
        },
        { 0,{ 0, 0 }, 0 },
        { 1,{ 1, 1 }, 1 },
        padding::zero,
        1,
        true
    });

    // 2nd branch

    auto inception_3a_3x3_reduce = convolution::create(
    {
        mem_format,
        {
            pool2_3x3_s2,
            wo.create_weights_from_file(join_path(weights_dir, "inception_3a_3x3_reduce_weights.nnd"), file::convolution),
            wo.create_weights_from_file(join_path(weights_dir, "inception_3a_3x3_reduce_bias.nnd"),  file::bias),
        },
        { 0,{ 0, 0 }, 0 },
        { 1,{ 1, 1 }, 1 },
        padding::zero,
        1,
        true
    });


    auto inception_3a_3x3 = convolution::create(
    {
        mem_format,
        {
            inception_3a_3x3_reduce,
            wo.create_weights_from_file(join_path(weights_dir, "inception_3a_3x3_weights.nnd"), file::convolution),
            wo.create_weights_from_file(join_path(weights_dir, "inception_3a_3x3_bias.nnd"),  file::bias),
        },
        { 0,{ -1, -1 }, 0 },
        { 1,{ 1, 1 }, 1 },
        padding::zero,
        1,
        true
    });

    // 3rd branch

    auto inception_3a_5x5_reduce = convolution::create(
    {
        mem_format,
        {
            pool2_3x3_s2,
            wo.create_weights_from_file(join_path(weights_dir, "inception_3a_5x5_reduce_weights.nnd"), file::convolution),
            wo.create_weights_from_file(join_path(weights_dir, "inception_3a_5x5_reduce_bias.nnd"),  file::bias),
        },
        { 0,{ 0, 0 }, 0 },
        { 1,{ 1, 1 }, 1 },
        padding::zero,
        1,
        true
    });


    auto inception_3a_5x5 = convolution::create(
    {
        mem_format,
        {
            inception_3a_5x5_reduce,
            wo.create_weights_from_file(join_path(weights_dir, "inception_3a_5x5_weights.nnd"), file::convolution),
            wo.create_weights_from_file(join_path(weights_dir, "inception_3a_5x5_bias.nnd"),  file::bias),
        },
        { 0,{ -2, -2 }, 0 },
        { 1,{ 1, 1 }, 1 },
        padding::zero,
        1,
        true
    });

    // 4th branch

    auto inception_3a_pool = pooling::create(
    {
        pooling::mode::max,
        mem_format,
        pool2_3x3_s2,
        { 0,{ -1, -1 },0 }, //padding 
        { 1,{ 1, 1 },1 }, // strd
        { 1,{ 3, 3 },1 }, // kernel
        padding::zero
    });

    auto inception_3a_pool_proj = convolution::create(
    {
        mem_format,
        {
            inception_3a_pool,
            wo.create_weights_from_file(join_path(weights_dir, "inception_3a_pool_proj_weights.nnd"), file::convolution),
            wo.create_weights_from_file(join_path(weights_dir, "inception_3a_pool_proj_bias.nnd"),  file::bias),
        },
        { 0,{ 0, 0 }, 0 },
        { 1,{ 1, 1 }, 1 },
        padding::zero,
        1,
        true
    });

    auto inception_3a_output = depth_concatenate::create(
    {
        mem_format,
        {
            inception_3a_1x1,
            inception_3a_3x3,
            inception_3a_5x5,
            inception_3a_pool_proj
        }
    });
    // End of 1st nception
    
    // --------------------- 2nd inception ---------------------------------

    // 1st branch
    auto inception_3b_1x1 = convolution::create(
    {
        mem_format,
        {
            inception_3a_output,
            wo.create_weights_from_file(join_path(weights_dir, "inception_3b_1x1_weights.nnd"), file::convolution),
            wo.create_weights_from_file(join_path(weights_dir, "inception_3b_1x1_bias.nnd"),  file::bias),
        },
        { 0,{ 0, 0 }, 0 },
        { 1,{ 1, 1 }, 1 },
        padding::zero,
        1,
        true
    });

    // 2nd branch

    auto inception_3b_3x3_reduce = convolution::create(
    {
        mem_format,
        {
            inception_3a_output,
            wo.create_weights_from_file(join_path(weights_dir, "inception_3b_3x3_reduce_weights.nnd"), file::convolution),
            wo.create_weights_from_file(join_path(weights_dir, "inception_3b_3x3_reduce_bias.nnd"),  file::bias),
        },
        { 0,{ 0, 0 }, 0 },
        { 1,{ 1, 1 }, 1 },
        padding::zero,
        1,
        true
    });

    auto inception_3b_3x3 = convolution::create(
    {
        mem_format,
        {
            inception_3b_3x3_reduce,
            wo.create_weights_from_file(join_path(weights_dir, "inception_3b_3x3_weights.nnd"), file::convolution),
            wo.create_weights_from_file(join_path(weights_dir, "inception_3b_3x3_bias.nnd"),  file::bias),
        },
        { 0,{ -1, -1 }, 0 },
        { 1,{ 1, 1 }, 1 },
        padding::zero,
        1,
        true
    });

    // 3rd branch

    auto inception_3b_5x5_reduce = convolution::create(
    {
        mem_format,
        {
            inception_3a_output,
            wo.create_weights_from_file(join_path(weights_dir, "inception_3b_5x5_reduce_weights.nnd"), file::convolution),
            wo.create_weights_from_file(join_path(weights_dir, "inception_3b_5x5_reduce_bias.nnd"),  file::bias),
        },
        { 0,{ 0, 0 }, 0 },
        { 1,{ 1, 1 }, 1 },
        padding::zero,
        1,
        true
    });

    auto inception_3b_5x5 = convolution::create(
    {
        mem_format,
        {
            inception_3b_5x5_reduce,
            wo.create_weights_from_file(join_path(weights_dir, "inception_3b_5x5_weights.nnd"), file::convolution),
            wo.create_weights_from_file(join_path(weights_dir, "inception_3b_5x5_bias.nnd"),  file::bias),
        },
        { 0,{ -2, -2 }, 0 },
        { 1,{ 1, 1 }, 1 },
        padding::zero,
        1,
        true
    });

    // 4th branch

    auto inception_3b_pool = pooling::create(
    {
        pooling::mode::max,
        mem_format,
        inception_3a_output,
        { 0,{ -1, -1 },0 }, //padding 
        { 1,{ 1, 1 },1 }, // strd
        { 1,{ 3, 3 },1 }, // kernel
        padding::zero
    });

    auto inception_3b_pool_proj = convolution::create(
    {
        mem_format,
        {
            inception_3b_pool,
            wo.create_weights_from_file(join_path(weights_dir, "inception_3b_pool_proj_weights.nnd"), file::convolution),
            wo.create_weights_from_file(join_path(weights_dir, "inception_3b_pool_proj_bias.nnd"),  file::bias),
        },
        { 0,{ 0, 0 }, 0 },
        { 1,{ 1, 1 }, 1 },
        padding::zero,
        1,
        true
    });

    auto inception_3b_output = depth_concatenate::create(
    {
        mem_format,
        {
            inception_3b_1x1,
            inception_3b_3x3,
            inception_3b_5x5,
            inception_3b_pool_proj
        }
    });

    // End of 2nd inception

    auto pool3_3x3_s2 = pooling::create(
    {
        pooling::mode::max,
        mem_format,
        inception_3b_output,
        { 1,{ 2, 2 },1 }, // strd
        { 1,{ 3, 3 },1 }, // kernel
        padding::zero
    });

    // --------------------- 3rd inception ---------------------------------
    // 1st branch
    auto inception_4a_1x1 = convolution::create(
    {
        mem_format,
        {
            pool3_3x3_s2,
            wo.create_weights_from_file(join_path(weights_dir, "inception_4a_1x1_weights.nnd"), file::convolution),
            wo.create_weights_from_file(join_path(weights_dir, "inception_4a_1x1_bias.nnd"),  file::bias),
        },
        { 0,{ 0, 0 }, 0 },
        { 1,{ 1, 1 }, 1 },
        padding::zero,
        1,
        true
    });

    // 2nd branch

    auto inception_4a_3x3_reduce = convolution::create(
    {
        mem_format,
        {
            pool3_3x3_s2,
            wo.create_weights_from_file(join_path(weights_dir, "inception_4a_3x3_reduce_weights.nnd"), file::convolution),
            wo.create_weights_from_file(join_path(weights_dir, "inception_4a_3x3_reduce_bias.nnd"),  file::bias),
        },
        { 0,{ 0, 0 }, 0 },
        { 1,{ 1, 1 }, 1 },
        padding::zero,
        1,
        true
    });

    auto inception_4a_3x3 = convolution::create(
    {
        mem_format,
        {
            inception_4a_3x3_reduce,
            wo.create_weights_from_file(join_path(weights_dir, "inception_4a_3x3_weights.nnd"), file::convolution),
            wo.create_weights_from_file(join_path(weights_dir, "inception_4a_3x3_bias.nnd"),  file::bias),
        },
        { 0,{ -1, -1 }, 0 },
        { 1,{ 1, 1 }, 1 },
        padding::zero,
        1,
        true
    });

    // 3rd branch

    auto inception_4a_5x5_reduce = convolution::create(
    {
        mem_format,
        {
            pool3_3x3_s2,
            wo.create_weights_from_file(join_path(weights_dir, "inception_4a_5x5_reduce_weights.nnd"), file::convolution),
            wo.create_weights_from_file(join_path(weights_dir, "inception_4a_5x5_reduce_bias.nnd"),  file::bias),
        },
        { 0,{ 0, 0 }, 0 },
        { 1,{ 1, 1 }, 1 },
        padding::zero,
        1,
        true
    });

    auto inception_4a_5x5 = convolution::create(
    {
        mem_format,
        {
            inception_4a_5x5_reduce,
            wo.create_weights_from_file(join_path(weights_dir, "inception_4a_5x5_weights.nnd"), file::convolution),
            wo.create_weights_from_file(join_path(weights_dir, "inception_4a_5x5_bias.nnd"),  file::bias),
        },
        { 0,{ -2, -2 }, 0 },
        { 1,{ 1, 1 }, 1 },
        padding::zero,
        1,
        true
    });

    // 4th branch

    auto inception_4a_pool = pooling::create(
    {
        pooling::mode::max,
        mem_format,
        pool3_3x3_s2,
        { 0,{ -1, -1 },0 }, //padding 
        { 1,{ 1, 1 },1 }, // strd
        { 1,{ 3, 3 },1 }, // kernel
        padding::zero
    });

    auto inception_4a_pool_proj = convolution::create(
    {
        mem_format,
        {
            inception_4a_pool,
            wo.create_weights_from_file(join_path(weights_dir, "inception_4a_pool_proj_weights.nnd"), file::convolution),
            wo.create_weights_from_file(join_path(weights_dir, "inception_4a_pool_proj_bias.nnd"),  file::bias),
        },
        { 0,{ 0, 0 }, 0 },
        { 1,{ 1, 1 }, 1 },
        padding::zero,
        1,
        true
    });

    auto inception_4a_output = depth_concatenate::create(
    {
        mem_format,
        {
            inception_4a_1x1,
            inception_4a_3x3,
            inception_4a_5x5,
            inception_4a_pool_proj
        }
    });
    // End of 3rd inception

    // --------------------- 4th inception ---------------------------------
    // 1st branch
    auto inception_4b_1x1 = convolution::create(
    {
        mem_format,
        {
            inception_4a_output,
            wo.create_weights_from_file(join_path(weights_dir, "inception_4b_1x1_weights.nnd"), file::convolution),
            wo.create_weights_from_file(join_path(weights_dir, "inception_4b_1x1_bias.nnd"),  file::bias),
        },
        { 0,{ 0, 0 }, 0 },
        { 1,{ 1, 1 }, 1 },
        padding::zero,
        1,
        true
    });

    // 2nd branch

    auto inception_4b_3x3_reduce = convolution::create(
    {
        mem_format,
        {
            inception_4a_output,
            wo.create_weights_from_file(join_path(weights_dir, "inception_4b_3x3_reduce_weights.nnd"), file::convolution),
            wo.create_weights_from_file(join_path(weights_dir, "inception_4b_3x3_reduce_bias.nnd"),  file::bias),
        },
        { 0,{ 0, 0 }, 0 },
        { 1,{ 1, 1 }, 1 },
        padding::zero,
        1,
        true
    });

    auto inception_4b_3x3 = convolution::create(
    {
        mem_format,
        {
            inception_4b_3x3_reduce,
            wo.create_weights_from_file(join_path(weights_dir, "inception_4b_3x3_weights.nnd"), file::convolution),
            wo.create_weights_from_file(join_path(weights_dir, "inception_4b_3x3_bias.nnd"),  file::bias),
        },
        { 0,{ -1, -1 }, 0 },
        { 1,{ 1, 1 }, 1 },
        padding::zero,
        1,
        true
    });

    // 3rd branch

    auto inception_4b_5x5_reduce = convolution::create(
    {
        mem_format,
        {
            inception_4a_output,
            wo.create_weights_from_file(join_path(weights_dir, "inception_4b_5x5_reduce_weights.nnd"), file::convolution),
            wo.create_weights_from_file(join_path(weights_dir, "inception_4b_5x5_reduce_bias.nnd"),  file::bias),
        },
        { 0,{ 0, 0 }, 0 },
        { 1,{ 1, 1 }, 1 },
        padding::zero,
        1,
        true
    });

    auto inception_4b_5x5 = convolution::create(
    {
        mem_format,
        {
            inception_4b_5x5_reduce,
            wo.create_weights_from_file(join_path(weights_dir, "inception_4b_5x5_weights.nnd"), file::convolution),
            wo.create_weights_from_file(join_path(weights_dir, "inception_4b_5x5_bias.nnd"),  file::bias),
        },
        { 0,{ -2, -2 }, 0 },
        { 1,{ 1, 1 }, 1 },
        padding::zero,
        1,
        true
    });

    // 4th branch

    auto inception_4b_pool = pooling::create(
    {
        pooling::mode::max,
        mem_format,
        inception_4a_output,
        { 0,{ -1, -1 },0 }, //padding 
        { 1,{ 1, 1 },1 }, // strd
        { 1,{ 3, 3 },1 }, // kernel
        padding::zero
    });

    auto inception_4b_pool_proj = convolution::create(
    {
        mem_format,
        {
            inception_4b_pool,
            wo.create_weights_from_file(join_path(weights_dir, "inception_4b_pool_proj_weights.nnd"), file::convolution),
            wo.create_weights_from_file(join_path(weights_dir, "inception_4b_pool_proj_bias.nnd"),  file::bias),
        },
        { 0,{ 0, 0 }, 0 },
        { 1,{ 1, 1 }, 1 },
        padding::zero,
        1,
        true
    });

    auto inception_4b_output = depth_concatenate::create(
    {
        mem_format,
        {
            inception_4b_1x1,
            inception_4b_3x3,
            inception_4b_5x5,
            inception_4b_pool_proj
        }
    });

    // End of 4th inception
    
    // --------------------- 5th inception ---------------------------------
    // 1st branch
    auto inception_4c_1x1 = convolution::create(
    {
        mem_format,
        {
            inception_4b_output,
            wo.create_weights_from_file(join_path(weights_dir, "inception_4c_1x1_weights.nnd"), file::convolution),
            wo.create_weights_from_file(join_path(weights_dir, "inception_4c_1x1_bias.nnd"),  file::bias),
        },
        { 0,{ 0, 0 }, 0 },
        { 1,{ 1, 1 }, 1 },
        padding::zero,
        1,
        true
    });

    // 2nd branch

    auto inception_4c_3x3_reduce = convolution::create(
    {
        mem_format,
        {
            inception_4b_output,
            wo.create_weights_from_file(join_path(weights_dir, "inception_4c_3x3_reduce_weights.nnd"), file::convolution),
            wo.create_weights_from_file(join_path(weights_dir, "inception_4c_3x3_reduce_bias.nnd"),  file::bias),
        },
        { 0,{ 0, 0 }, 0 },
        { 1,{ 1, 1 }, 1 },
        padding::zero,
        1,
        true
    });

    auto inception_4c_3x3 = convolution::create(
    {
        mem_format,
        {
            inception_4c_3x3_reduce,
            wo.create_weights_from_file(join_path(weights_dir, "inception_4c_3x3_weights.nnd"), file::convolution),
            wo.create_weights_from_file(join_path(weights_dir, "inception_4c_3x3_bias.nnd"),  file::bias),
        },
        { 0,{ -1, -1 }, 0 },
        { 1,{ 1, 1 }, 1 },
        padding::zero,
        1,
        true
    });

    // 3rd branch
    auto inception_4c_5x5_reduce = convolution::create(
    {
        mem_format,
        {
            inception_4b_output,
            wo.create_weights_from_file(join_path(weights_dir, "inception_4c_5x5_reduce_weights.nnd"), file::convolution),
            wo.create_weights_from_file(join_path(weights_dir, "inception_4c_5x5_reduce_bias.nnd"),  file::bias),
        },
        { 0,{ 0, 0 }, 0 },
        { 1,{ 1, 1 }, 1 },
        padding::zero,
        1,
        true
    });

    auto inception_4c_5x5 = convolution::create(
    {
        mem_format,
        {
            inception_4c_5x5_reduce,
            wo.create_weights_from_file(join_path(weights_dir, "inception_4c_5x5_weights.nnd"), file::convolution),
            wo.create_weights_from_file(join_path(weights_dir, "inception_4c_5x5_bias.nnd"),  file::bias),
        },
        { 0,{ -2, -2 }, 0 },
        { 1,{ 1, 1 }, 1 },
        padding::zero,
        1,
        true
    });

    // 4th branch

    auto inception_4c_pool = pooling::create(
    {
        pooling::mode::max,
        mem_format,
        inception_4b_output,
        { 0,{ -1, -1 },0 }, //padding 
        { 1,{ 1, 1 },1 }, // strd
        { 1,{ 3, 3 },1 }, // kernel
        padding::zero
    });

    auto inception_4c_pool_proj = convolution::create(
    {
        mem_format,
        {
            inception_4c_pool,
            wo.create_weights_from_file(join_path(weights_dir, "inception_4c_pool_proj_weights.nnd"), file::convolution),
            wo.create_weights_from_file(join_path(weights_dir, "inception_4c_pool_proj_bias.nnd"),  file::bias),
        },
        { 0,{ 0, 0 }, 0 },
        { 1,{ 1, 1 }, 1 },
        padding::zero,
        1,
        true
    });

    auto inception_4c_output = depth_concatenate::create(
    {
        mem_format,
        {
            inception_4c_1x1,
            inception_4c_3x3,
            inception_4c_5x5,
            inception_4c_pool_proj
        }
    });

    // --------------------- 6th inception ---------------------------------
    // 1st branch
    auto inception_4d_1x1 = convolution::create(
    {
        mem_format,
        {
            inception_4c_output,
            wo.create_weights_from_file(join_path(weights_dir, "inception_4d_1x1_weights.nnd"), file::convolution),
            wo.create_weights_from_file(join_path(weights_dir, "inception_4d_1x1_bias.nnd"),  file::bias),
        },
        { 0,{ 0, 0 }, 0 },
        { 1,{ 1, 1 }, 1 },
        padding::zero,
        1,
        true
    });

    // 2nd branch

    auto inception_4d_3x3_reduce = convolution::create(
    {
        mem_format,
        {
            inception_4c_output,
            wo.create_weights_from_file(join_path(weights_dir, "inception_4d_3x3_reduce_weights.nnd"), file::convolution),
            wo.create_weights_from_file(join_path(weights_dir, "inception_4d_3x3_reduce_bias.nnd"),  file::bias),
        },
        { 0,{ 0, 0 }, 0 },
        { 1,{ 1, 1 }, 1 },
        padding::zero,
        1,
        true
    });

    auto inception_4d_3x3 = convolution::create(
    {
        mem_format,
        {
            inception_4d_3x3_reduce,
            wo.create_weights_from_file(join_path(weights_dir, "inception_4d_3x3_weights.nnd"), file::convolution),
            wo.create_weights_from_file(join_path(weights_dir, "inception_4d_3x3_bias.nnd"),  file::bias),
        },
        { 0,{ -1, -1 }, 0 },
        { 1,{ 1, 1 }, 1 },
        padding::zero,
        1,
        true
    });

    // 3rd branch
    auto inception_4d_5x5_reduce = convolution::create(
    {
        mem_format,
        {
            inception_4c_output,
            wo.create_weights_from_file(join_path(weights_dir, "inception_4d_5x5_reduce_weights.nnd"), file::convolution),
            wo.create_weights_from_file(join_path(weights_dir, "inception_4d_5x5_reduce_bias.nnd"),  file::bias),
        },
        { 0,{ 0, 0 }, 0 },
        { 1,{ 1, 1 }, 1 },
        padding::zero,
        1,
        true
    });

    auto inception_4d_5x5 = convolution::create(
    {
        mem_format,
        {
            inception_4d_5x5_reduce,
            wo.create_weights_from_file(join_path(weights_dir, "inception_4d_5x5_weights.nnd"), file::convolution),
            wo.create_weights_from_file(join_path(weights_dir, "inception_4d_5x5_bias.nnd"),  file::bias),
        },
        { 0,{ -2, -2 }, 0 },
        { 1,{ 1, 1 }, 1 },
        padding::zero,
        1,
        true
    });

    // 4th branch

    auto inception_4d_pool = pooling::create(
    {
        pooling::mode::max,
        mem_format,
        inception_4c_output,
        { 0,{ -1, -1 },0 }, //padding 
        { 1,{ 1, 1 },1 }, // strd
        { 1,{ 3, 3 },1 }, // kernel
        padding::zero
    });

    auto inception_4d_pool_proj = convolution::create(
    {
        mem_format,
        {
            inception_4d_pool,
            wo.create_weights_from_file(join_path(weights_dir, "inception_4d_pool_proj_weights.nnd"), file::convolution),
            wo.create_weights_from_file(join_path(weights_dir, "inception_4d_pool_proj_bias.nnd"),  file::bias),
        },
        { 0,{ 0, 0 }, 0 },
        { 1,{ 1, 1 }, 1 },
        padding::zero,
        1,
        true
    });

    auto inception_4d_output = depth_concatenate::create(
    {
        mem_format,
        {
            inception_4d_1x1,
            inception_4d_3x3,
            inception_4d_5x5,
            inception_4d_pool_proj
        }
    });

    // --------------------- 7th inception ---------------------------------
    // 1st branch
    auto inception_4e_1x1 = convolution::create(
    {
        mem_format,
        {
            inception_4d_output,
            wo.create_weights_from_file(join_path(weights_dir, "inception_4e_1x1_weights.nnd"), file::convolution),
            wo.create_weights_from_file(join_path(weights_dir, "inception_4e_1x1_bias.nnd"),  file::bias),
        },
        { 0,{ 0, 0 }, 0 },
        { 1,{ 1, 1 }, 1 },
        padding::zero,
        1,
        true
    });

    // 2nd branch

    auto inception_4e_3x3_reduce = convolution::create(
    {
        mem_format,
        {
            inception_4d_output,
            wo.create_weights_from_file(join_path(weights_dir, "inception_4e_3x3_reduce_weights.nnd"), file::convolution),
            wo.create_weights_from_file(join_path(weights_dir, "inception_4e_3x3_reduce_bias.nnd"),  file::bias),
        },
        { 0,{ 0, 0 }, 0 },
        { 1,{ 1, 1 }, 1 },
        padding::zero,
        1,
        true
    });

    auto inception_4e_3x3 = convolution::create(
    {
        mem_format,
        {
            inception_4e_3x3_reduce,
            wo.create_weights_from_file(join_path(weights_dir, "inception_4e_3x3_weights.nnd"), file::convolution),
            wo.create_weights_from_file(join_path(weights_dir, "inception_4e_3x3_bias.nnd"),  file::bias),
        },
        { 0,{ -1, -1 }, 0 },
        { 1,{ 1, 1 }, 1 },
        padding::zero,
        1,
        true
    });

    // 3rd branch
    auto inception_4e_5x5_reduce = convolution::create(
    {
        mem_format,
        {
            inception_4d_output,
            wo.create_weights_from_file(join_path(weights_dir, "inception_4e_5x5_reduce_weights.nnd"), file::convolution),
            wo.create_weights_from_file(join_path(weights_dir, "inception_4e_5x5_reduce_bias.nnd"),  file::bias),
        },
        { 0,{ 0, 0 }, 0 },
        { 1,{ 1, 1 }, 1 },
        padding::zero,
        1,
        true
    });

    auto inception_4e_5x5 = convolution::create(
    {
        mem_format,
        {
            inception_4e_5x5_reduce,
            wo.create_weights_from_file(join_path(weights_dir, "inception_4e_5x5_weights.nnd"), file::convolution),
            wo.create_weights_from_file(join_path(weights_dir, "inception_4e_5x5_bias.nnd"),  file::bias),
        },
        { 0,{ -2, -2 }, 0 },
        { 1,{ 1, 1 }, 1 },
        padding::zero,
        1,
        true
    });

    // 4th branch

    auto inception_4e_pool = pooling::create(
    {
        pooling::mode::max,
        mem_format,
        inception_4d_output,
        { 0,{ -1, -1 },0 }, //padding 
        { 1,{ 1, 1 },1 }, // strd
        { 1,{ 3, 3 },1 }, // kernel
        padding::zero
    });

    auto inception_4e_pool_proj = convolution::create(
    {
        mem_format,
        {
            inception_4e_pool,
            wo.create_weights_from_file(join_path(weights_dir, "inception_4e_pool_proj_weights.nnd"), file::convolution),
            wo.create_weights_from_file(join_path(weights_dir, "inception_4e_pool_proj_bias.nnd"),  file::bias),
        },
        { 0,{ 0, 0 }, 0 },
        { 1,{ 1, 1 }, 1 },
        padding::zero,
        1,
        true
    });

    auto inception_4e_output = depth_concatenate::create(
    {
        mem_format,
        {
            inception_4e_1x1,
            inception_4e_3x3,
            inception_4e_5x5,
            inception_4e_pool_proj
        }
    });
   
    // ----------- End of 7th inception -------------------

    auto pool4_3x3_s2 = pooling::create(
    {
        pooling::mode::max,
        mem_format,
        inception_4e_output,
        { 1,{ 2, 2 },1 }, // strd
        { 1,{ 3, 3 },1 }, // kernel
        padding::zero
    });

    // --------------------- 8th inception ---------------------------------
    // 1st branch
    auto inception_5a_1x1 = convolution::create(
    {
        mem_format,
        {
            pool4_3x3_s2,
            wo.create_weights_from_file(join_path(weights_dir, "inception_5a_1x1_weights.nnd"), file::convolution),
            wo.create_weights_from_file(join_path(weights_dir, "inception_5a_1x1_bias.nnd"),  file::bias),
        },
        { 0,{ 0, 0 }, 0 },
        { 1,{ 1, 1 }, 1 },
        padding::zero,
        1,
        true
    });

    // 2nd branch

    auto inception_5a_3x3_reduce = convolution::create(
    {
        mem_format,
        {
            pool4_3x3_s2,
            wo.create_weights_from_file(join_path(weights_dir, "inception_5a_3x3_reduce_weights.nnd"), file::convolution),
            wo.create_weights_from_file(join_path(weights_dir, "inception_5a_3x3_reduce_bias.nnd"),  file::bias),
        },
        { 0,{ 0, 0 }, 0 },
        { 1,{ 1, 1 }, 1 },
        padding::zero,
        1,
        true
    });

    auto inception_5a_3x3 = convolution::create(
    {
        mem_format,
        {
            inception_5a_3x3_reduce,
            wo.create_weights_from_file(join_path(weights_dir, "inception_5a_3x3_weights.nnd"), file::convolution),
            wo.create_weights_from_file(join_path(weights_dir, "inception_5a_3x3_bias.nnd"),  file::bias),
        },
        { 0,{ -1, -1 }, 0 },
        { 1,{ 1, 1 }, 1 },
        padding::zero,
        1,
        true
    });

    // 3rd branch
    auto inception_5a_5x5_reduce = convolution::create(
    {
        mem_format,
        {
            pool4_3x3_s2,
            wo.create_weights_from_file(join_path(weights_dir, "inception_5a_5x5_reduce_weights.nnd"), file::convolution),
            wo.create_weights_from_file(join_path(weights_dir, "inception_5a_5x5_reduce_bias.nnd"),  file::bias),
        },
        { 0,{ 0, 0 }, 0 },
        { 1,{ 1, 1 }, 1 },
        padding::zero,
        1,
        true
    });

    auto inception_5a_5x5 = convolution::create(
    {
        mem_format,
        {
            inception_5a_5x5_reduce,
            wo.create_weights_from_file(join_path(weights_dir, "inception_5a_5x5_weights.nnd"), file::convolution),
            wo.create_weights_from_file(join_path(weights_dir, "inception_5a_5x5_bias.nnd"),  file::bias),
        },
        { 0,{ -2, -2 }, 0 },
        { 1,{ 1, 1 }, 1 },
        padding::zero,
        1,
        true
    });

    // 4th branch

    auto inception_5a_pool = pooling::create(
    {
        pooling::mode::max,
        mem_format,
        pool4_3x3_s2,
        { 0,{ -1, -1 },0 }, //padding 
        { 1,{ 1, 1 },1 }, // strd
        { 1,{ 3, 3 },1 }, // kernel
        padding::zero
    });

    auto inception_5a_pool_proj = convolution::create(
    {
        mem_format,
        {
            inception_5a_pool,
            wo.create_weights_from_file(join_path(weights_dir, "inception_5a_pool_proj_weights.nnd"), file::convolution),
            wo.create_weights_from_file(join_path(weights_dir, "inception_5a_pool_proj_bias.nnd"),  file::bias),
        },
        { 0,{ 0, 0 }, 0 },
        { 1,{ 1, 1 }, 1 },
        padding::zero,
        1,
        true
    });

    auto inception_5a_output = depth_concatenate::create(
    {
        mem_format,
        {
            inception_5a_1x1,
            inception_5a_3x3,
            inception_5a_5x5,
            inception_5a_pool_proj
        }
    });

    // --------------------- 8th inception ---------------------------------
    // 1st branch
    auto inception_5b_1x1 = convolution::create(
    {
        mem_format,
        {
            inception_5a_output,
            wo.create_weights_from_file(join_path(weights_dir, "inception_5b_1x1_weights.nnd"), file::convolution),
            wo.create_weights_from_file(join_path(weights_dir, "inception_5b_1x1_bias.nnd"),  file::bias),
        },
        { 0,{ 0, 0 }, 0 },
        { 1,{ 1, 1 }, 1 },
        padding::zero,
        1,
        true
    });

    // 2nd branch

    auto inception_5b_3x3_reduce = convolution::create(
    {
        mem_format,
        {
            inception_5a_output,
            wo.create_weights_from_file(join_path(weights_dir, "inception_5b_3x3_reduce_weights.nnd"), file::convolution),
            wo.create_weights_from_file(join_path(weights_dir, "inception_5b_3x3_reduce_bias.nnd"),  file::bias),
        },
        { 0,{ 0, 0 }, 0 },
        { 1,{ 1, 1 }, 1 },
        padding::zero,
        1,
        true
    });

    auto inception_5b_3x3 = convolution::create(
    {
        mem_format,
        {
            inception_5b_3x3_reduce,
            wo.create_weights_from_file(join_path(weights_dir, "inception_5b_3x3_weights.nnd"), file::convolution),
            wo.create_weights_from_file(join_path(weights_dir, "inception_5b_3x3_bias.nnd"),  file::bias),
        },
        { 0,{ -1, -1 }, 0 },
        { 1,{ 1, 1 }, 1 },
        padding::zero,
        1,
        true
    });

    // 3rd branch
    auto inception_5b_5x5_reduce = convolution::create(
    {
        mem_format,
        {
            inception_5a_output,
            wo.create_weights_from_file(join_path(weights_dir, "inception_5b_5x5_reduce_weights.nnd"), file::convolution),
            wo.create_weights_from_file(join_path(weights_dir, "inception_5b_5x5_reduce_bias.nnd"),  file::bias),
        },
        { 0,{ 0, 0 }, 0 },
        { 1,{ 1, 1 }, 1 },
        padding::zero,
        1,
        true
    });

    auto inception_5b_5x5 = convolution::create(
    {
        mem_format,
        {
            inception_5b_5x5_reduce,
            wo.create_weights_from_file(join_path(weights_dir, "inception_5b_5x5_weights.nnd"), file::convolution),
            wo.create_weights_from_file(join_path(weights_dir, "inception_5b_5x5_bias.nnd"),  file::bias),
        },
        { 0,{ -2, -2 }, 0 },
        { 1,{ 1, 1 }, 1 },
        padding::zero,
        1,
        true
    });

    // 4th branch

    auto inception_5b_pool = pooling::create(
    {
        pooling::mode::max,
        mem_format,
        inception_5a_output,
        { 0,{ -1, -1 },0 }, //padding 
        { 1,{ 1, 1 },1 }, // strd
        { 1,{ 3, 3 },1 }, // kernel
        padding::zero
    });

    auto inception_5b_pool_proj = convolution::create(
    {
        mem_format,
        {
            inception_5b_pool,
            wo.create_weights_from_file(join_path(weights_dir, "inception_5b_pool_proj_weights.nnd"), file::convolution),
            wo.create_weights_from_file(join_path(weights_dir, "inception_5b_pool_proj_bias.nnd"),  file::bias),
        },
        { 0,{ 0, 0 }, 0 },
        { 1,{ 1, 1 }, 1 },
        padding::zero,
        1,
        true
    });

    auto inception_5b_output = depth_concatenate::create(
    {
        mem_format,
        {
            inception_5b_1x1,
            inception_5b_3x3,
            inception_5b_5x5,
            inception_5b_pool_proj
        }
    });

    // ------------------ ENDING PIPE --------------------------
    auto pool5_7x7_s1 = pooling::create(
    {
        pooling::mode::average,
        mem_format,
        inception_5b_output,
        { 1,{ 1, 1 },1 }, // strd
        { 1,{ 7, 7 },1 }, // kernel
        padding::zero
    });




    auto loss3_classifier = fully_connected::create(
    {
        fc_mem_format,
        pool5_7x7_s1,
        wo.create_weights_from_file(join_path(weights_dir, "loss3_classifier_weights.nnd"), file::fully_connected),
        wo.create_weights_from_file(join_path(weights_dir, "loss3_classifier_bias.nnd"),  file::bias),
        true,
        0
    });

    auto softmax = softmax(
        "output",
        loss3_classifier);

    return topology {
        input,
        reordered_input,
        conv1_7x7_s2,
        pool1_3x3_s2,
        pool1_norm1,
        conv2_3x3_reduce,
        conv2_3x3,
        conv2_norm2,
        pool2_3x3_s2,
        inception_3a_1x1,
        inception_3a_3x3_reduce,
        inception_3a_3x3,
        inception_3a_5x5_reduce,
        inception_3a_5x5,
        inception_3a_pool,
        inception_3a_pool_proj,
        inception_3a_output,
        inception_3b_1x1,
        inception_3b_3x3_reduce,
        inception_3b_3x3,
        inception_3b_5x5_reduce,
        inception_3b_5x5,
        inception_3b_pool,
        inception_3b_pool_proj,
        inception_3b_output,
        pool3_3x3_s2,
        inception_4a_1x1,
        inception_4a_3x3_reduce,
        inception_4a_3x3,
        inception_4a_5x5_reduce,
        inception_4a_5x5,
        inception_4a_pool,
        inception_4a_pool_proj,
        inception_4a_output,
        inception_4b_1x1,
        inception_4b_3x3_reduce,
        inception_4b_3x3,
        inception_4b_5x5_reduce,
        inception_4b_5x5,
        inception_4b_pool,
        inception_4b_pool_proj,
        inception_4b_output,
        inception_4c_1x1,
        inception_4c_3x3_reduce,
        inception_4c_3x3,
        inception_4c_5x5_reduce,
        inception_4c_5x5,
        inception_4c_pool,
        inception_4c_pool_proj,
        inception_4c_output,
        inception_4d_1x1,
        inception_4d_3x3_reduce,
        inception_4d_3x3,
        inception_4d_5x5_reduce,
        inception_4d_5x5,
        inception_4d_pool,
        inception_4d_pool_proj,
        inception_4d_output,
        inception_4e_1x1,
        inception_4e_3x3_reduce,
        inception_4e_3x3,
        inception_4e_5x5_reduce,
        inception_4e_5x5,
        inception_4e_pool,
        inception_4e_pool_proj,
        inception_4e_output,
        pool4_3x3_s2,
        inception_5a_1x1,
        inception_5a_3x3_reduce,
        inception_5a_3x3,
        inception_5a_5x5_reduce,
        inception_5a_5x5,
        inception_5a_pool,
        inception_5a_pool_proj,
        inception_5a_output,
        inception_5b_1x1,
        inception_5b_3x3_reduce,
        inception_5b_3x3,
        inception_5b_5x5_reduce,
        inception_5b_5x5,
        inception_5b_pool,
        inception_5b_pool_proj,
        inception_5b_output,
        pool5_7x7_s1,
        loss3_classifier,
        softmax
    };
}
