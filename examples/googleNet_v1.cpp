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
#include "output_parser.h"
#include <iostream>
#include <string>
#include "api/instrumentation.h"

using namespace neural;

// Building GoogLeNet v1 network with loading weights & biases from file
std::vector<std::pair<primitive, std::string>> build_googlenetv1(const primitive& input, const primitive& output, const std::string& weights_dir, weights_optimizer& wo, bool use_half)
{
    auto mem_format = use_half ? memory::format::yxfb_f16 : memory::format::yxfb_f32;
    auto fc_mem_format = use_half ? memory::format::xb_f16 : memory::format::xb_f32;

    // [224x224x3xB] convolution->relu->pooling->lrn [1000xB]
    std::cout << "Building GoogLeNet started" << std::endl;
    instrumentation::timer<> timer_build;

    // create conversion to yxfb format and subtract mean values
    auto reordered_input = reorder::create(
    {
        mem_format,
        input.as<const memory&>().argument.size,
        input,
        {104,117,123},
        true
    });

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

    auto softmax = normalization::softmax::create(
    {
        output,
        loss3_classifier
    });

    auto build_time = timer_build.uptime();
    std::cout << "Building VGG16 finished in " << instrumentation::to_string(build_time) << std::endl;

    return std::vector<std::pair<primitive, std::string>> {
        { reordered_input, "reorder"},
        { reordered_input,"reordered_input" },
        { conv1_7x7_s2,"conv1_7x7_s2" },
        { pool1_3x3_s2,"pool1_3x3_s2" },
        { pool1_norm1,"pool1_norm1" },
        { conv2_3x3_reduce,"conv2_3x3_reduce" },
        { conv2_3x3,"conv2_3x3" },
        { conv2_norm2,"conv2_norm2" },
        { pool2_3x3_s2,"pool2_3x3_s2" },
        { inception_3a_1x1,"inception_3a_1x1" },
        { inception_3a_3x3_reduce,"inception_3a_3x3_reduce" },
        { inception_3a_3x3,"inception_3a_3x3" },
        { inception_3a_5x5_reduce,"inception_3a_5x5_reduce" },
        { inception_3a_5x5,"inception_3a_5x5" },
        { inception_3a_pool,"inception_3a_pool" },
        { inception_3a_pool_proj,"inception_3a_pool_proj" },
        { inception_3a_output,"inception_3a_output" },
        { inception_3b_1x1,"inception_3b_1x1" },
        { inception_3b_3x3_reduce,"inception_3b_3x3_reduce" },
        { inception_3b_3x3,"inception_3b_3x3" },
        { inception_3b_5x5_reduce,"inception_3b_5x5_reduce" },
        { inception_3b_5x5,"inception_3b_5x5" },
        { inception_3b_pool,"inception_3b_pool" },
        { inception_3b_pool_proj,"inception_3b_pool_proj" },
        { inception_3b_output,"inception_3b_output" },
        { pool3_3x3_s2,"pool3_3x3_s2" },
        { inception_4a_1x1,"inception_4a_1x1" },
        { inception_4a_3x3_reduce,"inception_4a_3x3_reduce" },
        { inception_4a_3x3,"inception_4a_3x3" },
        { inception_4a_5x5_reduce,"inception_4a_5x5_reduce" },
        { inception_4a_5x5,"inception_4a_5x5" },
        { inception_4a_pool,"inception_4a_pool" },
        { inception_4a_pool_proj,"inception_4a_pool_proj" },
        { inception_4a_output,"inception_4a_output" },
        { inception_4b_1x1,"inception_4b_1x1" },
        { inception_4b_3x3_reduce,"inception_4b_3x3_reduce" },
        { inception_4b_3x3,"inception_4b_3x3" },
        { inception_4b_5x5_reduce,"inception_4b_5x5_reduce" },
        { inception_4b_5x5,"inception_4b_5x5" },
        { inception_4b_pool,"inception_4b_pool" },
        { inception_4b_pool_proj,"inception_4b_pool_proj" },
        { inception_4b_output,"inception_4b_output" },
        { inception_4c_1x1,"inception_4c_1x1" },
        { inception_4c_3x3_reduce,"inception_4c_3x3_reduce" },
        { inception_4c_3x3,"inception_4c_3x3" },
        { inception_4c_5x5_reduce,"inception_4c_5x5_reduce" },
        { inception_4c_5x5,"inception_4c_5x5" },
        { inception_4c_pool,"inception_4c_pool" },
        { inception_4c_pool_proj,"inception_4c_pool_proj" },
        { inception_4c_output,"inception_4c_output" },
        { inception_4d_1x1,"inception_4d_1x1" },
        { inception_4d_3x3_reduce,"inception_4d_3x3_reduce" },
        { inception_4d_3x3,"inception_4d_3x3" },
        { inception_4d_5x5_reduce,"inception_4d_5x5_reduce" },
        { inception_4d_5x5,"inception_4d_5x5" },
        { inception_4d_pool,"inception_4d_pool" },
        { inception_4d_pool_proj,"inception_4d_pool_proj" },
        { inception_4d_output,"inception_4d_output" },
        { inception_4e_1x1,"inception_4e_1x1" },
        { inception_4e_3x3_reduce,"inception_4e_3x3_reduce" },
        { inception_4e_3x3,"inception_4e_3x3" },
        { inception_4e_5x5_reduce,"inception_4e_5x5_reduce" },
        { inception_4e_5x5,"inception_4e_5x5" },
        { inception_4e_pool,"inception_4e_pool" },
        { inception_4e_pool_proj,"inception_4e_pool_proj" },
        { inception_4e_output,"inception_4e_output" },
        { pool4_3x3_s2,"pool4_3x3_s2" },
        { inception_5a_1x1,"inception_5a_1x1" },
        { inception_5a_3x3_reduce,"inception_5a_3x3_reduce" },
        { inception_5a_3x3,"inception_5a_3x3" },
        { inception_5a_5x5_reduce,"inception_5a_5x5_reduce" },
        { inception_5a_5x5,"inception_5a_5x5" },
        { inception_5a_pool,"inception_5a_pool" },
        { inception_5a_pool_proj,"inception_5a_pool_proj" },
        { inception_5a_output,"inception_5a_output" },
        { inception_5b_1x1,"inception_5b_1x1" },
        { inception_5b_3x3_reduce,"inception_5b_3x3_reduce" },
        { inception_5b_3x3,"inception_5b_3x3" },
        { inception_5b_5x5_reduce,"inception_5b_5x5_reduce" },
        { inception_5b_5x5,"inception_5b_5x5" },
        { inception_5b_pool,"inception_5b_pool" },
        { inception_5b_pool_proj,"inception_5b_pool_proj" },
        { inception_5b_output,"inception_5b_output" },
        { pool5_7x7_s1,"pool5_7x7_s1" },
        { loss3_classifier,"loss3_classifier" },
        { softmax,"softmax" }
    };
}



void googlenet_v1(uint32_t batch_size, std::string img_dir, const std::string& weights_dir, bool dump_hl, bool profiling, bool optimize_weights, bool use_half)
{
    uint32_t gpu_batch_size = get_gpu_batch_size(batch_size);
    if (gpu_batch_size != batch_size)
    {
        std::cout << "WARNING: This is not the optimal batch size. You have " << (gpu_batch_size - batch_size)
            << " dummy images per batch!!! Please use batch=" << gpu_batch_size << "." << std::endl;
    }
    gpu::configuration::get().enable_profiling = profiling;

    auto img_list = get_directory_images(img_dir);
    if (img_list.empty())
        throw std::runtime_error("specified input images directory is empty (does not contain image data)");

    auto number_of_batches = (img_list.size() % batch_size == 0)
        ? img_list.size() / batch_size : img_list.size() / batch_size + 1;

    html output_file("googlenet_v1", "googlenet_v1 run");

    weights_optimizer weights_optimizer(optimize_weights, use_half);

    auto input = memory::allocate({use_half ? memory::format::byxf_f16 : memory::format::byxf_f32, {gpu_batch_size, {224, 224}, 3}});
    auto output = memory::allocate({use_half ? memory::format::xb_f16 : memory::format::xb_f32, {gpu_batch_size, {1000}}});

    // build googlenet
    std::vector<std::pair<primitive, std::string>> primitives = build_googlenetv1(input, output, weights_dir, weights_optimizer, use_half);

    // create worker
    worker worker = create_worker();

    // optimize weights if needed
    if (optimize_weights)
    {
        weight_optimization(weights_optimizer, worker);
    }

    std::vector<std::string> images_in_batch;
    auto images_list_iterator = img_list.begin();
    auto images_list_end = img_list.end();
    for (decltype(number_of_batches) batch = 0; batch < number_of_batches; batch++)
    {
        images_in_batch.clear();
        for (uint32_t i = 0; i < batch_size && images_list_iterator != images_list_end; ++i, ++images_list_iterator)
        {
            images_in_batch.push_back(*images_list_iterator);
        }

        // load croped and resized images into input
        if (use_half)
        {
            load_images_from_file_list<half_t>(images_in_batch, input);
        }
        else
        {
            load_images_from_file_list(images_in_batch, input);
        }

        // execute Googlenet
        auto time = execute_topology(worker, primitives, output, dump_hl, "GoogLeNet_v1", 86);

        auto time_in_sec = std::chrono::duration_cast<std::chrono::duration<double, std::chrono::seconds::period>>(time).count();
        output_file.batch(output.as<const neural::memory&>(), join_path(get_executable_info()->dir(), "names.txt"), images_in_batch);
        if (time_in_sec != 0.0)
        {
            std::cout << "Frames per second:" << (double)batch_size / time_in_sec << std::endl;
        }
    }
}
