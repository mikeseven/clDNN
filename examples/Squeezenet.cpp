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
#include <api/primitives/reorder.hpp>
#include <api/primitives/convolution.hpp>
#include <api/primitives/pooling.hpp>
#include <api/primitives/depth_concatenate.hpp>
#include <api/primitives/softmax.hpp>

using namespace cldnn;

topology build_squeezenet(const std::string& weights_dir, weights_optimizer& wo, cldnn::layout& input_layout, int32_t batch_size)
{
    // [227x227x3xB] convolution->relu->pooling->lrn [1000xB]
    input_layout.size = { format::byxf,{ batch_size, 227, 227, 3 } };
    auto input = cldnn::input_layout("input", input_layout);

    // TODO: remove after enabling bfyx for all
    auto mem_format = format::yxfb;
    if (batch_size == 1 && input_layout.data_type == data_types::f32)
    {
        mem_format = format::bfyx;
    }

    // create conversion to yxfb format and subtract mean values
    tensor reorder_size = input_layout.size.transform(mem_format, 1);
    //auto reorder_mean = { (float)104.0069879317889, (float)116.66876761696767, (float)122.6789143406786 };
    auto reordered_input = reorder(
        "reorder",
        input,
        { input_layout.data_type, reorder_size },
        std::vector<float>{ (float)104.0069879317889, (float)116.66876761696767, (float)122.6789143406786 });

    auto conv1_weights = wo.create_weights_from_file(join_path(weights_dir, "conv1_weights.nnd"), file::convolution);
    auto conv1_biases = wo.create_weights_from_file(join_path(weights_dir, "conv1_bias.nnd"), file::bias);
    auto conv1 = convolution(
        "conv1",
        reordered_input,
        { conv1_weights },
        { conv1_biases },
        { format::yx,{ 0,0 } },
        { format::yx,{ 2,2 } },
        true);

    auto pool1 = pooling(
        "pool1",
        conv1,
        pooling_mode::max,
        { format::yx,{ 2,2 } }, // strd
        { format::yx,{ 3,3 } }); // kernel


    auto fire2_squeeze1x1_weights = wo.create_weights_from_file(join_path(weights_dir, "fire2_squeeze1x1_weights.nnd"), file::convolution);
    auto fire2_squeeze1x1_biases = wo.create_weights_from_file(join_path(weights_dir, "fire2_squeeze1x1_bias.nnd"), file::bias);
    auto fire2_squeeze1x1 = convolution(
        "fire2_squeeze1x1",
        pool1,
        { fire2_squeeze1x1_weights },
        { fire2_squeeze1x1_biases },
        { format::yx,{ 0,0 } },
        { format::yx,{ 1,1 } },
        true);

    auto fire2_expand1x1_weights = wo.create_weights_from_file(join_path(weights_dir, "fire2_expand1x1_weights.nnd"), file::convolution);
    auto fire2_expand1x1_biases = wo.create_weights_from_file(join_path(weights_dir, "fire2_expand1x1_bias.nnd"), file::bias);
    auto fire2_expand1x1 = convolution(
        "fire2_expand1x1",
        fire2_squeeze1x1,
        { fire2_expand1x1_weights },
        { fire2_expand1x1_biases },
        { format::yx,{ 0,0 } },
        { format::yx,{ 1,1 } },
        true);

    auto fire2_expand3x3_weights = wo.create_weights_from_file(join_path(weights_dir, "fire2_expand3x3_weights.nnd"), file::convolution);
    auto fire2_expand3x3_biases = wo.create_weights_from_file(join_path(weights_dir, "fire2_expand3x3_bias.nnd"), file::bias);
    auto fire2_expand3x3 = convolution(
        "fire2_expand3x3",
        fire2_squeeze1x1,
        { fire2_expand3x3_weights },
        { fire2_expand3x3_biases },
        { format::yx,{ -1,-1 } },
        { format::yx,{ 1,1 } },
        true);


    auto fire2_concat = depth_concatenate(   
        "fire2_concat",
        {
            fire2_expand1x1,
            fire2_expand3x3
        }
    );


    auto fire3_squeeze1x1_weights = wo.create_weights_from_file(join_path(weights_dir, "fire3_squeeze1x1_weights.nnd"), file::convolution);
    auto fire3_squeeze1x1_biases = wo.create_weights_from_file(join_path(weights_dir, "fire3_squeeze1x1_bias.nnd"), file::bias);
    auto fire3_squeeze1x1 = convolution(
        "fire3_squeeze1x1",
        fire2_concat,
        { fire3_squeeze1x1_weights },
        { fire3_squeeze1x1_biases },
        { format::yx,{ 0,0 } },
        { format::yx,{ 1,1 } },
        true);

    auto fire3_expand1x1_weights = wo.create_weights_from_file(join_path(weights_dir, "fire3_expand1x1_weights.nnd"), file::convolution);
    auto fire3_expand1x1_biases = wo.create_weights_from_file(join_path(weights_dir, "fire3_expand1x1_bias.nnd"), file::bias);
    auto fire3_expand1x1 = convolution(
        "fire3_expand1x1",
        fire3_squeeze1x1,
        { fire3_expand1x1_weights },
        { fire3_expand1x1_biases },
        { format::yx,{ 0,0 } },
        { format::yx,{ 1,1 } },
        true);

    auto fire3_expand3x3_weights = wo.create_weights_from_file(join_path(weights_dir, "fire3_expand3x3_weights.nnd"), file::convolution);
    auto fire3_expand3x3_biases = wo.create_weights_from_file(join_path(weights_dir, "fire3_expand3x3_bias.nnd"), file::bias);
    auto fire3_expand3x3 = convolution(
        "fire3_expand3x3",
        fire3_squeeze1x1,
        { fire3_expand3x3_weights },
        { fire3_expand3x3_biases },
        { format::yx,{ -1,-1 } },
        { format::yx,{ 1,1 } },
        true);

    auto fire3_concat = depth_concatenate(
        "fire3_concat",
        {
            fire3_expand1x1,
            fire3_expand3x3
        }
    );

    auto pool3 = pooling(
        "pool3",
        fire3_concat,
        pooling_mode::max,
        { format::yx,{ 2,2 } }, // strd
        { format::yx,{ 3,3 } }); // kernel

    auto fire4_squeeze1x1_weights = wo.create_weights_from_file(join_path(weights_dir, "fire4_squeeze1x1_weights.nnd"), file::convolution);
    auto fire4_squeeze1x1_biases = wo.create_weights_from_file(join_path(weights_dir, "fire4_squeeze1x1_bias.nnd"), file::bias);
    auto fire4_squeeze1x1 = convolution(
        "fire4_squeeze1x1",
        pool3,
        { fire4_squeeze1x1_weights },
        { fire4_squeeze1x1_biases },
        { format::yx,{ 0,0 } },
        { format::yx,{ 1,1 } },
        true);

    auto fire4_expand1x1_weights = wo.create_weights_from_file(join_path(weights_dir, "fire4_expand1x1_weights.nnd"), file::convolution);
    auto fire4_expand1x1_biases = wo.create_weights_from_file(join_path(weights_dir, "fire4_expand1x1_bias.nnd"), file::bias);
    auto fire4_expand1x1 = convolution(
        "fire4_expand1x1",
        fire4_squeeze1x1,
        { fire4_expand1x1_weights },
        { fire4_expand1x1_biases },
        { format::yx,{ 0,0 } },
        { format::yx,{ 1,1 } },
        true);

    auto fire4_expand3x3_weights = wo.create_weights_from_file(join_path(weights_dir, "fire4_expand3x3_weights.nnd"), file::convolution);
    auto fire4_expand3x3_biases = wo.create_weights_from_file(join_path(weights_dir, "fire4_expand3x3_bias.nnd"), file::bias);
    auto fire4_expand3x3 = convolution(
        "fire4_expand3x3",
        fire4_squeeze1x1,
        { fire4_expand3x3_weights },
        { fire4_expand3x3_biases },
        { format::yx,{ -1,-1 } },
        { format::yx,{ 1,1 } },
        true);

    auto fire4_concat = depth_concatenate(
        "fire4_concat",
        {
            fire4_expand1x1,
            fire4_expand3x3
        }
    );

    auto fire5_squeeze1x1_weights = wo.create_weights_from_file(join_path(weights_dir, "fire5_squeeze1x1_weights.nnd"), file::convolution);
    auto fire5_squeeze1x1_biases = wo.create_weights_from_file(join_path(weights_dir, "fire5_squeeze1x1_bias.nnd"), file::bias);
    auto fire5_squeeze1x1 = convolution(
        "fire5_squeeze1x1",
        fire4_concat,
        { fire5_squeeze1x1_weights },
        { fire5_squeeze1x1_biases },
        { format::yx,{ 0,0 } },
        { format::yx,{ 1,1 } },
        true);

    auto fire5_expand1x1_weights = wo.create_weights_from_file(join_path(weights_dir, "fire5_expand1x1_weights.nnd"), file::convolution);
    auto fire5_expand1x1_biases = wo.create_weights_from_file(join_path(weights_dir, "fire5_expand1x1_bias.nnd"), file::bias);
    auto fire5_expand1x1 = convolution(
        "fire5_expand1x1",
        fire5_squeeze1x1,
        { fire5_expand1x1_weights },
        { fire5_expand1x1_biases },
        { format::yx,{ 0,0 } },
        { format::yx,{ 1,1 } },
        true);

    auto fire5_expand3x3_weights = wo.create_weights_from_file(join_path(weights_dir, "fire5_expand3x3_weights.nnd"), file::convolution);
    auto fire5_expand3x3_biases = wo.create_weights_from_file(join_path(weights_dir, "fire5_expand3x3_bias.nnd"), file::bias);
    auto fire5_expand3x3 = convolution(
        "fire5_expand3x3",
        fire5_squeeze1x1,
        { fire5_expand3x3_weights },
        { fire5_expand3x3_biases },
        { format::yx,{ -1,-1 } },
        { format::yx,{ 1,1 } },
        true);

    auto fire5_concat = depth_concatenate(
        "fire5_concat",
        {
            fire5_expand1x1,
            fire5_expand3x3
        }
    );

    auto pool5 = pooling(
        "pool5",
        fire5_concat,
        pooling_mode::max,
        { format::yx,{ 2,2 } }, // strd
        { format::yx,{ 3,3 } }); // kernel

    auto fire6_squeeze1x1_weights = wo.create_weights_from_file(join_path(weights_dir, "fire6_squeeze1x1_weights.nnd"), file::convolution);
    auto fire6_squeeze1x1_biases = wo.create_weights_from_file(join_path(weights_dir, "fire6_squeeze1x1_bias.nnd"), file::bias);
    auto fire6_squeeze1x1 = convolution(
        "fire6_squeeze1x1",
        pool5,
        { fire6_squeeze1x1_weights },
        { fire6_squeeze1x1_biases },
        { format::yx,{ 0,0 } },
        { format::yx,{ 1,1 } },
        true);

    auto fire6_expand1x1_weights = wo.create_weights_from_file(join_path(weights_dir, "fire6_expand1x1_weights.nnd"), file::convolution);
    auto fire6_expand1x1_biases = wo.create_weights_from_file(join_path(weights_dir, "fire6_expand1x1_bias.nnd"), file::bias);
    auto fire6_expand1x1 = convolution(
        "fire6_expand1x1",
        fire6_squeeze1x1,
        { fire6_expand1x1_weights },
        { fire6_expand1x1_biases },
        { format::yx,{ 0,0 } },
        { format::yx,{ 1,1 } },
        true);

    auto fire6_expand3x3_weights = wo.create_weights_from_file(join_path(weights_dir, "fire6_expand3x3_weights.nnd"), file::convolution);
    auto fire6_expand3x3_biases = wo.create_weights_from_file(join_path(weights_dir, "fire6_expand3x3_bias.nnd"), file::bias);
    auto fire6_expand3x3 = convolution(
        "fire6_expand3x3",
        fire6_squeeze1x1,
        { fire6_expand3x3_weights },
        { fire6_expand3x3_biases },
        { format::yx,{ -1,-1 } },
        { format::yx,{ 1,1 } },
        true);

    auto fire6_concat = depth_concatenate( 
        "fire6_concat",
        {
            fire6_expand1x1,
            fire6_expand3x3
        }
    );

    auto fire7_squeeze1x1_weights = wo.create_weights_from_file(join_path(weights_dir, "fire7_squeeze1x1_weights.nnd"), file::convolution);
    auto fire7_squeeze1x1_biases = wo.create_weights_from_file(join_path(weights_dir, "fire7_squeeze1x1_bias.nnd"), file::bias);
    auto fire7_squeeze1x1 = convolution(
        "fire7_squeeze1x1",
        fire6_concat,
        { fire7_squeeze1x1_weights },
        { fire7_squeeze1x1_biases },
        { format::yx,{ 0,0 } },
        { format::yx,{ 1,1 } },
        true);

    auto fire7_expand1x1_weights = wo.create_weights_from_file(join_path(weights_dir, "fire7_expand1x1_weights.nnd"), file::convolution);
    auto fire7_expand1x1_biases = wo.create_weights_from_file(join_path(weights_dir, "fire7_expand1x1_bias.nnd"), file::bias);
    auto fire7_expand1x1 = convolution(
        "fire7_expand1x1",
        fire7_squeeze1x1,
        { fire7_expand1x1_weights },
        { fire7_expand1x1_biases },
        { format::yx,{ 0,0 } },
        { format::yx,{ 1,1 } },
        true);

    auto fire7_expand3x3_weights = wo.create_weights_from_file(join_path(weights_dir, "fire7_expand3x3_weights.nnd"), file::convolution);
    auto fire7_expand3x3_biases = wo.create_weights_from_file(join_path(weights_dir, "fire7_expand3x3_bias.nnd"), file::bias);
    auto fire7_expand3x3 = convolution(
        "fire7_expand3x3",
        fire7_squeeze1x1,
        { fire7_expand3x3_weights },
        { fire7_expand3x3_biases },
        { format::yx,{ -1,-1 } },
        { format::yx,{ 1,1 } },
        true);

    auto fire7_concat = depth_concatenate(
        "fire7_concat",
        {
            fire7_expand1x1,
            fire7_expand3x3
        }
    );

    auto fire8_squeeze1x1_weights = wo.create_weights_from_file(join_path(weights_dir, "fire8_squeeze1x1_weights.nnd"), file::convolution);
    auto fire8_squeeze1x1_biases = wo.create_weights_from_file(join_path(weights_dir, "fire8_squeeze1x1_bias.nnd"), file::bias);
    auto fire8_squeeze1x1 = convolution(
        "fire8_squeeze1x1",
        fire7_concat,
        { fire8_squeeze1x1_weights },
        { fire8_squeeze1x1_biases },
        { format::yx,{ 0,0 } },
        { format::yx,{ 1,1 } },
        true);

    auto fire8_expand1x1_weights = wo.create_weights_from_file(join_path(weights_dir, "fire8_expand1x1_weights.nnd"), file::convolution);
    auto fire8_expand1x1_biases = wo.create_weights_from_file(join_path(weights_dir, "fire8_expand1x1_bias.nnd"), file::bias);
    auto fire8_expand1x1 = convolution(
        "fire8_expand1x1",
        fire8_squeeze1x1,
        { fire8_expand1x1_weights },
        { fire8_expand1x1_biases },
        { format::yx,{ 0,0 } },
        { format::yx,{ 1,1 } },
        true);

    auto fire8_expand3x3_weights = wo.create_weights_from_file(join_path(weights_dir, "fire8_expand3x3_weights.nnd"), file::convolution);
    auto fire8_expand3x3_biases = wo.create_weights_from_file(join_path(weights_dir, "fire8_expand3x3_bias.nnd"), file::bias);
    auto fire8_expand3x3 = convolution(
        "fire8_expand3x3",
        fire8_squeeze1x1,
        { fire8_expand3x3_weights },
        { fire8_expand3x3_biases },
        { format::yx,{ -1,-1 } },
        { format::yx,{ 1,1 } },
        true);

    auto fire8_concat = depth_concatenate(
    
        "fire8_concat",
        {
            fire8_expand1x1,
            fire8_expand3x3
        }
    );

    auto fire9_squeeze1x1_weights = wo.create_weights_from_file(join_path(weights_dir, "fire9_squeeze1x1_weights.nnd"), file::convolution);
    auto fire9_squeeze1x1_biases = wo.create_weights_from_file(join_path(weights_dir, "fire9_squeeze1x1_bias.nnd"), file::bias);
    auto fire9_squeeze1x1 = convolution(
        "fire9_squeeze1x1",
        fire8_concat,
        { fire9_squeeze1x1_weights },
        { fire9_squeeze1x1_biases },
        { format::yx,{ 0,0 } },
        { format::yx,{ 1,1 } },
        true);

    auto fire9_expand1x1_weights = wo.create_weights_from_file(join_path(weights_dir, "fire9_expand1x1_weights.nnd"), file::convolution);
    auto fire9_expand1x1_biases = wo.create_weights_from_file(join_path(weights_dir, "fire9_expand1x1_bias.nnd"), file::bias);
    auto fire9_expand1x1 = convolution(
        "fire9_expand1x1",
        fire9_squeeze1x1,
        { fire9_expand1x1_weights },
        { fire9_expand1x1_biases },
        { format::yx,{ 0,0 } },
        { format::yx,{ 1,1 } },
        true);

    auto fire9_expand3x3_weights = wo.create_weights_from_file(join_path(weights_dir, "fire9_expand3x3_weights.nnd"), file::convolution);
    auto fire9_expand3x3_biases = wo.create_weights_from_file(join_path(weights_dir, "fire9_expand3x3_bias.nnd"), file::bias);
    auto fire9_expand3x3 = convolution(
        "fire9_expand3x3",
        fire9_squeeze1x1,
        { fire9_expand3x3_weights },
        { fire9_expand3x3_biases },
        { format::yx,{ -1,-1 } },
        { format::yx,{ 1,1 } },
        true);

    auto fire9_concat = depth_concatenate(
    
        "fire9_concat",
        {
            fire9_expand1x1,
            fire9_expand3x3
        }
    );

    auto conv10_weights = wo.create_weights_from_file(join_path(weights_dir, "conv10_weights.nnd"), file::convolution);
    auto conv10_biases = wo.create_weights_from_file(join_path(weights_dir, "conv10_bias.nnd"), file::bias);
    auto conv10 = convolution(
        "conv10",
        fire9_concat,
        { conv10_weights },
        { conv10_biases },
        { format::yx,{ 0,0 } },
        { format::yx,{ 1,1 } },
        true);

    auto pool10 = pooling(
        "pool10",
        conv10,
        pooling_mode::average,
        { format::yx,{ 1,1 } }, // strd
        { format::yx,{ 14,14 } }); // kernel

    auto softmax = cldnn::softmax(
        "output",
        pool10);

    cldnn::topology topology(
        input,
        reordered_input,
        conv1,
        pool1,
        fire2_squeeze1x1);
    
    topology.add(fire2_expand1x1);
    topology.add(fire2_expand3x3);
        
    topology.add(
        fire2_concat);
    topology.add(
        fire3_squeeze1x1,
        fire3_expand1x1,
        fire3_expand3x3,
        fire3_concat,
        pool3);
     topology.add(
        fire4_squeeze1x1,
        fire4_expand1x1,
        fire4_expand3x3,
        fire4_concat,
        fire5_squeeze1x1,
        fire5_expand1x1,
        fire5_expand3x3,
        fire5_concat,
        pool5);
    topology.add(
        fire6_squeeze1x1,
        fire6_expand1x1,
        fire6_expand3x3,
        fire6_concat,
        fire7_squeeze1x1,
        fire7_expand1x1,
        fire7_expand3x3,
        fire7_concat,
        fire8_squeeze1x1,
        fire8_expand1x1,
        fire8_expand3x3,
        fire8_concat,
        fire9_squeeze1x1,
        fire9_expand1x1,
        fire9_expand3x3,
        fire9_concat,
        conv10,
        pool10,
        softmax
    );
    return topology;
}
