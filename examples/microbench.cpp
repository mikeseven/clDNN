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

using namespace neural;

primitive test_conv(
    uint32_t batch_size,
    uint32_t in_w,
    uint32_t in_h,
    uint32_t in_feature,
    neural::vector<int32_t> offset,
    neural::vector<uint32_t> stride,
    uint32_t out_feature,
    uint32_t kernel_w,
    uint32_t kernel_h,
    bool use_half)
{
    auto input = memory::allocate({ use_half ? memory::format::yxfb_f16 : memory::format::yxfb_f32,{ batch_size,{ in_w, in_h }, in_feature } });
    auto weight = memory::allocate({ use_half ? memory::format::yxio_f16 : memory::format::yxio_f32,{ 1,{ kernel_w, kernel_h } ,{ out_feature, in_feature } } });
    auto bias = memory::allocate({ use_half ? memory::format::x_f16 : memory::format::x_f32,{ 1,{ { out_feature } } , 1 } });

    return convolution::create(
    {
        use_half ? memory::format::yxfb_f16 : memory::format::yxfb_f32,
        {
            input,
            weight,
            bias
        },
        offset,
        stride,
        padding::zero,
        1,
        true
    });
}

std::vector<std::pair<primitive, std::string>> build_microbench(const std::string& weights_dir, weights_optimizer& wo, uint32_t batch_size, bool use_half)
{
    auto conv1 = test_conv(
        batch_size,
        424, 712, 3,
        { 0, { -1, -1 }, 0 },
        { 1, { 2, 2 }, 1 },
        32,
        3, 3,
        use_half
    );

    auto conv2 = test_conv(
        batch_size,
        212, 356, 32,
        { 0,{ -1, -1 }, 0 },
        { 1,{ 1, 1 }, 1 },
        32,
        3, 3,
        use_half
    );

    auto conv3 = test_conv(
        batch_size,
        212, 356, 32,
        { 0,{ -1, -1 }, 0 },
        { 1,{ 1, 1 }, 1 },
        64,
        3, 3,
        use_half
    );

    auto conv4 = test_conv(
        batch_size,
        106, 178, 64,
        { 0,{ 0, 0 }, 0 },
        { 1,{ 1, 1 }, 1 },
        48,
        1, 1,
        use_half
    );

    return std::vector<std::pair<primitive, std::string>> {
        { conv1, "conv1"},
        { conv2, "conv2"},
        { conv3, "conv3"},
        { conv4, "conv4"}
    };
}
