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

#include <gtest/gtest.h>
#include <api/CPP/tensor.hpp>

TEST(tensor_api, order_yxfb)
{
    cldnn::tensor test{ cldnn::format::yxfb, 0, { 1, 2, 3, 4 } };

    //sizes
    EXPECT_EQ(test.batch.size(), size_t(1));
    EXPECT_EQ(test.feature.size(), size_t(1));
    EXPECT_EQ(test.spatial.size(), size_t(2));

    //passed values
    EXPECT_EQ(test.spatial[0], cldnn::tensor::value_type(2));
    EXPECT_EQ(test.spatial[1], cldnn::tensor::value_type(1));
    EXPECT_EQ(test.feature[0], cldnn::tensor::value_type(3));
    EXPECT_EQ(test.batch[0], cldnn::tensor::value_type(4));

    //reverse
    auto sizes = test.sizes();
    EXPECT_EQ(sizes.size(), size_t(4));
    for (size_t i = 0; i < sizes.size(); ++i)
        EXPECT_EQ(sizes[i], cldnn::tensor::value_type(i + 1));
}

TEST(tensor_api, order_byxf)
{
    cldnn::tensor test{ cldnn::format::byxf, 0, { 1, 2, 3, 4 } };

    //sizes
    EXPECT_EQ(test.batch.size(), size_t(1));
    EXPECT_EQ(test.feature.size(), size_t(1));
    EXPECT_EQ(test.spatial.size(), size_t(2));

    //passed values
    EXPECT_EQ(test.spatial[0], cldnn::tensor::value_type(3));
    EXPECT_EQ(test.spatial[1], cldnn::tensor::value_type(2));
    EXPECT_EQ(test.feature[0], cldnn::tensor::value_type(4));
    EXPECT_EQ(test.batch[0], cldnn::tensor::value_type(1));

    //reverse
    auto sizes = test.sizes();
    EXPECT_EQ(sizes.size(), size_t(4));
    for (size_t i = 0; i < sizes.size(); ++i)
        EXPECT_EQ(sizes[i], cldnn::tensor::value_type(i + 1));
}

TEST(tensor_api, order_bfyx)
{
    cldnn::tensor test{ cldnn::format::bfyx, 0, { 1, 2, 3, 4 } };

    //sizes
    EXPECT_EQ(test.batch.size(), size_t(1));
    EXPECT_EQ(test.feature.size(), size_t(1));
    EXPECT_EQ(test.spatial.size(), size_t(2));

    //passed values
    EXPECT_EQ(test.spatial[0], cldnn::tensor::value_type(4));
    EXPECT_EQ(test.spatial[1], cldnn::tensor::value_type(3));
    EXPECT_EQ(test.feature[0], cldnn::tensor::value_type(2));
    EXPECT_EQ(test.batch[0], cldnn::tensor::value_type(1));

    //reverse
    auto sizes = test.sizes();
    EXPECT_EQ(sizes.size(), size_t(4));
    for (size_t i = 0; i < sizes.size(); ++i)
        EXPECT_EQ(sizes[i], cldnn::tensor::value_type(i + 1));
}

TEST(tensor_api, order_fyxb)
{
    cldnn::tensor test{ cldnn::format::fyxb, 0, { 1, 2, 3, 4 } };

    //sizes
    EXPECT_EQ(test.batch.size(), size_t(1));
    EXPECT_EQ(test.feature.size(), size_t(1));
    EXPECT_EQ(test.spatial.size(), size_t(2));

    //passed values
    EXPECT_EQ(test.spatial[0], cldnn::tensor::value_type(3));
    EXPECT_EQ(test.spatial[1], cldnn::tensor::value_type(2));
    EXPECT_EQ(test.feature[0], cldnn::tensor::value_type(1));
    EXPECT_EQ(test.batch[0], cldnn::tensor::value_type(4));

    //reverse
    auto sizes = test.sizes();
    EXPECT_EQ(sizes.size(), size_t(4));
    for (size_t i = 0; i < sizes.size(); ++i)
        EXPECT_EQ(sizes[i], cldnn::tensor::value_type(i + 1));
}

TEST(tensor_api, order_os_iyx_osv16)
{
    cldnn::tensor test{ cldnn::format::os_iyx_osv16, 0, { 1, 2, 3, 4 } };

    //sizes
    EXPECT_EQ(test.batch.size(), size_t(1));
    EXPECT_EQ(test.feature.size(), size_t(2));
    EXPECT_EQ(test.spatial.size(), size_t(2));

    //passed values
    EXPECT_EQ(test.spatial[0], cldnn::tensor::value_type(4));
    EXPECT_EQ(test.spatial[1], cldnn::tensor::value_type(3));
    EXPECT_EQ(test.feature[0], cldnn::tensor::value_type(1));
    EXPECT_EQ(test.feature[1], cldnn::tensor::value_type(2));

    //defaults
    EXPECT_EQ(test.batch[0], cldnn::tensor::value_type(0));

    //reverse
    auto sizes = test.sizes();
    EXPECT_EQ(sizes.size(), size_t(4));
    for (size_t i = 0; i < sizes.size(); ++i)
        EXPECT_EQ(sizes[i], cldnn::tensor::value_type(i + 1));
}

TEST(tensor_api, order_bs_xs_xsv8_bsv8)
{
    cldnn::tensor test{ cldnn::format::bs_xs_xsv8_bsv8, 0, { 1, 2 } };

    //sizes
    EXPECT_EQ(test.batch.size(), size_t(1));
    EXPECT_EQ(test.feature.size(), size_t(1));
    EXPECT_EQ(test.spatial.size(), size_t(1));

    //passed values
    EXPECT_EQ(test.spatial[0], cldnn::tensor::value_type(2));
    EXPECT_EQ(test.batch[0], cldnn::tensor::value_type(1));

    //defaults
    EXPECT_EQ(test.feature[0], cldnn::tensor::value_type(0));

    //reverse
    auto sizes = test.sizes();
    EXPECT_EQ(sizes.size(), size_t(2));
    for (size_t i = 0; i < sizes.size(); ++i)
        EXPECT_EQ(sizes[i], cldnn::tensor::value_type(i + 1));
}

TEST(tensor_api, order_bs_x_bsv16)
{
    cldnn::tensor test{ cldnn::format::bs_x_bsv16, 0,{ 1, 2 } };

    //sizes
    EXPECT_EQ(test.batch.size(), size_t(1));
    EXPECT_EQ(test.feature.size(), size_t(1));
    EXPECT_EQ(test.spatial.size(), size_t(1));

    //passed values
    EXPECT_EQ(test.spatial[0], cldnn::tensor::value_type(2));
    EXPECT_EQ(test.batch[0], cldnn::tensor::value_type(1));

    //defaults
    EXPECT_EQ(test.feature[0], cldnn::tensor::value_type(0));

    //reverse
    auto sizes = test.sizes();
    EXPECT_EQ(sizes.size(), size_t(2));
    for (size_t i = 0; i < sizes.size(); ++i)
        EXPECT_EQ(sizes[i], cldnn::tensor::value_type(i + 1));
}
