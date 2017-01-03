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

///////////////////////////////////////////////////////////////////////////////////////////////////


#include <gtest/gtest.h>
#include "api/engine.hpp"

using namespace cldnn;

TEST(gpu_engine, engine_info)
{
    engine engine;
    auto info = engine.get_info();
    //EXPECT_GE(info.model, engine_info::models::HSW);
    //EXPECT_GE(info.architecture, engine_info::architectures::GEN7);
    EXPECT_GE(info.configuration, engine_info::configurations::GT0);
    EXPECT_GT(info.cores_count, 0u);
    EXPECT_GT(info.core_frequency, 0u);
}