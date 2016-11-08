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
#pragma once
#include <cstdint>

namespace neural { namespace gpu {

class gpu_toolkit;
struct engine_info
{
    enum models
    {
        HSW, BDW, BXT, CHV, SKL, CNL, ICL, GLV, KBL, GLK
    };

    enum architectures
    {
        GEN7, GEN7_5, GEN8, GEN9, GEN10, GEN11, GEN12
    };

    enum configurations
    {
        GT0 = 0,
        GT1,
        GT2,
        GT3,
        GT4
    };

    models model;
    architectures architecture;
    configurations configuration;
    uint32_t cores_count;
    uint32_t core_frequency;

    // Flags (for layout compatibility fixed size types are used).
    uint8_t supports_fp16;
    uint8_t supports_fp16_denorms;
private:
    friend class gpu_toolkit;
    explicit engine_info(gpu_toolkit& context);
};

}}
