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

#include <string>

namespace neural 
{
    extern const char convolution_code_yxfb_memory[];
    extern const char convolution_code_yxfb_yxoi_memory[];
    extern const char convolution_code_yxfb_oyxi_memory[];
    extern const char convolution_code_yxfb_yxoi_b8_memory[];
    extern const char convolution_code_yxfb_yxoi_B8_F8_memory[];
    extern const char convolution_code_yxfb_yxio_b1_memory[];
    extern const char convolution_code_yxfb_yxio_b1_vload_memory[];
    extern const char convolution_code_yxfb_yxio_b1_block_memory[];
    extern const char convolution_code_yxfb_yxio_b8_memory[];
    extern const char convolution_code_yxfb_yxio_b16_memory[];
}