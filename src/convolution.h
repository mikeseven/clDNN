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

#include "api/neural.h"
#include "implementation_map.h"
#include "convolution_cpu_jit.h"
#include "convolution_cpu_reference.h"

namespace neural{
    //                                           engine                          output                  input
    using conv_fw_key = std::tuple<neural::engine::type, neural::memory_obsolete::format::type, neural::memory_obsolete::format::type>;
    using conv_bw_key = std::tuple<neural::engine::type, neural::memory_obsolete::format::type, neural::memory_obsolete::format::type>; //todo

    extern singleton_map<conv_fw_key, std::function<is_an_implementation *(convolution &)>>         & conv_fw_implementation_map;
    extern singleton_map<conv_bw_key, std::function<is_an_implementation *(convolution_backward &)>>& conv_bw_implementation_map;
}
