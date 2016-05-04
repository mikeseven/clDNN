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
#include "pooling_cpu_reference.h"

namespace neural{
    //                                           engine                          output                  input
    using pool_fw_key = std::tuple<neural::engine::type, neural::memory::format::type, neural::memory::format::type>;
    using pool_bw_key = std::tuple<neural::engine::type, neural::memory::format::type, neural::memory::format::type>; //todo

    extern singleton_map<pool_fw_key, std::function<is_an_implementation *(pooling &)>>         & pool_fw_implementation_map;
//    extern singleton_map<pool_bw_key, std::function<is_an_implementation *(pooling_backward &)>>& pool_bw_implementation_map;
}
