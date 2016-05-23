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

namespace neural {
    //                                           engine                        output                         input
    using fully_connected_fw_key = std::tuple<neural::engine::type, neural::memory::format::type, neural::memory::format::type>;;

    using fully_connected_fw_implementation_map = singleton_map<fully_connected_fw_key, std::function<is_an_implementation *(fully_connected &)>>;
}
#pragma once
