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
#pragma once
#include "api/engine.hpp"
#include "api/memory.hpp"
#include "primitive_arg.h"
#include "reorder_arg.h"
#include "implementation_map.h"

namespace neural
{
    using is_an_implementation = cldnn::primitive_impl;
    using reorder = cldnn::reorder_arg;
    using memory = cldnn::neural_memory;
    template<typename primitive_kind> using implementation_map = cldnn::implementation_map<primitive_kind>;
    struct engine
    {
        struct type
        {
            static const cldnn::engine_types gpu = cldnn::engine_types::ocl;
        };
    };

}
