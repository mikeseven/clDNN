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

#include <functional>

namespace neural {

    struct memory_allocator {
        std::function<void*(size_t)> allocate;
        std::function<void(void*, size_t)> deallocate;
    };

    using allocators_map = singleton_map<engine::type, memory_allocator>;
}