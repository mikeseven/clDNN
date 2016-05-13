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

#include "api/neural.h"

namespace neural {

execution_resource_cpu::arguments::arguments(uint64_t arg_threadpool_size)
    : threadpool_size(arg_threadpool_size) {}

execution_resource_cpu::arguments::arguments()
    : threadpool_size(0) {}

// creates primitive with memry buffer loaded from specified file
execution_resource execution_resource_cpu::create(execution_resource_cpu::arguments arg) {
    try {
        // crete result with output owning memory, load data into it
        auto result = std::unique_ptr<execution_resource_cpu>(new execution_resource_cpu(arg));

        return result.release();
    }
    catch(...) {
        throw std::runtime_error("file::create: error loading file");
    }
}

}