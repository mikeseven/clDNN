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

#include <functional>
#include <numeric>

namespace neural {

memory::arguments::arguments(neural::engine::type aengine, memory::format::type aformat, vector<uint32_t> asize)
    : engine(aengine)
    , format(aformat)
    , size(asize)
    , owns_memory(false) {}

size_t memory::count() const {
    return std::accumulate(argument.size.raw.begin(), argument.size.raw.end(), size_t(1), std::multiplies<size_t>());
}

memory::~memory() {
    if(argument.owns_memory) delete[] static_cast<char *>(pointer);
}

primitive memory::describe(memory::arguments arg){
    return new memory(arg);
}

primitive memory::allocate(memory::arguments arg){
    auto result = std::unique_ptr<memory>(new memory(arg));
    result->pointer = new char[result->count()*memory::traits(arg.format).type->size];
    const_cast<memory::arguments &>(result->argument).owns_memory = true;
    return result.release();
}

} // namespace neural