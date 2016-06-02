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

#include "memory.h"

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
    if (!argument.owns_memory) return;

    auto key = argument.engine;
    auto it = allocators_map::instance().find(key);
    if (it == std::end(allocators_map::instance())) return;

    it->second.deallocate(pointer, count()*memory::traits(argument.format).type->size);
}

primitive memory::describe(memory::arguments arg){
    return new memory(arg);
}

primitive memory::allocate(memory::arguments arg){
    auto key = arg.engine;
    auto it = allocators_map::instance().find(key);
    if(it == std::end(allocators_map::instance())) throw std::runtime_error("Memory allocator is not yet implemented.");

    auto result = std::unique_ptr<memory>(new memory(arg));
    result->pointer = it->second.allocate(result->count()*memory::traits(arg.format).type->size);
//    result->pointer = new char[result->count()*memory::traits(arg.format).type->size];
    const_cast<memory::arguments &>(result->argument).owns_memory = true;
    return result.release();
}

namespace {
    struct attach {
        attach() {
            memory_allocator default_allocator {
                [](size_t size) { return new char[size]; },
                [](void* pointer, size_t) { delete[] static_cast<char *>(pointer); }
            };

            allocators_map::instance().insert({ engine::reference, default_allocator });
            allocators_map::instance().insert({ engine::cpu, default_allocator });
        }
        ~attach() {}
    };

#ifdef __GNUC__
    __attribute__((visibility("default"))) //todo meybe dll_sym?
#elif _MSC_VER
#   pragma section(".nn_init$m", read, write)
#endif
    attach attach_impl;

}

} // namespace neural