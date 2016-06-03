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
#include "api/neural.h"
#include "memory.h"
#include "ocl_toolkit.h"

namespace {

    void* allocate_memory(size_t) {
        throw std::runtime_error("not implemented");
    }
    void deallocate_memory(void*, size_t) {
//        throw std::runtime_error("not implemented");
    }

    struct attach {
        attach() {
            neural::memory_allocator default_allocator{
                [](size_t size) { return new char[size]; },
                [](void* pointer, size_t) { delete[] static_cast<char *>(pointer); }
            };

            neural::allocators_map::instance().insert({neural::engine::reference, default_allocator });
            neural::allocators_map::instance().insert({neural::engine::cpu, default_allocator });
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
