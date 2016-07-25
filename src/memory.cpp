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
#include <atomic>

namespace neural {

memory::arguments::arguments(neural::engine::type aengine, memory::format::type aformat, vector<uint32_t> asize)
    : engine(aengine)
    , format(aformat)
    , size(asize) {}

size_t memory::count() const {
    return _buffer->size() / traits(argument.format).type->size;
}

size_t memory::size_of(arguments arg) {
    return std::accumulate(
        arg.size.raw.begin(),
        arg.size.raw.end(),
        memory::traits(arg.format).type->size,
        std::multiplies<size_t>()
    );
}

std::shared_ptr<memory::buffer> create_buffer(memory::arguments arg, bool allocate) {
    auto key = arg.engine;
    auto it = allocators_map::instance().find(key);
    if (it == std::end(allocators_map::instance())) throw std::runtime_error("Memory allocator is not yet implemented.");

    return it->second(arg, allocate);
}

primitive memory::describe(memory::arguments arg){
    auto buffer = create_buffer(arg, false);
    return new memory(arg, buffer);
}

primitive memory::allocate(memory::arguments arg){
    auto buffer = create_buffer(arg, true);
    return new memory(arg, buffer);
}

namespace {
    // simple CPU memory buffer
    struct cpu_buffer : public memory::buffer {
        explicit cpu_buffer(size_t size, bool allocate = true) : _size(size),
                                                                 _pointer(nullptr) {
            if (allocate) {
                not_my_pointer.clear();
                _pointer = new char[_size];
            }
            else
                not_my_pointer.test_and_set();
        }

        ~cpu_buffer() override {
            clear();
        }

        void* lock() override {
            return _pointer;
        }

        void release() override {}

        void reset(void* ptr) override {
            clear();
            _pointer = static_cast<char *>(ptr);
        }

        size_t size() override { return _size; }

    private:
        void clear() {
            if (!not_my_pointer.test_and_set())
                delete[] _pointer;
        }
        std::atomic_flag not_my_pointer;
        size_t _size;
        char* _pointer;
    };

    struct attach {
        attach() {
            auto create_buff = [](memory::arguments arg, bool allocate) -> std::shared_ptr<memory::buffer> {
                return std::make_shared<cpu_buffer>(memory::size_of(arg), allocate);
            };
            allocators_map::instance().insert({ engine::reference, create_buff });
            allocators_map::instance().insert({ engine::cpu, create_buff });
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