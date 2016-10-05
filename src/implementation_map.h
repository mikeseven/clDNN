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
#include <map>
#include <functional>

template<typename T, typename U>
class singleton_map : public std::map<T, U> {
    singleton_map() : std::map<T, U>() {};
    singleton_map(singleton_map const&) = delete;
    void operator=(singleton_map const&) = delete;

    public:
    static singleton_map &instance() {
        static singleton_map instance_;
        return instance_;
    }
};

namespace neural {

template<typename primitive_kind>
struct implementation_key 
{
    typedef std::tuple<neural::engine::type, neural::memory::format::type, neural::memory::format::type> type;
    type operator()(primitive_kind& primitive) 
	{
        return std::make_tuple(neural::engine::gpu, primitive.input_memory(0).argument.format, primitive.output_memory(0).argument.format);
    }
};

template<>
struct implementation_key<reorder>
{
	typedef neural::engine::type type;
	type operator()(reorder& primitive)
	{
		(void)primitive;
		return neural::engine::gpu;
	}
};

template<typename primitive_kind>
class implementation_map {
public:
    using key_builder = implementation_key<primitive_kind>;
    using key_type = typename key_builder::type;
    using factory_type = std::function<is_an_implementation *(primitive_kind &)>;
    using map_type = singleton_map<key_type, factory_type>;

    static factory_type get(primitive_kind& primitive) {
        // lookup in database; throw if not found 
        auto key = key_builder()(primitive);
        auto it = map_type::instance().find(key);
        if (it == std::end(map_type::instance())) throw std::runtime_error("not yet implemented");

        // create implementation & attach it to result 
        return it->second;
    }

    static void add(typename map_type::key_type key, factory_type factory) {
        map_type::instance().insert({ key, factory });
    }

    static void add(std::initializer_list<typename map_type::value_type> il) {
        map_type::instance().insert(il);
    }
};

template <class primitive_kind>
is_a_primitive* is_a_primitive::create(typename primitive_kind::arguments arg) 
{
    std::unique_ptr<primitive_kind> result(new primitive_kind(arg));

     auto factory = implementation_map<primitive_kind>::get(*result);
     auto implementation = factory(*result);
     result->_impl.reset(implementation);
     result->_work = implementation->work();

    return result.release();
}

}
