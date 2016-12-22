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
#include <cstdint>
#include "cldnn_defs.h"
#include "compounds.h"
#include "memory.hpp"
#include "topology.hpp"
#include "event.hpp"
#include <algorithm>
#include <map>

namespace cldnn
{

enum class build_option_type
{
    fusing, profiling, optimize_data, debug, outputs
};

struct build_option_ref
{
    build_option_type type;
    const void* data;
};

inline bool test_option_bool(build_option_type type)
{
    switch (type)
    {
    case build_option_type::fusing:
    case build_option_type::profiling:
    case build_option_type::optimize_data:
    case build_option_type::debug:
        return true;
    default:
        return false;
    }
}

struct build_option
{
    static const build_option_type fusing = build_option_type::fusing;
    static const build_option_type profiling = build_option_type::profiling;
    static const build_option_type optimize_data = build_option_type::optimize_data;
    static const build_option_type debug = build_option_type::debug;
    static const build_option* outputs(const std::vector<primitive_id>& outs);
    static const build_option* outputs(array_ref<primitive_id_ref> outs);
    virtual ~build_option() = default;

protected:
    build_option(const build_option_ref& value) : _value(value) {}
private:
    build_option(const build_option& other) = delete;
    build_option& operator=(const build_option& other) = delete;
    const build_option_ref _value;
    friend class build_options;
};

struct build_option_outputs : build_option
{
    typedef const array_ref<primitive_id_ref>* data_pointer_type;
    const std::vector<primitive_id> outputs;

    explicit build_option_outputs(const std::vector<primitive_id>& outs)
        : build_option({ build_option_type::outputs, &_outputs_ref })
        , outputs(outs)
        , _outputs_ref_store(outputs.size())
        , _outputs_ref(_outputs_ref_store)
    {
        std::copy(std::begin(outputs), std::end(outputs), std::begin(_outputs_ref_store));
    }

    explicit build_option_outputs(const build_option_ref& value)
        : build_option_outputs(make_outputs_from_ref(value))
    {}

    build_option_outputs(const build_option_outputs& other) = delete;
    build_option_outputs& operator=(const build_option_outputs& other) = delete;

private:
    std::vector<primitive_id_ref> _outputs_ref_store;
    array_ref<primitive_id_ref> _outputs_ref;

    static std::vector<primitive_id> make_outputs_from_ref(const build_option_ref& value)
    {
        if (value.type != build_option_type::outputs) throw std::invalid_argument("option type does not match: should be 'output'");
        if (value.data == nullptr) throw std::invalid_argument("output data is empty");
        auto refs = reinterpret_cast<data_pointer_type>(value.data);
        std::vector<primitive_id> result(refs->size());
        std::copy(std::begin(*refs), std::end(*refs), std::begin(result));
        return result;
    }
};

template<build_option_type OptType>
struct build_option_traits;

template<> struct build_option_traits<build_option_type::fusing>        { typedef build_option object_type; };
template<> struct build_option_traits<build_option_type::profiling>     { typedef build_option object_type; };
template<> struct build_option_traits<build_option_type::optimize_data> { typedef build_option object_type; };
template<> struct build_option_traits<build_option_type::debug>         { typedef build_option object_type; };
template<> struct build_option_traits<build_option_type::outputs>       { typedef build_option_outputs object_type; };

inline const build_option* build_option::outputs(const std::vector<primitive_id>& outs)
{
    return new build_option_outputs(outs);
}

inline const build_option* build_option::outputs(array_ref<primitive_id_ref> outs)
{
    auto data = static_cast<build_option_outputs::data_pointer_type>(&outs);
    return new build_option_outputs({ build_option_type::outputs, data });
}


class build_options
{
public:
    void set_option(const build_option* opt)
    {
        add_or_replace_option(opt);
    }

    void set_option(build_option_type type)
    {
        assert(test_option_bool(type));
        add_or_replace_option(new build_option({ type, nullptr }));
    }

    template<typename ...Args>
    void set_option(const build_option* opt, Args... args)
    {
        add_or_replace_option(opt);
        set_option(args...);
    }

    template<typename ...Args>
    void set_option(build_option_type type, Args... args)
    {
        assert(test_option_bool(type));
        add_or_replace_option(new build_option({ type, nullptr }));
        set_option(args...);
    }

    template<typename ...Args>
    build_options(Args... args)
    {
        set_option(args...);
    }

    build_options(array_ref<build_option_ref> options)
    {
        for(auto& o: options)
        {
            _options.emplace_back(make_option(o));
        }
    }

    build_options(const build_options& other)
        : _options(other._options)
    {
    }

    build_options& operator=(const build_options& other)
    {
        if (this == &other)
            return *this;
        _options = other._options;
        return *this;
    }

    std::vector<build_option_ref> get_refs() const
    {
        std::vector<build_option_ref> result;
        for (auto& o : _options)
        {
            result.push_back(o->_value);
        }
        return result;
    }

    const build_option* get(build_option_type type)
    {
        for(auto& option: _options)
        {
            if (option->_value.type == type)
                return option.get();
        }
        return nullptr;
    }

    template<build_option_type OptType, class T = typename build_option_traits<OptType>::object_type>
    const T* get()
    {
        return static_cast<const T*>(get(OptType));
    }

private:
    std::vector<std::shared_ptr<const build_option>> _options;

    void add_or_replace_option(const build_option* opt)
    {
        for(auto& p : _options)
        {
            if(p->_value.type == opt->_value.type)
            {
                p.reset(opt);
                return;
            }
        }
        _options.emplace_back(opt);
    }

    static const build_option* make_option(const build_option_ref& option)
    {
        switch (option.type)
        {
        case build_option_type::fusing:
        case build_option_type::profiling:
        case build_option_type::optimize_data:
        case build_option_type::debug:
            return new build_option(option);
        case build_option_type::outputs:
            return new build_option_outputs(option);
        default: throw std::out_of_range("unsupported build option type");
        }
    }
};

struct network_output
{
    primitive_id id() const { return _id; }
    event get_event() const
    {
        return _event;
    }
    memory get_memory() const
    {
        _event.wait();
        return _result;
    }
private:
    primitive_id _id;
    event _event;
    memory _result;
    network_output(primitive_id id, event evt, memory mem): _id(id), _event(evt), _result(mem){}
    friend struct network;
};

class network_impl;
struct network
{
    struct network_output_ref
    {
        primitive_id_ref id_ref;
        event_impl* event_impl;
        memory_impl* memory_impl;
    };

    static network build(const engine& engine, const topology& topology, const build_options& options )
    {
        return check_status<network_impl*>("network build failed", [&](status_t* status) { return build_impl(engine, topology, options.get_refs(), status); });
    }

    typedef network_impl impl_type;
    DLL_SYM network(const network& other);
    DLL_SYM network& operator=(const network& other);
    DLL_SYM ~network();
    friend bool operator==(const network& lhs, const network& rhs) { return lhs._impl == rhs._impl; }
    friend bool operator!=(const network& lhs, const network& rhs) { return !(lhs == rhs); }

    engine get_engine() const
    {
        return check_status<engine_impl*>("get network engine failed", [&](status_t* status) { return get_engine_impl(status); });
    }

    topology get_topology() const
    {
        return check_status<topology_impl*>("get network topology failed", [&](status_t* status) { return get_topology_impl(status); });
    }

    DLL_SYM void set_input_data(primitive_id id, memory mem)
    {
        status_t status = set_input_data_impl(id, mem);
        if (status != CLDNN_SUCCESS)
            CLDNN_THROW("set data input failed", status);
    }

    std::vector<network_output> execute(const std::vector<event>& dependencies)
    {
        array_ref<network_output_ref> result_ref = check_status<array_ref<network_output_ref>>("network execute failed", [&](status_t* status) { return execute_impl(dependencies, status); });
        std::vector<network_output> result;
        for(auto& ref : result_ref)
        {
            result.push_back({ ref.id_ref, event(ref.event_impl), memory(ref.memory_impl) });
        }
        return result;
    }

    network_impl* get() const { return _impl; }

private:
    friend struct engine;
    network(network_impl* impl) :_impl(impl) {}
    network_impl* _impl;
    DLL_SYM static network_impl* build_impl(const engine& engine, const topology& topology, array_ref<build_option_ref> options, status_t* status) noexcept;
    DLL_SYM status_t set_input_data_impl(primitive_id_ref id, memory mem) noexcept;
    DLL_SYM array_ref<network_output_ref> execute_impl(array_ref<event> dependencies, status_t* status) noexcept;
    DLL_SYM engine_impl* get_engine_impl(status_t* status) const noexcept;
    DLL_SYM topology_impl* get_topology_impl(status_t* status) const noexcept;
};
API_CLASS(network)
}
