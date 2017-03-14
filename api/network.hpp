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
#include "cldnn_defs.h"
#include "compounds.h"
#include "memory.hpp"
#include "topology.hpp"
#include "event.hpp"

#include <cstdint>
#include <algorithm>
#include <map>

namespace cldnn
{
enum class build_option_type : int32_t
{
    fusing        = cldnn_build_option_fusing,
    profiling     = cldnn_build_option_profiling,
    optimize_data = cldnn_build_option_optimize_data,
    debug         = cldnn_build_option_debug,
    outputs       = cldnn_build_option_outputs
};

struct build_option
{
    static const build_option_type fusing        = build_option_type::fusing;
    static const build_option_type profiling     = build_option_type::profiling;
    static const build_option_type optimize_data = build_option_type::optimize_data;
    static const build_option_type debug         = build_option_type::debug;
    static const build_option* outputs(const std::vector<primitive_id>& outs);
    virtual ~build_option() = default;

protected:
    build_option(const cldnn_build_option& value) : _value(value) {}
private:
    build_option(const build_option& other) = delete;
    build_option& operator=(const build_option& other) = delete;
    const cldnn_build_option _value;
    friend class build_options;

    static bool is_option_bool(build_option_type type)
    {
        switch (type)
        {
        case fusing:
        case profiling:
        case optimize_data:
        case debug:
            return true;
        default:
            return false;
        }
    }
};

struct build_option_outputs : build_option
{
    typedef const cldnn_primitive_id_arr* data_pointer_type;
    const std::vector<primitive_id> outputs;

    explicit build_option_outputs(const std::vector<primitive_id>& outs)
        : build_option({ cldnn_build_option_outputs, &_outputs_ref })
        , outputs(outs)
        , _ref_store(to_refs(outputs))
        , _outputs_ref({ _ref_store.data(), _ref_store.size()})
    {}

    explicit build_option_outputs(const cldnn_build_option& value)
        : build_option_outputs(make_outputs_from_ref(value))
    {}

    build_option_outputs(const build_option_outputs& other) = delete;
    build_option_outputs& operator=(const build_option_outputs& other) = delete;

private:
    const std::vector<cldnn_primitive_id> _ref_store;
    const cldnn_primitive_id_arr _outputs_ref;

    static std::vector<cldnn_primitive_id> to_refs(const std::vector<primitive_id>& stor)
    {
        std::vector<cldnn_primitive_id> result(stor.size());
        for (size_t i = 0; i < stor.size(); i++)
        {
            result[i] = stor[i].c_str();
        }
        return std::move(result);
    }

    static std::vector<primitive_id> make_outputs_from_ref(const cldnn_build_option& value)
    {
        if (value.type != cldnn_build_option_outputs) throw std::invalid_argument("option type does not match: should be 'output'");
        if (value.data == nullptr) throw std::invalid_argument("output data is empty");
        auto refs = reinterpret_cast<data_pointer_type>(value.data);
        std::vector<primitive_id> result;
        result.reserve(refs->size);
        for(decltype(refs->size) i = 0; i < refs->size; i++)
        {
            result.push_back(refs->data[i]);
        }
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

class build_options
{
public:
    void set_option(void){}

    void set_option(const build_option* opt)
    {
        add_or_replace_option(opt);
    }

    void set_option(build_option_type type)
    {
        assert(build_option::is_option_bool(type));
        add_or_replace_option(new build_option({ static_cast<cldnn_build_option_type>(type), nullptr }));
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
        assert(build_option::is_option_bool(type));
        add_or_replace_option(new build_option({ static_cast<cldnn_build_option_type>(type), nullptr }));
        set_option(args...);
    }

    template<typename ...Args>
    build_options(Args... args)
    {
        set_option(args...);
    }

    build_options(array_ref<cldnn_build_option> options)
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

    std::vector<cldnn_build_option> get_refs() const
    {
        std::vector<cldnn_build_option> result;
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
            if (option->_value.type == static_cast<cldnn_build_option_type>(type))
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

    static const build_option* make_option(const cldnn_build_option& option)
    {
        switch (option.type)
        {
        case cldnn_build_option_fusing:
        case cldnn_build_option_profiling:
        case cldnn_build_option_optimize_data:
        case cldnn_build_option_debug:
            return new build_option(option);
        case cldnn_build_option_outputs:
            return new build_option_outputs(option);
        default: throw std::out_of_range("unsupported build option type");
        }
    }
};

struct network_output
{
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
    event _event;
    memory _result;
    network_output(event evt, memory mem): _event(evt), _result(mem){}
    network_output(cldnn_event evt, cldnn_memory mem): _event(evt), _result(mem){}
    friend struct network;
};

struct network
{
    network(const engine& engine, const topology& topology, const build_options& options = build_options())
        :_impl(check_status<cldnn_network>("network build failed", [&](status_t* status)
                {
                    auto options_refs = options.get_refs();
                    return cldnn_build_network(engine.get(), topology.get(), options_refs.data(), options_refs.size(), status);
                }))
    {}

    network(cldnn_network impl) :_impl(impl)
    {
        if (_impl == nullptr) throw std::invalid_argument("implementation pointer should not be null");
    }

    network(const network& other) :_impl(other._impl)
    {
        retain();
    }
    network& operator=(const network& other)
    {
        if (_impl == other._impl) return *this;
        release();
        _impl = other._impl;
        retain();
        return *this;
    }
    ~network()
    {
        release();
    }

    friend bool operator==(const network& lhs, const network& rhs) { return lhs._impl == rhs._impl; }
    friend bool operator!=(const network& lhs, const network& rhs) { return !(lhs == rhs); }

    engine get_engine() const
    {
        return check_status<cldnn_engine>("get network engine failed", [&](status_t* status) { return cldnn_get_network_engine(_impl, status); });
    }

    topology get_topology() const
    {
        return check_status<cldnn_topology>("get network topology failed", [&](status_t* status) { return cldnn_get_network_topology(_impl, status); });
    }

    void set_input_data(const primitive_id& id, const memory& mem) const
    {
        check_status<void>("set network input failed", [&](status_t* status) {cldnn_set_network_input(_impl, id.c_str(), mem.get(), status); });
    }

    std::vector<primitive_id> get_output_ids() const
    {
        size_t size_ret = 0;
        status_t err_invalid_arg = CLDNN_SUCCESS;
        cldnn_get_network_output_names(_impl, nullptr, 0, &size_ret, &err_invalid_arg);
        assert(err_invalid_arg == CLDNN_INVALID_ARG);
        assert(size_ret > 0);
        std::vector<char> names_buf(size_ret);
        
        check_status<void>("get network output ids failed", [&](status_t* status)
        {
            cldnn_get_network_output_names(_impl, names_buf.data(), names_buf.size(), &size_ret, status);
        });
        assert(names_buf.size() == size_ret);

        std::vector<primitive_id> result;
        for(auto buf_ptr = names_buf.data(); *buf_ptr != 0; buf_ptr += result.back().size() + 1)
        {
            result.emplace_back(buf_ptr);
        }
        return result;
    }

    network_output get_output(const primitive_id& output_id) const
    {
        cldnn_network_output output =
        check_status<cldnn_network_output>("get network output failed", [&](status_t* status)
        {
            return cldnn_get_network_output(_impl, output_id.c_str(), status);
        });
        return network_output( output.event, output.memory );
    }

    std::map<primitive_id, network_output> execute(const std::vector<event>& dependencies = {}) const
    {
        std::vector<cldnn_event> dep_refs(dependencies.size());
        for(decltype(dependencies.size()) i = 0; i < dependencies.size(); i++)
        {
            dep_refs[i] = dependencies[i].get();
        }

        check_status<void>("network execute failed", [&](status_t* status)
        {
            return cldnn_execute_network(_impl, dep_refs.data(), dep_refs.size(), status);
        });

        auto output_ids = get_output_ids();
        std::map<primitive_id, network_output> result;
        for(auto& id : output_ids)
        {
            result.emplace(id, get_output(id));
        }
        return result;
    }

    cldnn_network get() const { return _impl; }

private:
    cldnn_network _impl;

    void retain()
    {
        check_status<void>("retain topology failed", [=](status_t* status) { cldnn_retain_network(_impl, status); });
    }
    void release()
    {
        check_status<void>("retain topology failed", [=](status_t* status) { cldnn_release_network(_impl, status); });
    }
};
CLDNN_API_CLASS(network)
}
