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

/// @addtogroup cpp_api C++ API
/// @{

/// @defgroup cpp_network Network Execution
/// @{

/// @brief Represents user-provided network build option.
struct build_option
{   
    /// @brief Allow primitives fusing during network build (default: false).
    static std::shared_ptr<const build_option> fusing(bool enable = false);
    /// @brief Enable primitives profiling (default: false).
    /// @details This option allows to collect @ref profiling_interval for every network output.
    /// This option reduces performance.
    static std::shared_ptr<const build_option> profiling(bool enable = false);
    /// @brief Enable implicit reordering for user inputs (default: false).
    static std::shared_ptr<const build_option> optimize_data(bool enable = false);
    /// @brief Enable debug mode (default: false).
    /// @details This option enforce all network primitives to be accessible as outputs.
    static std::shared_ptr<const build_option> debug(bool enable = false);

    /// @brief User selected list of network outputs.
    static std::shared_ptr<const build_option> outputs(const std::vector<primitive_id>& outs);

    virtual ~build_option() = default;

private:
    /// @brief Returns option type represented by this object.
    virtual cldnn_build_option_type get_type() const = 0;
    /// @brief Returns option @ref ::cldnn_build_option::data represented by this object.
    virtual const void* get_data() const = 0;

    friend class build_options;
};

/// @brief @ref build_option specialization for boolean options.
template<cldnn_build_option_type OptType>
struct build_option_bool : build_option
{
    /// @brief Constructs option.
    /// @param value Is option enabled.
    explicit build_option_bool(bool value) : _value(value) {}

    /// @brief Constructs from C API @ref ::cldnn_build_option.
    explicit build_option_bool(const cldnn_build_option& value)
        : _value(*reinterpret_cast<const bool*>(value.data))
    {
        assert(value.type == static_cast<int32_t>(OptType));
    }

    /// @brief Is option enabled.
    bool enabled() const { return _value; }
private:
    cldnn_build_option_type get_type() const override { return OptType; }
    const void* get_data() const override { return &_value; }
    bool _value;
};

/// @brief @ref build_option specialization for network outputs list.
struct build_option_outputs : build_option
{
    /// @brief The list of output ids (names)
    const std::vector<primitive_id> outputs;

    /// @brief Constructs option.
    /// @param outs List of ouput ids (names)
    explicit build_option_outputs(const std::vector<primitive_id>& outs)
        : outputs(outs)
        , _ref_store(to_refs(outputs))
        , _outputs_ref({ _ref_store.data(), _ref_store.size() })
    {}

    /// @brief Constructs from C API @ref ::cldnn_build_option.
    explicit build_option_outputs(const cldnn_build_option& value)
        : build_option_outputs(make_outputs_from_ref(value))
    {
        assert(value.type == static_cast<int32_t>(cldnn_build_option_outputs));
    }

private:
    /// @brief Returns cldnn_build_option_outputs.
    cldnn_build_option_type get_type() const override { return cldnn_build_option_outputs; }
    /// @brief Returns pointer to @ref cldnn_primitive_is_arr
    const void* get_data() const override { return &_outputs_ref; }

    build_option_outputs(const build_option_outputs& other) = delete;
    build_option_outputs& operator=(const build_option_outputs& other) = delete;

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
        auto refs = reinterpret_cast<const cldnn_primitive_id_arr*>(value.data);
        std::vector<primitive_id> result;
        result.reserve(refs->size);
        for (decltype(refs->size) i = 0; i < refs->size; i++)
        {
            result.push_back(refs->data[i]);
        }
        return result;
    }
};

namespace detail
{
/// @brief Helper template to convert @ref build_option_type value to particular @ref build_option class.
template<cldnn_build_option_type OptType>
struct build_option_traits
{
    /// @brief @ref build_option object type which represents the particular @p OptType.
    typedef build_option object_type;
    /// @brief Make default @ref build_option corresponding @p OptType
    static std::shared_ptr<const build_option> make_default();
    /// @brief Make @ref build_option from C API @ref ::cldnn_build_option
    static std::shared_ptr<const build_option> make_option(const cldnn_build_option& option);
};

#ifndef DOXYGEN_SHOULD_SKIP_THIS
template<> struct build_option_traits<cldnn_build_option_fusing>
{
    typedef build_option_bool<cldnn_build_option_fusing> object_type;
    static std::shared_ptr<const build_option> make_default() { return build_option::fusing(); }
    static std::shared_ptr<const build_option> make_option(const cldnn_build_option& option)
    {
        assert(option.type == cldnn_build_option_fusing);
        return std::make_shared<build_option_bool<cldnn_build_option_fusing>>(option);
    }
};
template<> struct build_option_traits<cldnn_build_option_profiling>
{
    typedef build_option_bool<cldnn_build_option_profiling> object_type;
    static std::shared_ptr<const build_option> make_default() { return build_option::profiling(); }
    static std::shared_ptr<const build_option> make_option(const cldnn_build_option& option)
    {
        assert(option.type == cldnn_build_option_profiling);
        return std::make_shared<build_option_bool<cldnn_build_option_profiling>>(option);
    }
};
template<> struct build_option_traits<cldnn_build_option_optimize_data>
{
    typedef build_option_bool<cldnn_build_option_optimize_data> object_type;
    static std::shared_ptr<const build_option> make_default() { return build_option::optimize_data(); }
    static std::shared_ptr<const build_option> make_option(const cldnn_build_option& option)
    {
        assert(option.type == cldnn_build_option_optimize_data);
        return std::make_shared<build_option_bool<cldnn_build_option_optimize_data>>(option);
    }
};
template<> struct build_option_traits<cldnn_build_option_debug>
{
    typedef build_option_bool<cldnn_build_option_debug> object_type;
    static std::shared_ptr<const build_option> make_default() { return build_option::debug(); }
    static std::shared_ptr<const build_option> make_option(const cldnn_build_option& option)
    {
        assert(option.type == cldnn_build_option_debug);
        return std::make_shared<build_option_bool<cldnn_build_option_debug>>(option);
    }
};
template<> struct build_option_traits<cldnn_build_option_outputs>
{
    typedef build_option_outputs object_type;
    static std::shared_ptr<const build_option> make_default() { return build_option::outputs({}); }
    static std::shared_ptr<const build_option> make_option(const cldnn_build_option& option)
    {
        assert(option.type == cldnn_build_option_outputs);
        return std::make_shared<build_option_outputs>(option);
    }
};
} // namespace detail
#endif

#ifndef DOXYGEN_SHOULD_SKIP_THIS
inline std::shared_ptr<const build_option> build_option::fusing(bool enable)
{
    return std::make_shared<build_option_bool<cldnn_build_option_fusing>>(enable);
}

inline std::shared_ptr<const build_option> build_option::profiling(bool enable)
{
    return std::make_shared<build_option_bool<cldnn_build_option_profiling>>(enable);
}

inline std::shared_ptr<const build_option> build_option::optimize_data(bool enable)
{
    return std::make_shared<build_option_bool<cldnn_build_option_optimize_data>>(enable);
}

inline std::shared_ptr<const build_option> build_option::debug(bool enable)
{
    return std::make_shared<build_option_bool<cldnn_build_option_debug>>(enable);
}

inline std::shared_ptr<const build_option> build_option::outputs(const std::vector<primitive_id>& outs)
{
    return std::make_shared<build_option_outputs>(outs);
}
#endif

/// @brief Represents network build options list.
class build_options
{
public:
    /// @brief Adds or replace option to the options list
    void set_option(std::shared_ptr<const build_option> opt)
    {
        add_or_replace_option(opt);
    }

    /// @brief Adds or replace options to the options list
    template<typename ...Args>
    void set_option(std::shared_ptr<const build_option> opt, Args... args)
    {
        add_or_replace_option(opt);
        set_option(args...);
    }

    /// @brief Constructs build options list from its arguments.
    template<typename ...Args>
    build_options(Args... args)
    {
        set_option(args...);
    }

    /// @brief Constructs build options list from C API ::cldnn_build_options.
    build_options(array_ref<cldnn_build_option> options)
    {
        for(auto& o: options)
        {
            _options.emplace_back(make_option(o));
        }
    }

    /// @brief Copy constructor.
    build_options(const build_options& other)
        : _options(other._options)
    {}

    /// @brief Copy assignment.
    build_options& operator=(const build_options& other)
    {
        if (this == &other)
            return *this;
        _options = other._options;
        return *this;
    }

    /// @brief Returns network build option for @p OptType
    template<cldnn_build_option_type OptType, class T = typename detail::build_option_traits<OptType>::object_type>
    std::shared_ptr<const T> get()
    {
        for (auto& option : _options)
        {
            if (option->get_type() == OptType)
                return std::static_pointer_cast<const T>(option);
        }
        return std::static_pointer_cast<const T>(detail::build_option_traits<OptType>::make_default());
    }

private:
    friend struct network;
    std::vector<std::shared_ptr<const build_option>> _options;
    void set_option(void) {}

    /// @brief Returns C API compatible list of ::cldnn_build_option
    std::vector<cldnn_build_option> get_refs() const
    {
        std::vector<cldnn_build_option> result;
        for (auto& o : _options)
        {
            result.push_back({ o->get_type(), o->get_data() });
        }
        return result;
    }

    void add_or_replace_option(std::shared_ptr<const build_option> opt)
    {
        for(auto& p : _options)
        {
            if(p->get_type() == opt->get_type())
            {
                p = opt;
                return;
            }
        }
        _options.push_back(opt);
    }

    static std::shared_ptr<const build_option> make_option(const cldnn_build_option& option)
    {
        switch (option.type)
        {
        case cldnn_build_option_fusing:
            return  detail::build_option_traits<cldnn_build_option_fusing>::make_option(option);
        case cldnn_build_option_profiling:
            return detail::build_option_traits<cldnn_build_option_profiling>::make_option(option);
        case cldnn_build_option_optimize_data:
            return detail::build_option_traits<cldnn_build_option_optimize_data>::make_option(option);
        case cldnn_build_option_debug:
            return detail::build_option_traits<cldnn_build_option_debug>::make_option(option);
        case cldnn_build_option_outputs:
            return detail::build_option_traits<cldnn_build_option_outputs>::make_option(option);
        default: throw std::out_of_range("unsupported build option type");
        }
    }
};

/// @brief Represents network output returned by @ref network::get_output().
struct network_output
{
    /// @brief Returns @ref event associated with the output.
    event get_event() const { return _event; }

    /// @brief Returns @ref memory object of the output. Blocked until associated @ref event is not complete.
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

/// @brief Executable network built from @ref topology by @ref engine.
struct network
{
    /// @brief Build network
    /// @param engine Engine to be used to compile @p topology into @network.
    /// @param topology Network topology definition.
    /// @param options Network build options. See @ref build_option and @ref build_options for details.
    network(const engine& engine, const topology& topology, const build_options& options = build_options())
        :_impl(check_status<cldnn_network>("network build failed", [&](status_t* status)
                {
                    auto options_refs = options.get_refs();
                    return cldnn_build_network(engine.get(), topology.get(), options_refs.data(), options_refs.size(), status);
                }))
    {}

    /// @brief Constructs network object from C API @ref cldnn_network.
    network(cldnn_network impl) :_impl(impl)
    {
        if (_impl == nullptr) throw std::invalid_argument("implementation pointer should not be null");
    }

    /// @brief Copy construction.
    network(const network& other) :_impl(other._impl)
    {
        retain();
    }

    /// @brief Copy assignment.
    network& operator=(const network& other)
    {
        if (_impl == other._impl) return *this;
        release();
        _impl = other._impl;
        retain();
        return *this;
    }

    /// @brief Releases wrapped C API @ref cldnn_network.
    ~network()
    {
        release();
    }

    friend bool operator==(const network& lhs, const network& rhs) { return lhs._impl == rhs._impl; }
    friend bool operator!=(const network& lhs, const network& rhs) { return !(lhs == rhs); }

    /// @brief Returns @ref engine by which network waas built.
    engine get_engine() const
    {
        return check_status<cldnn_engine>("get network engine failed", [&](status_t* status) { return cldnn_get_network_engine(_impl, status); });
    }

    /// @brief Returns network internal @ref topology. Can be differ that the @p topology passed to construction.
    topology get_topology() const
    {
        return check_status<cldnn_topology>("get network topology failed", [&](status_t* status) { return cldnn_get_network_topology(_impl, status); });
    }

    /// @brief Provides @ref memory for @ref input_layout primitives defined by user in source @ref topology.
    void set_input_data(const primitive_id& id, const memory& mem) const
    {
        check_status<void>("set network input failed", [&](status_t* status) {cldnn_set_network_input(_impl, id.c_str(), mem.get(), status); });
    }

    /// @brief Returns the list of available network outputs.
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

    /// @brief Returns @ref network_output object for particular @p output
    network_output get_output(const primitive_id& output_id) const
    {
        cldnn_network_output output =
        check_status<cldnn_network_output>("get network output failed", [&](status_t* status)
        {
            return cldnn_get_network_output(_impl, output_id.c_str(), status);
        });
        return network_output( output.event, output.memory );
    }

    /// @brief Executes network and returns the list of @ref network_output.
    /// @param dependencies List of @ref event objects to be waited before network execution.
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

    /// @brief Returns wrapped C API @ref cldnn_network handler.
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
/// @}
/// @}
}
