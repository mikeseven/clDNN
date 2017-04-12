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
#include "program.hpp"
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

/// @brief Executable network allocated from @ref program.
struct network
{
    /// @brief Allocate network
    /// @param program The program object which contains compiled primitives this network should allocate memory for.
    network(program const& program)
        :_impl(check_status<cldnn_network>("network allocation failed", [&](status_t* status)
                {
                    return cldnn_allocate_network(program.get(), status);
                }))
    {}

    /// @brief Constructs network object from implicitly created program object. This is a shorthand for network(program(engine, topology, options))
    /// @param engine
    /// @param topology
    /// @param options 
    network(const engine& engine, const topology& topology, const build_options& options = build_options())
        :network(program(engine, topology, options))
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

    /// @brief Returns @ref engine by which network was built.
    engine get_engine() const
    {
        return check_status<cldnn_engine>("get network engine failed", [&](status_t* status) { return cldnn_get_network_engine(_impl, status); });
    }

    /// @brief Returns network internal @ref program.
    program get_program() const
    {
        return check_status<cldnn_program>("get network program failed", [&](status_t* status) { return cldnn_get_network_program(_impl, status); });
    }

    /// @brief Provides @ref memory for @ref input_layout primitives defined by user in source @ref topology.
    void set_input_data(const primitive_id& id, const memory& mem) const
    {
        check_status<void>("set network input failed", [&](status_t* status) { cldnn_set_network_input(_impl, id.c_str(), mem.get(), status); });
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
    /// @note User should call set_input_data() for every @ref input_layout defined in source @ref topology
    /// before network execution.
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
