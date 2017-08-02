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
#include "api/CPP/program.hpp"

#include "refcounted_obj.h"
#include "topology_impl.h"
#include "engine_impl.h"
#include "program_node.h"

#include <list>
#include <algorithm>

namespace cldnn
{

struct primitive_impl;
class layout_optimizer;

/*
    cldnn_program implementation
*/
struct program_impl : public refcounted_obj<program_impl>
{
    friend struct program_node;

public:
    program_impl(engine_impl::ptr engine, topology_impl const& topology, build_options const& options);

    auto get_engine() const { return engine; }
    auto get_options() const { return options; }

    std::list<std::shared_ptr<program_node>> get_nodes() const;

    auto& get_node(primitive_id const& id) 
    {
        try 
        {
            return *nodes_map.at(id);
        }
        catch (...)
        {
            throw std::runtime_error("Program doesn't contain primtive node: " + id);
        }
    }
    auto const& get_node(primitive_id const& id) const
    {
        try
        {
            return *nodes_map.at(id);
        }
        catch (...)
        {
            throw std::runtime_error("Program doesn't contain primtive node: " + id);
        }
    }

private:
    engine_impl::ptr engine;
    build_options options;

    std::list<program_node*> inputs;
    std::vector<program_node*> outputs;
    std::list<program_node*> processing_order;

    std::map<primitive_id, std::shared_ptr<program_node>> nodes_map;

    std::list<primitive_id> optimized_out;

    // TODO: Remove once we will get full support for input/output padding in all primitive implementations.
    bool output_size_handling_enabled;

    /*
    ** High-level functions, in order of usage
    */
    void init_graph(topology_impl const& topology);
    void pre_optimize_graph();
    void post_optimize_graph();
    void compile_graph();

    /*
    ** Initialization functions
    */
    void set_outputs();
    void calc_processing_order();
    void mark_constants();
    void mark_data_flow();
    void calc_dominators();

    /*
    ** Analysis functions
    */
    // TODO: Remove once we will get full support for input/output padding in all primitive implementations.
    void analyze_output_size_handling_need();

    /*
    ** Optimization functions
    */
    void trim_to_outputs();
    void reorder_nodes_for_parallel_execution();
    void reorder_inputs(layout_optimizer& lo);
    void pre_optimize_bias(layout_optimizer& lo);
    void post_optimize_weights(layout_optimizer& lo);
    void apply_needed_padding(program_node& node, program_node& prev_node, const padding& needed_padding);
    void prepare_padding();
    void prepare_buffer_fusing();

    /*
    ** Utilities
    */

    //returns already existing program_node for given primitive 'prim' (lookup in 'nodes_map')
    //if it was previously created, otherwise creates and then returns program_node
    program_node& get_or_create(std::shared_ptr<primitive> prim);

    // Inserts given program_node 'node' as an intermediate node between 'next' and it's
    //  dependency at 'prev_idx' index.
    void add_intermediate(program_node& prim, program_node& next, size_t prev_idx);

    // Gets or creates program_node for given primitive 'prim' and inserts it as an intermediate
    // node between 'next' and it's dependency at 'prev_idx' index.
    void add_intermediate(std::shared_ptr<primitive> prim, program_node& next, size_t prev_idx)
    {
        add_intermediate(get_or_create(prim), next, prev_idx);
    }

    void add_connection(program_node& prev, program_node& next)
    {
        prev.users.push_back(&next);
        next.dependencies.push_back(&prev);
    }

    void remove_if_dangling(program_node& node);


    void forward_bfs(std::function<void(program_node&)> const& mark_func = nullptr, std::function<void(program_node&)> const& unmark_func = nullptr) const;
    void backward_bfs(std::function<void(program_node&)> const& mark_func = nullptr, std::function<void(program_node&)> const& unmark_func = nullptr) const;
};
}

API_CAST(::cldnn_program, cldnn::program_impl)
