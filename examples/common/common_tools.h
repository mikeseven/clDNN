// Copyright (c) 2016-2018 Intel Corporation
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


#pragma once

#include "power_instrumentation.h"
#include "instrumentation.h"

// TODO: [FUTURE] The common tools should contain only inclusion of other utilities headers.
#include "executable_utils.h"
#include "file_system_utils.h"

#include "api/CPP/topology.hpp"
#include "api/CPP/network.hpp"
#include "api/CPP/profiling.hpp"
#include "api/CPP/memory.hpp"

#include <chrono>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

using cldnn_output = std::map<cldnn::primitive_id, cldnn::network_output>;
namespace cldnn
{
namespace utils
{
namespace examples
{

/// @brief Time duration represented as floating-point (double precision) time in seconds (alias).
using fp_seconds_type = std::chrono::duration<double, std::chrono::seconds::period>;

enum class print_type
{
    verbose,
    performance,
    extended_testing,

    // ReSharper disable once CppInconsistentNaming
    _enum_count
};

} // namespace examples
} // namespace utils
} // namespace cldnn

// --------------------------------------------------------------------------------------------------------------------
// 
// --------------------------------------------------------------------------------------------------------------------

/// LSTM execution params
struct lstm_execution_params {
    uint32_t lstm_input_size;
    uint32_t lstm_hidden_size;
    uint32_t lstm_sequence_len;
    uint32_t lstm_batch_size;
    bool     lstm_no_biases;
    bool     lstm_initial_hidden;
    bool     lstm_initial_cell;
};

/// Information about executed topology
struct execution_params {
    std::string input_dir;
    std::string weights_dir;
    std::string topology_name;

    uint32_t batch;
    bool meaningful_kernels_names;
    bool profiling;
    bool optimize_weights;
    bool use_half;
    bool use_oooq;
    std::string run_until_primitive_name;
    std::string run_single_kernel_name;

    // for dumping
    bool        dump_hidden_layers; // dump all
    std::string dump_layer_name;    // dump only this specific layer
    bool        dump_weights;       // true when dumping weights: <primitive_name>.nnd
    bool        dump_single_batch;
    uint32_t    dump_batch_id;
    bool        dump_single_feature;
    uint32_t    dump_feature_id;
    bool        dump_graphs;
    bool        log_engine;
    bool        dump_sources;
    std::string serialization;
    std::string load_program;
    cldnn::utils::examples::print_type print_type; // printing modes - to support Verbose, vs performance ony, vs ImageNet testing prints
    size_t loop;            // running the same input in a loop for smoothing perf results

    bool perf_per_watt; // power instrumentation
    bool disable_mem_pool; // memory 
    bool calibration; // int8 precission
    lstm_execution_params lstm_ep; // LSTM microbench parameters

    //training
    uint32_t image_number;
    uint32_t image_offset;
    uint32_t epoch_number;
    bool use_existing_weights;
    bool compute_imagemean;
    float learning_rate;
    uint32_t train_snapshot;
    std::string image_set;
};

struct memory_filler
{
    typedef enum
    {
        zero = 0,
        one,
        zero_to_nine
    } filler_type;

    template<typename T>
    static void fill_memory(const cldnn::memory& memory, filler_type fill)
    {
        auto mem_ptr = memory.pointer<T>();
        float val = (fill == filler_type::zero) ? 0.0f : 1.0f;
        for (auto& it : mem_ptr)
        {
            if (fill == zero_to_nine)
                val += fmod(val + 1.0f, 10.0f);
            it = T(val);
        }
    }

    template<typename Fp, typename T>
    static void fill_memory(const cldnn::memory& memory, std::vector<T> values)
    {
        auto mem_ptr = memory.pointer<Fp>();
        auto it = mem_ptr.begin();

        for (auto x : values)
        {
            *it++ = x;
        }
    }
};

void compute_image_mean(const execution_params &ep, cldnn::engine& engine, bool use_cifar10);

/// function moved from alexnet.cpp, they will be probably used by each topology
void print_profiling_table(std::ostream& os, const std::vector<cldnn::instrumentation::profiling_info>& profiling_info);
cldnn::network build_network(const cldnn::engine& engine, const cldnn::topology& topology, const execution_params &ep, const std::vector<cldnn::primitive_id> &output_ids = std::vector<cldnn::primitive_id>(0));
uint32_t get_next_nearest_power_of_two(int number);
uint32_t get_gpu_batch_size(int number);

bool do_log_energy(const execution_params&, CIntelPowerGadgetLib&);

void make_instrumentations(const execution_params&, cldnn::memory&, std::map<cldnn::primitive_id, cldnn::network_output>&);

std::chrono::nanoseconds get_execution_time(cldnn::instrumentation::timer<>& timer_execution,
                                            const execution_params &ep,
                                            cldnn::memory& output,
                                            cldnn_output& outputs,
                                            bool log_energy,
                                            CIntelPowerGadgetLib& energyLib,
                                            const uint32_t iteration = 0,
                                            const uint32_t execution_count = 0);

std::chrono::nanoseconds execute_cnn_topology(cldnn::network network,
                                            const execution_params &ep,
                                            CIntelPowerGadgetLib& energyLib,
                                            cldnn::memory& output,
                                            const uint32_t iteration = 0,
                                            const uint32_t execution_count = 0);

void run_topology(const execution_params &ep);

std::string get_model_name(const std::string& topology_name);