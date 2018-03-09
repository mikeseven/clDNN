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

#include "power_instrumentation.h"
#include "instrumentation.h"

#include "api/CPP/topology.hpp"
#include "api/CPP/network.hpp"
#include "api/CPP/profiling.hpp"
#include "api/CPP/memory.hpp"

#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

enum PrintType
{
    Verbose,
    Perf,
    ExtendedTesting,

    PrintType_count // must be last
};

/// Information about executable.
class executable_info
{
    std::string _path;
    std::string _file_name_wo_ext;
    std::string _dir;


public:
    /// Gets absolute path to executable.
    const std::string& path() const
    {
        return _path;
    }

    /// Gets executable file name without extension (stem name).
    const std::string& file_name_wo_ext() const
    {
        return _file_name_wo_ext;
    }

    /// Gets aboulte path to executable directory.
    const std::string& dir() const
    {
        return _dir;
    }

    /// Creates new instance of information about current executable.
    ///
    /// @tparam StrTy1_ Type of first string argument (std::string constructible; use for forwarding).
    /// @tparam StrTy2_ Type of second string argument (std::string constructible; use for forwarding).
    /// @tparam StrTy3_ Type of third string argument (std::string constructible; use for forwarding).
    ///
    /// @param path             Absolute path to executable.
    /// @param file_name_wo_ext Executable file name without extension (stem name).
    /// @param dir              Absolute path to executable directory.
    template <typename StrTy1_, typename StrTy2_, typename StrTy3_,
              typename = std::enable_if_t<std::is_constructible<std::string, StrTy1_>::value &&
                                          std::is_constructible<std::string, StrTy2_>::value &&
                                          std::is_constructible<std::string, StrTy3_>::value, void>>
    executable_info(StrTy1_&& path, StrTy2_&& file_name_wo_ext, StrTy3_&& dir)
        : _path(std::forward<StrTy1_>(path)),
          _file_name_wo_ext(std::forward<StrTy2_>(file_name_wo_ext)),
          _dir(std::forward<StrTy3_>(dir))
    {}
};


/// Sets information about executable based on "main"'s command-line arguments.
///
/// It works only once (if successful). Next calls to this function will not modify
/// global executable's information object.
///
/// @param argc Main function arguments count.
/// @param argv Main function argument values.
///
/// @exception std::runtime_error Main function arguments do not contain executable name.
/// @exception boost::filesystem::filesystem_error Cannot compute absolute path to executable.
void set_executable_info(int argc, const char* const argv[]);

/// Gets information about executable.
///
/// Information is fetched only if information was set using set_executable_info() and not yet
/// destroyed (during global destruction). Otherwise, exception is thrown.
///
/// @return Shared pointer pointing to valid executable information.
///
/// @exception std::runtime_error Executable information was not set or it is no longer valid.
std::shared_ptr<const executable_info> get_executable_info();


/// Joins path using native path/directory separator.
///
/// @param parent Parent path.
/// @param child  Child part of path.
///
/// @return Joined path.
std::string join_path(const std::string& parent, const std::string& child);

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

    PrintType print_type = PrintType::Verbose; // printing modes - to support Verbose, vs Perf ony, vs ImageNet testing prints
    size_t loop = 1; // running the same input in a loop for smoothing perf results

    bool perf_per_watt; // power instrumentation
    bool disable_mem_pool; // memory optimization
};

struct memory_filler
{
    typedef enum
    {
        zero = 0,
        one,
        zero_to_nine,
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
};

std::vector<std::string> get_directory_images(const std::string& images_path);
std::vector<std::string> get_directory_weights(const std::string& images_path);

template <typename MemElemTy = float>
void load_images_from_file_list(const std::vector<std::string>& images_list, cldnn::memory& memory); 

/// function moved from alexnet.cpp, they will be probably used by each topology
void print_profiling_table(std::ostream& os, const std::vector<cldnn::instrumentation::profiling_info>& profiling_info);
cldnn::network build_network(const cldnn::engine& engine, const cldnn::topology& topology, const execution_params &ep);
uint32_t get_next_nearest_power_of_two(int number);
uint32_t get_gpu_batch_size(int number);

std::chrono::nanoseconds execute_topology(cldnn::network network,
                                          const execution_params &ep,
                                          CIntelPowerGadgetLib& energyLib,
                                          cldnn::memory& output);

void run_topology(const execution_params &ep);
