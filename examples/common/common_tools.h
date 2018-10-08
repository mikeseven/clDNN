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
#include "lstm_utils.h"

#include "api/CPP/topology.hpp"
#include "api/CPP/network.hpp"
#include "api/CPP/profiling.hpp"
#include "api/CPP/memory.hpp"

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

    /// Gets absolute path to executable directory.
    const std::string& dir() const
    {
        return _dir;
    }

    /// Creates new instance of information about current executable.
    ///
    /// @tparam Str1Ty Type of first string argument (std::string constructible; use for forwarding).
    /// @tparam Str2Ty Type of second string argument (std::string constructible; use for forwarding).
    /// @tparam Str3Ty Type of third string argument (std::string constructible; use for forwarding).
    ///
    /// @param path             Absolute path to executable.
    /// @param file_name_wo_ext Executable file name without extension (stem name).
    /// @param dir              Absolute path to executable directory.
    template <typename Str1Ty, typename Str2Ty, typename Str3Ty,
              typename = std::enable_if_t<std::is_constructible<std::string, Str1Ty>::value &&
                                          std::is_constructible<std::string, Str2Ty>::value &&
                                          std::is_constructible<std::string, Str3Ty>::value, void>>
    executable_info(Str1Ty&& path, Str2Ty&& file_name_wo_ext, Str3Ty&& dir)
        : _path(std::forward<Str1Ty>(path)),
          _file_name_wo_ext(std::forward<Str2Ty>(file_name_wo_ext)),
          _dir(std::forward<Str3Ty>(dir))
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

    //RNN specific
    bool        rnn_type_of_topology = false;
    float       temperature;
    std::string vocabulary_file;
    uint32_t    sequence_length;

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
};

/// @brief Returns list of all input files from specified directory (recursive).
///
/// @param root_dir         Root directory to scan for input files.
/// @param image_files_only Indicates that only (single) image files should
///                         be returned (exclude text files and image archives).
/// @return                 List of file paths (paths are normalized and returned as absolute paths).
std::vector<std::string> list_input_files(const std::string& root_dir, bool image_files_only = false);
std::vector<std::string> get_directory_weights(const std::string& images_path);

template <typename MemElemTy = float>
void load_images_from_file_list(const std::vector<std::string>& images_list, cldnn::memory& memory, const uint32_t min_size = 0); 

template <typename MemElemTy = float>
void load_data_from_file_list_lenet(const std::vector<std::string>& images_list, cldnn::memory& memory, const uint32_t images_offset, const uint32_t images_number, const bool train, cldnn::memory& memory_labels);

template <typename MemElemTy = float>
void load_data_from_file_list_imagenet(const std::vector<std::string>& images_list, const std::string& input_dir, cldnn::memory& memory, const uint32_t images_offset, const uint32_t images_number, const bool train, cldnn::memory& memory_labels);

template <typename MemElemTy = float>
void load_data_from_file_list_cifar10(const std::vector<std::string>& images_list, const std::string& input_dir, cldnn::memory& memory, const uint32_t images_offset, const uint32_t images_number, const bool train, cldnn::memory& memory_labels);

void compute_image_mean(const execution_params &ep, cldnn::engine& engine, bool use_cifar10);

/// function moved from alexnet.cpp, they will be probably used by each topology
void print_profiling_table(std::ostream& os, const std::vector<cldnn::instrumentation::profiling_info>& profiling_info);
cldnn::network build_network(const cldnn::engine& engine, const cldnn::topology& topology, const execution_params &ep, const std::vector<cldnn::primitive_id> &output_ids = std::vector<cldnn::primitive_id>(0));
uint32_t get_next_nearest_power_of_two(int number);
uint32_t get_gpu_batch_size(int number);

void make_instrumentations(const execution_params&, cldnn::memory&, std::map<cldnn::primitive_id, cldnn::network_output>&);

template<typename MemElemTy = float>
void prepare_data_for_lstm(lstm_utils& lstm_data, const std::vector<std::string>& input_files, const std::string& vocabulary_file);

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

std::chrono::nanoseconds execute_rnn_topology(cldnn::network network,
                                                const execution_params &ep,
                                                CIntelPowerGadgetLib& energyLib,
                                                cldnn::memory& output,
                                                cldnn::memory& input,
                                                lstm_utils& lstm_data);

void run_topology(const execution_params &ep);

std::string get_model_name(const std::string& topology_name);