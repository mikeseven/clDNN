/*
// Copyright (c) 2018 Intel Corporation
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

#include "../common/common_tools.h"
#include "../common/neural_memory.h"
#include "../common/command_line_utils.h"
#include "../common/output_parser.h"

#include "api/CPP/memory.hpp"
#include "topologies.h"

#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/numeric/conversion/cast.hpp>
#include <boost/program_options.hpp>
#include <boost/optional.hpp>

#include <cstdint>
#include <iostream>
#include <regex>
#include <string>
#include <sstream>
#include <type_traits>


  /// Prepares command-line options for current application.
  ///
  /// @param exec_info Executable information.
  ///
  /// @return Helper class with basic messages and command-line options.
static cmdline_options prepare_cmdline_options(const std::shared_ptr<const executable_info>& exec_info)
{
    namespace bpo = boost::program_options;

    // ----------------------------------------------------------------------------------------------------------------
    // Standard options.
    bpo::options_description standard_cmdline_options("Standard options");
    add_base_options(standard_cmdline_options);
    standard_cmdline_options.add_options()
        ("model", bpo::value<std::string>()->value_name("<model-name>")->default_value("microbench_conv"),
            "Name of a neural network model that is used for classification.\n"
            "It can be one of:\n  \tmicrobench_conv, microbench_lstm.")
        ("lstm_input_size", bpo::value<std::uint32_t>()->value_name("<input_size>")->default_value(10),
            "LSTM microbench input size.")
        ("lstm_hidden_size", bpo::value<std::uint32_t>()->value_name("<hidden_size>")->default_value(7),
            "LSTM microbench hidden size.")
        ("lstm_sequence_len", bpo::value<std::uint32_t>()->value_name("<seq_len>")->default_value(1),
            "LSTM sequence length.")
        ("lstm_batch_size", bpo::value<std::uint32_t>()->value_name("<batch_size>")->default_value(1),
            "LSTM batch size.")
        ("lstm_no_biases", bpo::bool_switch(),
            "LSTM disable use of biases.")
        ("lstm_initial_hidden", bpo::bool_switch(),
            "LSTM use initial hidden tensor.")
        ("lstm_initial_cell", bpo::bool_switch(),
            "LSTM use initial cell tensor.");

    // All options.
    bpo::options_description all_cmdline_options;
    all_cmdline_options.add(standard_cmdline_options);
    // All visible options.
    bpo::options_description all_visible_cmdline_options;
    all_visible_cmdline_options.add(standard_cmdline_options);

    auto version_msg_out = print_version_message(exec_info);

    auto help_msg_out = print_help_message(exec_info);
    help_msg_out << all_visible_cmdline_options;

    return {all_cmdline_options, help_msg_out.str(), version_msg_out.str()};
}

void run_topology(const execution_params &ep)
{
    uint32_t batch_size = ep.batch;

    uint32_t gpu_batch_size = get_gpu_batch_size(batch_size);
    if (gpu_batch_size != batch_size && !ep.rnn_type_of_topology)
    {
        std::cout << "WARNING: This is not the optimal batch size. You have " << (gpu_batch_size - batch_size)
            << " dummy images per batch!!! Please use batch=" << gpu_batch_size << "." << std::endl;
    }

    boost::optional<cldnn::engine> eng_storage;

    const auto get_config = [&ep](bool use_ooq)
    {
        std::string engine_log;
        std::string sources_dir;
        if (ep.log_engine)
            engine_log = instrumentation::logger::get_dumps_dir() + "/engine_log.txt";
        if (ep.dump_sources)
        {
            std::string err;
            sources_dir = instrumentation::logger::create_sources_dumps_dir(err);
            if (!err.empty())
            {
                std::cout << "Could not create directory for sources dumps, directory path: '" + sources_dir + "'\n    error: " + err + "\n    -- dumping will be disabled." << std::endl;
                sources_dir = "";
            }
        }

        return cldnn::engine_configuration(ep.profiling, ep.meaningful_kernels_names, false, "", ep.run_single_kernel_name, use_ooq, engine_log, sources_dir, cldnn::priority_mode_types::disabled, cldnn::throttle_mode_types::disabled, !ep.disable_mem_pool);
    };

    if (ep.use_oooq)
    {
        //try to init oooq engine
        try {
            eng_storage.emplace(get_config(true));
        }
        catch (cldnn::error& err) {
            std::cout << "Could not initialize cldnn::engine with out-of-order queue,\n    error: (" + std::to_string(err.status()) + ") " + err.what() << "\n    --- fallbacking to in-order-queue" << std::endl;
        }
    }

    //if initialization failed, fallback to in-order queue
    if (!eng_storage.is_initialized())
        eng_storage.emplace(get_config(false));

    cldnn::engine& engine = eng_storage.get();

    CIntelPowerGadgetLib energyLib;
    if (ep.perf_per_watt)
    {
        if (energyLib.IntelEnergyLibInitialize() == false)
        {
            std::cout << "WARNING: Intel Power Gadget isn't initialized. msg: " << energyLib.GetLastError();
        }
    }

    html output_file(ep.topology_name, ep.topology_name + " run");

    cldnn::topology primitives;

    if (ep.print_type == Verbose)
    {
        std::cout << "Building " << ep.topology_name << " started" << std::endl;
    }
    else if (ep.print_type == ExtendedTesting)
    {
        std::cout << "Extended testing of " << ep.topology_name << std::endl;
    }

    cldnn::instrumentation::timer<> timer_build;
    cldnn::layout input_layout = { ep.use_half ? cldnn::data_types::f16 : cldnn::data_types::f32, cldnn::format::byxf,{} };
    std::map<cldnn::primitive_id, cldnn::layout> microbench_conv_inputs;
    std::map<cldnn::primitive_id, cldnn::layout> microbench_lstm_inputs;
    microbench_conv_inputs.insert({ "input_layout", input_layout }); //add dummy input so we pass data format to mcirobench_conv topology

    if (ep.topology_name == "microbench_conv")
        primitives = build_microbench_conv(ep.weights_dir, engine, microbench_conv_inputs, gpu_batch_size);
    else if (ep.topology_name == "microbench_lstm")
        primitives = build_microbench_lstm(ep.weights_dir, engine, ep.lstm_ep, microbench_lstm_inputs);
    else
        throw std::runtime_error("Topology \"" + ep.topology_name + "\" not implemented!");
    microbench_conv_inputs.erase("input_layout");

    auto build_time = timer_build.uptime();

    if (ep.print_type == Verbose)
    {
        std::cout << "Building " << ep.topology_name << " finished in " << instrumentation::to_string(build_time) << std::endl;
    }
    if (!ep.run_single_kernel_name.empty())
    {
        auto all_ids = primitives.get_primitive_ids();
        if (std::find(all_ids.begin(), all_ids.end(), ep.run_single_kernel_name) == all_ids.end())
        {
            throw std::runtime_error("Topology does not contain actual run_single_kernel name!");
        }
    }
    auto network = build_network(engine, primitives, ep);
    //TODO check if we can define the 'empty' memory
    float zero = 0;
    cldnn::layout zero_layout(cldnn::data_types::f32, cldnn::format::bfyx, { 1,1,1,1 });
    auto output = cldnn::memory::attach(zero_layout, &zero, 1);

    if (ep.topology_name == "microbench_conv") {
        auto fill_with = memory_filler::filler_type::zero;
        for (const auto& inp_data : microbench_conv_inputs)
        {
            auto mem = cldnn::memory::allocate(engine, inp_data.second);
            if (!ep.use_half)
            {
                memory_filler::fill_memory<float>(mem, fill_with);
            }
            else
            {
                memory_filler::fill_memory<half_t>(mem, fill_with);
            }

            network.set_input_data(inp_data.first, mem);
        }

        auto time = execute_cnn_topology(network, ep, energyLib, output);
        auto time_in_sec = std::chrono::duration_cast<std::chrono::duration<double, std::chrono::seconds::period>>(time).count();
        if (time_in_sec != 0.0)
        {
            if (ep.print_type != ExtendedTesting)
            {
                std::cout << "Frames per second:" << (double)(ep.loop * batch_size) / time_in_sec << std::endl;

                if (ep.perf_per_watt)
                {
                    if (!energyLib.print_power_results((double)(ep.loop * batch_size) / time_in_sec))
                        std::cout << "WARNING: power file parsing failed." << std::endl;
                }
            }
        }
    }
    else if (ep.topology_name == "microbench_lstm") {
        auto fill_with = memory_filler::filler_type::zero;
        for (const auto& inp_data : microbench_lstm_inputs)
        {
            auto mem = cldnn::memory::allocate(engine, inp_data.second);
            memory_filler::fill_memory<float>(mem, fill_with);
            network.set_input_data(inp_data.first, mem);
        }

        auto time = execute_cnn_topology(network, ep, energyLib, output);
        auto time_in_sec = std::chrono::duration_cast<std::chrono::duration<double, std::chrono::seconds::period>>(time).count();
        if (time_in_sec != 0.0)
        {
            if (ep.print_type != ExtendedTesting)
            {
                std::cout << "Frames per second:" << (double)(ep.loop * batch_size) / time_in_sec << std::endl;

                if (ep.perf_per_watt)
                {
                    if (!energyLib.print_power_results((double)(ep.loop * batch_size) / time_in_sec))
                        std::cout << "WARNING: power file parsing failed." << std::endl;
                }
            }
        }
    }

}

// --------------------------------------------------------------------------------------------------------------------

int main(int argc, char* argv[])
{
    namespace bpo = boost::program_options;
    namespace bfs = boost::filesystem;
    bool microbench_conv = false;
    bool microbench_lstm = false;

    set_executable_info(argc, argv); // Must be set before using get_executable_info().

    // Parsing command-line and handling/presenting basic options.
    auto exec_info = get_executable_info();
    auto options = prepare_cmdline_options(exec_info);
    bpo::variables_map parsed_args;
    try
    {
        parsed_args = parse_cmdline_options(options, argc, argv);
        microbench_conv = parsed_args["model"].as<std::string>() == "microbench_conv";
        microbench_lstm = parsed_args["model"].as<std::string>() == "microbench_lstm";

        if (parse_help_version(parsed_args, options))
            return 0;
    }
    catch (const std::exception& ex)
    {
        std::cerr << "ERROR: " << ex.what() << "!!!\n\n";
        std::cerr << options.help_message() << std::endl;
        return 1;
    }

    // Execute network or convert weights.
    try
    {
        // Execute network otherwise.
        execution_params ep;
        parse_common_args(parsed_args, ep);
        ep.input_dir = "NA";
        ep.weights_dir = "NA";

        ep.topology_name = parsed_args["model"].as<std::string>();

        std::uint32_t print = parsed_args["print_type"].as<std::uint32_t>();
        ep.print_type = (PrintType)((print >= (std::uint32_t)PrintType::PrintType_count) ? 0 : print);

        if (microbench_lstm) {
            ep.lstm_ep.lstm_input_size = parsed_args["lstm_input_size"].as<std::uint32_t>();
            ep.lstm_ep.lstm_hidden_size = parsed_args["lstm_hidden_size"].as<std::uint32_t>();
            ep.lstm_ep.lstm_sequence_len = parsed_args["lstm_sequence_len"].as<std::uint32_t>();
            ep.lstm_ep.lstm_batch_size = parsed_args["lstm_batch_size"].as<std::uint32_t>();
            ep.lstm_ep.lstm_initial_cell = parsed_args["lstm_initial_cell"].as<bool>();
            ep.lstm_ep.lstm_initial_hidden = parsed_args["lstm_initial_hidden"].as<bool>();
            ep.lstm_ep.lstm_no_biases = parsed_args["lstm_no_biases"].as<bool>();
        }

        if (!ep.run_single_kernel_name.empty())
            ep.meaningful_kernels_names = true;

        if (ep.topology_name == "microbench_conv" ||
            ep.topology_name == "microbench_lstm")
        {
            run_topology(ep);
            return 0;
        }
        else
        {
            std::cerr << "ERROR: model/topology (\"" << ep.topology_name << "\") is not implemented!!!" << std::endl;
        }
    }

    catch (const std::exception& ex)
    {
        std::cerr << "ERROR: " << ex.what() << "!!!" << std::endl;
    }
    catch (...)
    {
        std::cerr << "ERROR: unknown exception!!!" << std::endl;
    }
    return 1;
}
