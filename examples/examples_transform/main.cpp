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

#include "common_tools.h"
#include "neural_memory.h"
#include "command_line_utils.h"
#include "output_parser.h"

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
#include <type_traits>

using namespace cldnn::utils::examples;
using namespace cldnn::utils::examples::cmdline;


namespace cldnn
{
namespace utils
{
namespace examples
{
namespace cmdline
{
    const unsigned long app_version = 0x10000L;
} // namespace cmdline
} // namespace examples
} // namespace utils
} // namespace cldnn


struct transform_execution_params : execution_params
{
    std::string compare_ref_img_dir;
};

  /// Prepares command-line options for current application.
  ///
  /// @param exec_info Executable information.
  ///
  /// @return Helper class with basic messages and command-line options.
static cmdline_options prepare_cmdline_options(const std::shared_ptr<const executable_info>& exec_info)
{
    // ----------------------------------------------------------------------------------------------------------------
    // Standard options.
    auto standard_cmdline_options = cmdline_options::create_group("Standard options");
    standard_cmdline_options->add_options()
        ("input,i", bpo::value<std::string>()->value_name("<input-dir>")->required(),
            "Path to input directory containing images to transform (mandatory when running "
            "transformation / segmentation).")
        ("model,m", bpo::value<std::string>()->value_name("<model-name>")->default_value("fns-candy"),
            "Name of a neural network model that is used for transformation / segmentation.\n"
            "It can be one of:\n  \tfns-candy.")
        ("weights,w", bpo::value<std::string>()->value_name("<weights-dir>"),
            "Path to directory containing weights used in classification.\n"
            "Non-absolute paths are computed in relation to <executable-dir> (not working directory).\n"
            "If not specified, the \"<executable-dir>/<model-name>\" path is used in first place with "
            "\"<executable-dir>/weights\" as fallback.")
        ("compare,C", bpo::value<std::string>()->value_name("<ref-images-dir>"),
            "Path to directory containing reference output images to compare with actual output.");

    const auto help_msg_formatter =
        [](std::ostream& out, const std::shared_ptr<const executable_info>& exec_info) -> std::ostream&
    {
        // ------------------------------------------------------------------------------------------------------------
        // Help message.
        out << "Usage:\n  " << exec_info->file_name_wo_ext() << " [standard options]\n\n";
        out << "Executes image transformation / segmentation on specified neural network\n";
        out << "and writes transformed images to output.";
        return out;
    };

    return {exec_info, standard_cmdline_options, true, help_msg_formatter};
}

static void run_topology(const transform_execution_params& exec_params)
{
    // Calculating preferred batch size.
    const auto batch_size               = exec_params.batch;
    const auto gpu_preferred_batch_size = get_gpu_batch_size(batch_size);
    if (gpu_preferred_batch_size != batch_size && !exec_params.rnn_type_of_topology)
    {
        std::cout << "WARNING: This is not the optimal batch size. You have "
            << (gpu_preferred_batch_size - batch_size)
            << " dummy images per batch!!! Please use --batch=" << gpu_preferred_batch_size
            << " for best performance." << std::endl;
    }

    // Creating engine configurator.
    boost::optional<cldnn::engine> engine;
    const auto engine_configurator = [&exec_params](const bool use_ooq) -> cldnn::engine_configuration
    {
        std::string engine_log_file_path;
        std::string sources_dump_dir;

        if (exec_params.log_engine)
            engine_log_file_path = join_path(instrumentation::logger::get_dumps_dir(), "engine_log.txt");

        if (exec_params.dump_sources)
        {
            std::string err;
            sources_dump_dir = instrumentation::logger::create_sources_dumps_dir(err);
            if (!err.empty())
            {
                std::cout << "WARNING: Could not create directory for dump of sources. Path to directory: \""
                    << sources_dump_dir << "\" does not exist and cannot be created."
                    << "\n    Error: " << err
                    << "\n    --- dumping will be disabled." << std::endl;
                sources_dump_dir = "";
            }
        }

        return {
            exec_params.profiling,
            exec_params.meaningful_kernels_names,
            false,
            "",
            exec_params.run_single_kernel_name,
            use_ooq,
            engine_log_file_path,
            sources_dump_dir,
            cldnn::priority_mode_types::disabled,
            cldnn::throttle_mode_types::disabled,
            !exec_params.disable_mem_pool
        };
    };

    // Configuring engine.
    if (exec_params.use_oooq)
    {
        try
        {
            engine.emplace(engine_configurator(true));
        }
        catch (cldnn::error& ex)
        {
            std::cout << "WARNING: Could not initialize cldnn::engine with out-of-order queue."
                << "\n    Error (" << std::to_string(ex.status()) << "): " << ex.what()
                << "\n    --- falling back to in-order queue." << std::endl;
        }
        catch (std::exception& ex)
        {
            std::cout << "WARNING: Could not initialize cldnn::engine with out-of-order queue."
                << "\n    Error: " << ex.what()
                << "\n    --- falling back to in-order queue." << std::endl;
        }
    }
    if (!engine.is_initialized())
        engine.emplace(engine_configurator(false));
    auto& selected_engine = engine.get();

    // Configuring power measurement.
    CIntelPowerGadgetLib power_measure_lib;
    if (exec_params.perf_per_watt)
    {
        if (!power_measure_lib.IntelEnergyLibInitialize())
        {
            std::cout << "WARNING: Intel(C) Power Gadget(C) is not initialized."
                << "\n    Error: " << power_measure_lib.GetLastError() << std::endl;
        }
    }

    html output_file(exec_params.topology_name, exec_params.topology_name + " run");

    // Building topology.
    cldnn::topology selected_topology;

    if (exec_params.print_type == print_type::verbose)
        std::cout << "Building of \"" << exec_params.topology_name << "\" started..." << std::endl;
    else if (exec_params.print_type == print_type::extended_testing)
        std::cout << "Extended testing of \"" << exec_params.topology_name << "\" started..." << std::endl;

    // --- Measurement: build time (start).
    cldnn::instrumentation::timer<> timer_build;

    auto input_layout = cldnn::layout{
        exec_params.use_half ? cldnn::data_types::f16 : cldnn::data_types::f32,
        cldnn::format::byxf, {}
    };

    if (exec_params.topology_name == "fns-candy")
        selected_topology = build_fns_instance_norm(exec_params.weights_dir, selected_engine, input_layout,
                                                    gpu_preferred_batch_size);
    else
        throw std::runtime_error("Topology \"" + exec_params.topology_name + "\" not implemented!");

    const auto build_time = timer_build.uptime();
    // --- Measurement: build time (stop).

    if (exec_params.print_type == print_type::verbose)
    {
        std::cout << "Building of \"" << exec_params.topology_name << "\" finished (time: "
            << instrumentation::to_string(build_time) << ")." << std::endl;
    }
    if (!exec_params.run_single_kernel_name.empty())
    {
        auto all_ids = selected_topology.get_primitive_ids();
        if (std::find(all_ids.begin(), all_ids.end(), exec_params.run_single_kernel_name) == all_ids.end())
            throw std::runtime_error("Topology does not contain primitive with name specified by run_single_kernel");
    }

    auto network = build_network(selected_engine, selected_topology, exec_params);
    //TODO check if we can define the 'empty' memory
    float zero = 0;
    cldnn::layout zero_layout(cldnn::data_types::f32, cldnn::format::bfyx, { 1,1,1,1 });
    auto output = cldnn::memory::attach(zero_layout, &zero, 1);

    auto input = cldnn::memory::allocate(selected_engine, input_layout);

    auto input_list = list_input_files(exec_params.input_dir);
    if (input_list.empty())
        throw std::runtime_error("specified input images directory is empty (does not contain image data)");

    auto number_of_batches = (input_list.size() % batch_size == 0)
        ? input_list.size() / batch_size : input_list.size() / batch_size + 1;
    std::vector<std::string> input_files_in_batch;
    auto input_list_iterator = input_list.begin();
    auto input_list_end = input_list.end();

    for (decltype(number_of_batches) batch = 0; batch < number_of_batches; batch++)
    {
        input_files_in_batch.clear();
        for (uint32_t i = 0; i < batch_size && input_list_iterator != input_list_end; ++i, ++input_list_iterator)
        {
            input_files_in_batch.push_back(*input_list_iterator);
        }
        double time_in_sec = 0.0;
        lstm_utils lstm_data(exec_params.sequence_length, batch_size, (unsigned int)exec_params.loop, exec_params.temperature);
        // load croped and resized images into input
        if (!exec_params.rnn_type_of_topology)
        {
            if (exec_params.use_half)
            {
                load_images_from_file_list<half_t>(input_files_in_batch, input);
            }
            else
            {
                load_images_from_file_list(input_files_in_batch, input);
            }

        }
        else
        {
            prepare_data_for_lstm(lstm_data, input_files_in_batch, exec_params.vocabulary_file);
            if (exec_params.use_half)
            {
                lstm_data.fill_memory<half_t>(input);
            }
            else
            {
                lstm_data.fill_memory<float>(input);
            }
        }
        network.set_input_data("input", input);

        std::chrono::nanoseconds time;
        if (!exec_params.rnn_type_of_topology)
        {
            time = execute_cnn_topology(network, exec_params, power_measure_lib, output);
        }
        else
        {
            time = execute_rnn_topology(network, exec_params, power_measure_lib, output, input, lstm_data);
        }

        time_in_sec = std::chrono::duration_cast<std::chrono::duration<double, std::chrono::seconds::period>>(time).count();

        if (exec_params.run_until_primitive_name.empty() && exec_params.run_single_kernel_name.empty() && !exec_params.rnn_type_of_topology && exec_params.topology_name != "lenet" && exec_params.topology_name != "lenet_train")
        {
            output_file.batch(output, join_path(get_executable_info()->dir(), neurons_list_filename), input_files_in_batch, exec_params.print_type);
        }
        else if (!exec_params.run_until_primitive_name.empty())
        {
            std::cout << "Finished at user custom primtive: " << exec_params.run_until_primitive_name << std::endl;
        }
        else if (!exec_params.run_single_kernel_name.empty())
        {
            std::cout << "Run_single_layer finished correctly." << std::endl;
        }

        if (time_in_sec != 0.0)
        {
            if (exec_params.print_type != print_type::extended_testing)
            {
                std::cout << "Frames per second:" << (double)(exec_params.loop * batch_size) / time_in_sec << std::endl;

                if (exec_params.perf_per_watt)
                {
                    if (!power_measure_lib.print_power_results((double)(exec_params.loop * batch_size) / time_in_sec))
                        std::cout << "WARNING: power file parsing failed." << std::endl;
                }
            }
        }
    }

}

// --------------------------------------------------------------------------------------------------------------------

int main(const int argc, char* argv[])
{
    namespace bpo = boost::program_options;
    namespace bfs = boost::filesystem;

    set_executable_info(argc, argv); // Must be set before using get_executable_info().

    // Parsing command-line and handling/presenting basic options.
    const auto exec_info = get_executable_info();
    auto options = prepare_cmdline_options(exec_info);
    bpo::variables_map parsed_args;
    try
    {
        parsed_args = parse_cmdline_options(options, argc, argv);
        if (parse_version_help_options(parsed_args, options))
            return 0;

        if (!parsed_args.count("input"))
        {
            std::cerr << "ERROR: --input required option was not specified!!!\n\n";
            std::cerr << options.help_message() << std::endl;
            return 1;
        }
    }
    catch (const std::exception& ex)
    {
        std::cerr << "ERROR: " << ex.what() << "!!!\n\n";
        std::cerr << options.help_message() << std::endl;
        return 1;
    }

    // Execute network for transformation / segmentation.
    try
    {
        transform_execution_params ep;
        parse_common_options(parsed_args, ep);

        // Validate input directory.
        const auto input_dir = parsed_args["input"].as<std::string>();
        if (!bfs::exists(input_dir) || !bfs::is_directory(input_dir))
        {
            std::cerr << "ERROR: specified input images path (\"" << input_dir
                << "\") does not exist or does not point to directory (--input option invalid)!!!" << std::endl;
            return 1;
        }

        // Determine weights directory (either based on executable directory - if not specified, or
        // relative to current working directory or absolute - if specified).
        std::string weights_dir;
        if (parsed_args.count("weights"))
            weights_dir = bfs::absolute(parsed_args["weights"].as<std::string>(), exec_info->dir()).string();
        else
        {
            weights_dir = join_path(exec_info->dir(), parsed_args["model"].as<std::string>());
            if (!bfs::exists(weights_dir) || !bfs::is_directory(weights_dir))
                weights_dir = join_path(exec_info->dir(), "weights");
        }

        // Validate weights directory.
        if (!bfs::exists(weights_dir) || !bfs::is_directory(weights_dir))
        {
            if (parsed_args.count("weights"))
            {
                std::cerr << "ERROR: specified network weights path (\"" << weights_dir
                    << "\") does not exist or does not point to directory (--weights option invalid)!!!" << std::endl;
            }
            else
            {
                std::cerr << "ERROR: could not find default network weights path for selected topology. Neither '"
                    << parsed_args["model"].as<std::string>() << "' nor 'weights' folder exist!!!" << std::endl;
            }

            return 1;
        }

        // Validate reference images directory.
        std::string cmp_ref_img_dir;
        if (parsed_args.count("compare"))
        {
            cmp_ref_img_dir = parsed_args["compare"].as<std::string>();
            if (!bfs::exists(cmp_ref_img_dir) || !bfs::is_directory(cmp_ref_img_dir))
            {
                std::cerr << "ERROR: specified reference images path (\"" << cmp_ref_img_dir
                    << "\") does not exist or does not point to directory (--compare option invalid)!!!" << std::endl;
                return 1;
            }
        }

        ep.input_dir           = input_dir;
        ep.topology_name       = parsed_args["model"].as<std::string>();
        ep.weights_dir         = weights_dir;
        ep.compare_ref_img_dir = cmp_ref_img_dir;

        // Validate and run topology.
        if (ep.topology_name == "fns-candy")
        {
            run_topology(ep);
            return 0;
        }
        std::cerr << "ERROR: model / topology (\"" << ep.topology_name << "\") is not implemented!!!" << std::endl;
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
