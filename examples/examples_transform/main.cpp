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
#include "command_line_utils.h"
#include "transform_output_printer.h"

#include "topologies.h"

#include <api/CPP/layout.hpp>
#include <api/CPP/memory.hpp>

#include <boost/filesystem.hpp>
#include <boost/optional.hpp>

#include <cstdint>
#include <iostream>
#include <string>

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

namespace
{

/// @brief Type that can store image dimensions (alias).
using img_dims_type = cmdline::pair<unsigned, unsigned>;

/// @brief Extended execution parameters (for transformation / segmentation).
struct transform_execution_params : execution_params
{
    cmdline::pair<unsigned, unsigned> expected_image_dims = {0, 0};
    bool use_image_dims                                   = false;
    bool lossless                                         = false;

    std::string compare_ref_img_dir;
    bool compare_per_channel = false;
    bool compare_histograms  = false;
};

} // unnamed namespace

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
            "It can be one of:\n  \tfns-candy, fns-feathers, fns-lamuse, fns-mosaic, fns-thescream, "
            "fns-udnie, test-transpose.")
        ("weights,w", bpo::value<std::string>()->value_name("<weights-dir>"),
            "Path to directory containing weights used in classification.\n"
            "Non-absolute paths are computed in relation to <executable-dir> (not working directory).\n"
            "If not specified, the \"<executable-dir>/<model-name>\" path is used in first place with "
            "\"<executable-dir>/weights\" as fallback.")
        ("dims", bpo::value<img_dims_type>()->value_name("<img-dims>")->default_value({0, 0}),
            "Expected dimensions (width and height) of input images. The dimensions are specified in the following "
            "form:\n  \t\"<image-width> x <image-height>\"."
            "\nIf dimensions are specified (both non-zero), the input images are normalized to these dimensions. "
            "Otherwise, the expected input sizes are calculated from topology default spatial sizes or from "
            "sizes of first listed image found in input directory (if --use_image_dims is present) and the rest "
            "of images is normalized to those sizes."
            "\nNOTE: If topology / model does not support multiple input dimensions or specific input dimension, "
            "it can overwrite the passed / calculated expected dimensions.")
        ("use_image_dims", bpo::bool_switch(),
            "Indicates that expected dimensions of input images should be calculated / taken from first listed image "
            "found in input directory. If not present, the default spatial dimensions of used model / topology "
            "will be applied as expected dimensions of input images."
            "\nNOTE: If topology / model does not support multiple input dimensions or specific input dimension, "
            "it can overwrite the passed / calculated expected dimensions.")
        ("lossless", bpo::bool_switch(),
            "Indicates that output should be written to files in loss-less format (PNG)."
            "\nAll output files will be written with full name of input file (including original extension) plus "
            "additional PNG file extension.")
        ("compare,C", bpo::value<std::string>()->value_name("<ref-images-dir>"),
            "Path to directory containing reference output images to compare with actual output.")
        ("compare_per_channel", bpo::bool_switch(),
            "Indicates that comparison should incorporate per-channel comparison.\nIgnored if --compare is absent.")
        ("compare_histograms", bpo::bool_switch(),
            "Indicates that comparison should incorporate comparison histograms.\nIgnored if --compare is absent.");

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

static bool run_topology(const transform_execution_params& exec_params)
{
    // Calculating preferred batch size.
    const auto batch_size               = exec_params.batch;
    const auto gpu_preferred_batch_size = get_gpu_batch_size(batch_size);
    if (gpu_preferred_batch_size != batch_size && !exec_params.rnn_type_of_topology)
    {
        std::cerr << "WARNING: This is not the optimal batch size. You have "
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
                std::cerr << "WARNING: Could not create directory for dump of sources. Path to directory: \""
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
            std::cerr << "WARNING: Could not initialize cldnn::engine with out-of-order queue."
                << "\n    Error (" << std::to_string(ex.status()) << "): " << ex.what()
                << "\n    --- falling back to in-order queue." << std::endl;
        }
        catch (std::exception& ex)
        {
            std::cerr << "WARNING: Could not initialize cldnn::engine with out-of-order queue."
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
            std::cerr << "WARNING: Intel(C) Power Gadget(C) is not initialized."
                << "\n    Error: " << power_measure_lib.GetLastError() << std::endl;
        }
    }

    transform::output_printer printer(exec_params.topology_name, exec_params.topology_name,
                                      !exec_params.compare_ref_img_dir.empty(), exec_params.compare_per_channel,
                                      exec_params.compare_histograms);

    // Building topology.
    cldnn::topology selected_topology;

    if (exec_params.print_type == print_type::verbose)
        std::cout << "Building of \"" << exec_params.topology_name << "\" started..." << std::endl;
    else if (exec_params.print_type == print_type::extended_testing)
        std::cout << "Extended testing of \"" << exec_params.topology_name << "\" started..." << std::endl;

    // Calculating properties of input.
    auto input_paths = list_input_files(exec_params.input_dir, input_file_type::image);
    if (input_paths.empty())
        throw std::runtime_error("Specified input images / files directory is empty "
                                 "(does not contain any useful data / files)");

    auto expected_img_dims = exec_params.expected_image_dims;
    if (exec_params.use_image_dims)
    {
        auto image_dims = get_image_dims(input_paths, 1);
        if (!image_dims.empty())
            expected_img_dims = cmdline::make_pair(std::get<1>(image_dims.front()), std::get<2>(image_dims.front()));
    }

    // --- Measurement: build time (start).
    cldnn::instrumentation::timer<> timer_build;

    auto input_layout = cldnn::layout{
        exec_params.use_half ? cldnn::data_types::f16 : cldnn::data_types::f32,
        cldnn::format::byxf, {
            static_cast<std::int32_t>(gpu_preferred_batch_size),
            3,
            static_cast<std::int32_t>(expected_img_dims.first),
            static_cast<std::int32_t>(expected_img_dims.second)
        }
    };

    if (exec_params.topology_name.compare(0, 4, "fns-") == 0)
        selected_topology = build_fns_instance_norm(exec_params.weights_dir, selected_engine, input_layout);
    else if(exec_params.topology_name == "test-transpose")
        selected_topology = build_test_transpose(exec_params.weights_dir, selected_engine, input_layout);
    else
        throw std::runtime_error("Topology \"" + exec_params.topology_name + "\" not implemented");

    const auto build_time = timer_build.uptime();
    // --- Measurement: build time (stop).

    if (exec_params.print_type == print_type::verbose)
    {
        std::cout << "Building of \"" << exec_params.topology_name << "\" finished (time: "
            << instrumentation::to_string(build_time) << ")." << std::endl;
    }
    if (!exec_params.run_single_kernel_name.empty())
    {
        const auto ids = selected_topology.get_primitive_ids();
        if (std::find(ids.cbegin(), ids.cend(), exec_params.run_single_kernel_name) == ids.cend())
            throw std::runtime_error("Topology does not contain primitive with name specified by "
                                     "\"--run_single_kernel\"");
    }

    auto network = build_network(selected_engine, selected_topology, exec_params);

    float zero = 0;
    cldnn::layout zero_layout(cldnn::data_types::f32, cldnn::format::bfyx, cldnn::tensor(1));
    auto output = cldnn::memory::attach(zero_layout, &zero, 1);

    auto input = cldnn::memory::allocate(selected_engine, input_layout);

    auto is_all_matched       = true;
    auto img_paths_it         = input_paths.cbegin();
    const auto img_paths_last = input_paths.cend();
    while (img_paths_it != img_paths_last)
    {
        std::pair<std::vector<std::string>::const_iterator, std::vector<std::string>> loaded_image_info;
        if (!exec_params.rnn_type_of_topology)
        {
            if (exec_params.use_half)
                loaded_image_info = load_image_files<half_t>(img_paths_it, img_paths_last, input);
            else
                loaded_image_info = load_image_files(img_paths_it, img_paths_last, input);

            img_paths_it = loaded_image_info.first;
        }
        else
            throw std::runtime_error("Recurrent neural networks are not supported");

        network.set_input_data("input", input);

        fp_seconds_type time{0.0};
        if (!exec_params.rnn_type_of_topology)
            time = execute_cnn_topology(network, exec_params, power_measure_lib, output);
        // NOTE: Leaving place for future RNN topologies.

        if (!exec_params.rnn_type_of_topology &&
            exec_params.run_until_primitive_name.empty() && exec_params.run_single_kernel_name.empty())
        {
            auto is_batch_matched = printer.batch(loaded_image_info.second, output, exec_params.input_dir,
                                                  exec_params.print_type, exec_params.compare_ref_img_dir);

            if (!is_batch_matched && !exec_params.compare_ref_img_dir.empty())
                is_all_matched = false;
        }
        else if (!exec_params.run_until_primitive_name.empty())
        {
            std::cout << "Finished at user-specified primitive: \""
                << exec_params.run_until_primitive_name << "\"." << std::endl;
        }
        else if (!exec_params.run_single_kernel_name.empty())
        {
            std::cout << "Run of single layer \"" << exec_params.run_single_kernel_name
                << "\" finished correctly." << std::endl;
        }

        if (exec_params.print_type != print_type::extended_testing && time.count() > 0.0)
        {
            std::cout << "Frames per second: " << exec_params.loop * batch_size / time.count() << std::endl;

            if (exec_params.perf_per_watt)
            {
                if (!power_measure_lib.print_power_results(exec_params.loop * batch_size / time.count()))
                    std::cerr << "WARNING: Parsing of results file from power measurement tool failed!!!" << std::endl;
            }
        }
    }

    return is_all_matched;
}

// --------------------------------------------------------------------------------------------------------------------

int main(const int argc, char* argv[])
{
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

        // Validate expected image dimensions.
        auto expected_img_dims = parsed_args["dims"].as<img_dims_type>();
        if ((expected_img_dims.first == 0 || expected_img_dims.second == 0) &&
            expected_img_dims.first != expected_img_dims.second)
        {
            std::cerr << "WARNING: specified expected image dimensions are incorrect (\"" << expected_img_dims
                << "\"). Either both or none dimensions must be zero. Ignoring --dims parameter!!!" << std::endl;

            expected_img_dims.first = 0;
            expected_img_dims.second = 0;
        }

        auto use_image_dims = parsed_args["use_image_dims"].as<bool>();
        if (use_image_dims && (expected_img_dims.first != 0 || expected_img_dims.second != 0))
        {
            std::cerr << "WARNING: --use_image_dims parameter is ignored (--dims parameter is specified)!!!" << std::endl;

            use_image_dims = false;
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
        ep.expected_image_dims = expected_img_dims;
        ep.use_image_dims      = use_image_dims;
        ep.lossless            = parsed_args["lossless"].as<bool>();
        ep.compare_ref_img_dir = cmp_ref_img_dir;
        ep.compare_per_channel = parsed_args["compare_per_channel"].as<bool>();
        ep.compare_histograms  = parsed_args["compare_histograms"].as<bool>();

        // Validate and run topology.
        if (ep.topology_name == "fns-candy" ||
            ep.topology_name == "fns-feathers" ||
            ep.topology_name == "fns-lamuse" ||
            ep.topology_name == "fns-mosaic" ||
            ep.topology_name == "fns-thescream" ||
            ep.topology_name == "fns-udnie" ||
            ep.topology_name == "test-transpose")
        {
            if (run_topology(ep))
                return 0;
            return 100; // Comparision is enabled and matching/diff failed.
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
