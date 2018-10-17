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
#include <sstream>
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
    const unsigned long app_version = 0x10200L;
} // namespace cmdline
} // namespace examples
} // namespace utils
} // namespace cldnn


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
        ("input", bpo::value<std::string>()->value_name("<input-dir>"),
            "Path to input directory containing images to classify (mandatory when running classification).")
        ("model", bpo::value<std::string>()->value_name("<model-name>")->default_value("alexnet"),
            "Name of a neural network model that is used for classification.\n"
            "It can be one of:\n  \talexnet, vgg16, vgg16_face, googlenet, gender, squeezenet, resnet50, resnet50-i8, ssd_mobilenet, ssd_mobilenet-i8")
        ("weights", bpo::value<std::string>()->value_name("<weights-dir>"),
            "Path to directory containing weights used in classification.\n"
            "Non-absolute paths are computed in relation to <executable-dir> (not working directory).\n"
            "If not specified, the \"<executable-dir>/<model-name>\" path is used in first place with \"<executable-dir>/weights\" as fallback.")
        ("use_calibration", bpo::bool_switch(),
            "Uses int8 precision and output calibration. Supported topologies: squeezenet");

    // Conversions options.
    auto weights_conv_cmdline_options = cmdline_options::create_group("Weights conversion options");
    weights_conv_cmdline_options->add_options()
        ("convert", bpo::value<cldnn::backward_comp::neural_memory::nnd_layout_format::type>()->value_name("<nnd_layout_format-type>"),
            "Convert weights of a neural network to given nnd_layout_format (<nnd_layout_format-type> represents numeric value of "
            "cldnn::neural_memory::nnd_layout_format enum).")
        ("convert_filter", bpo::value<std::string>()->value_name("<filter>"),
            "Name or part of the name of weight file(s) to be converted.\nFor example:\n"
            "  \"conv1\" - first convolution,\n  \"fc\" - every fully connected.");

    return {exec_info, standard_cmdline_options, true, {}, {standard_cmdline_options, weights_conv_cmdline_options}};
}

void run_topology(const execution_params &ep)
{
    uint32_t batch_size = ep.batch;

    uint32_t gpu_batch_size = get_gpu_batch_size(batch_size);
    if (gpu_batch_size != batch_size)
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

    if (ep.print_type == print_type::verbose)
    {
        std::cout << "Building " << ep.topology_name << " started" << std::endl;
    }
    else if (ep.print_type == print_type::extended_testing)
    {
        std::cout << "Extended testing of " << ep.topology_name << std::endl;
    }

    cldnn::instrumentation::timer<> timer_build;
    cldnn::layout input_layout = { ep.use_half ? cldnn::data_types::f16 : cldnn::data_types::f32, cldnn::format::byxf,{} };
    if (ep.topology_name == "alexnet")
        primitives = build_alexnet(ep.weights_dir, engine, input_layout, gpu_batch_size);
    else if (ep.topology_name == "vgg16" || ep.topology_name == "vgg16_face")
        primitives = build_vgg16(ep.weights_dir, engine, input_layout, gpu_batch_size);
    else if (ep.topology_name == "googlenet")
        primitives = build_googlenetv1(ep.weights_dir, engine, input_layout, gpu_batch_size);
    else if (ep.topology_name == "gender")
        primitives = build_gender(ep.weights_dir, engine, input_layout, gpu_batch_size);
    else if (ep.topology_name == "resnet50")
        primitives = build_resnet50(ep.weights_dir, engine, input_layout, gpu_batch_size);
    else if (ep.topology_name == "resnet50-i8")
        primitives = build_resnet50_i8(ep.weights_dir, engine, input_layout, gpu_batch_size);
    else if (ep.topology_name == "squeezenet")
    {
        if (ep.calibration)
        {
            primitives = build_squeezenet_quant(ep.weights_dir, engine, input_layout, gpu_batch_size);
        }
        else
        {
            primitives = build_squeezenet(ep.weights_dir, engine, input_layout, gpu_batch_size);
        }
    }
    else if (ep.topology_name == "ssd_mobilenet")
        primitives = build_ssd_mobilenet(ep.weights_dir, engine, input_layout, gpu_batch_size);
    else if (ep.topology_name == "ssd_mobilenet-i8")
        primitives = build_ssd_mobilenet_i8(ep.weights_dir, engine, input_layout, gpu_batch_size);
    else
        throw std::runtime_error("Topology \"" + ep.topology_name + "\" not implemented!");

    auto build_time = timer_build.uptime();

    if (ep.print_type == print_type::verbose)
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

    auto input = cldnn::memory::allocate(engine, input_layout);
    auto neurons_list_filename = "names.txt";
    if (ep.topology_name == "vgg16_face")
        neurons_list_filename = "vgg16_face.txt";
    else if (ep.topology_name == "gender")
        neurons_list_filename = "gender.txt";

    auto input_list = list_input_files(ep.input_dir);
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

        // load croped and resized images into input
        if (ep.use_half)
        {
            load_image_files<half_t>(input_files_in_batch, input);
        }
        else
        {
            load_image_files(input_files_in_batch, input);
        }
        network.set_input_data("input", input);

        std::chrono::nanoseconds time;

        time = execute_cnn_topology(network, ep, energyLib, output);

        time_in_sec = std::chrono::duration_cast<std::chrono::duration<double, std::chrono::seconds::period>>(time).count();

        if (ep.run_until_primitive_name.empty() && ep.run_single_kernel_name.empty() && ep.topology_name != "lenet" && ep.topology_name != "lenet_train")
        {
            output_file.batch(output, join_path(get_executable_info()->dir(), neurons_list_filename), input_files_in_batch, ep.print_type);
        }
        else if (!ep.run_until_primitive_name.empty())
        {
            std::cout << "Finished at user custom primtive: " << ep.run_until_primitive_name << std::endl;
        }
        else if (!ep.run_single_kernel_name.empty())
        {
            std::cout << "Run_single_layer finished correctly." << std::endl;
        }

        if (time_in_sec != 0.0)
        {
            if (ep.print_type != print_type::extended_testing)
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
    // TODO: create header file for all examples
    extern void alexnet(const execution_params &ep);
    extern void vgg16(const execution_params &ep);
    extern void googlenet_v1(const execution_params &ep);
    extern void convert_weights(cldnn::data_types dt, cldnn::format::type format,
                                const std::string& conv_path_filter, bool validate_magic = false);

    set_executable_info(argc, argv); // Must be set before using get_executable_info().

                                     // Parsing command-line and handling/presenting basic options.
    auto exec_info = get_executable_info();
    auto options = prepare_cmdline_options(exec_info);
    bpo::variables_map parsed_args;
    try
    {
        parsed_args = parse_cmdline_options(options, argc, argv);
        if (parse_version_help_options(parsed_args, options))
            return 0;

        if (!parsed_args.count("input") && !parsed_args.count("convert"))
        {
            std::cerr << "ERROR: none of required options was specified (either --input or\n";
            std::cerr << "       --convert is needed)!!!\n\n";
            std::cerr << options.help_message() << std::endl;
            return 1;
        }
        if (parsed_args["use_calibration"].as<bool>() && parsed_args["model"].as<std::string>() != "squeezenet" )
        {
            std::cerr << "calibration is supported for squeezenet only" << std::endl;
            
            if (parsed_args["use_half"].as<bool>())
            {
                std::cerr << "Can't use half and int8 precision together" << std::endl;
            }
            return 0;
        }
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
        // Convert weights if convertion options are used.
        if (parsed_args.count("convert"))
        {
            auto convert_filter = parsed_args.count("convert_filter")
                ? parsed_args["convert_filter"].as<std::string>()
                : "";
            auto format = parsed_args["convert"].as<cldnn::backward_comp::neural_memory::nnd_layout_format::type>();
            convert_weights(
                cldnn::backward_comp::neural_memory::to_cldnn_data_type_old(format),
                cldnn::backward_comp::neural_memory::to_cldnn_format(format),
                convert_filter);
            return 0;
        }

        // Execute network otherwise.
        execution_params ep;
        parse_common_options(parsed_args, ep);
        if (parsed_args.count("input"))
        {
            // Validate input directory.
            auto input_dir = parsed_args["input"].as<std::string>();
            if (!bfs::exists(input_dir) || !bfs::is_directory(input_dir))
            {
                std::cerr << "ERROR: specified input images path (\"" << input_dir
                    << "\") does not exist or does not point to directory (--input option invalid)!!!" << std::endl;
                return 1;
            }

            // Determine weights directory (either based on executable directory - if not specified, or
            // relative to current working directory or absolute - if specified).
            auto weights_dir = std::string(); 
            
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
                        << "\") does not exist or does not point to directory (--weights option invald)!!!" << std::endl;
                }
                else
                {
                    std::cerr << "ERROR: could not find default network weights path for selected topology. Neither '"
                        << parsed_args["model"].as<std::string>() << "' nor 'weights' folder exist!!!" << std::endl;
                }

                return 1;
            }

            ep.input_dir = input_dir;
            ep.weights_dir = weights_dir;
        }

        ep.topology_name = parsed_args["model"].as<std::string>();
        ep.calibration = parsed_args["use_calibration"].as<bool>();

        if (ep.topology_name == "alexnet" ||
            ep.topology_name == "vgg16" ||
            ep.topology_name == "vgg16_face" ||
            ep.topology_name == "googlenet" ||
            ep.topology_name == "gender" ||
            ep.topology_name == "squeezenet" ||
            ep.topology_name == "resnet50" ||
            ep.topology_name == "resnet50-i8" ||
            ep.topology_name == "ssd_mobilenet" ||
            ep.topology_name == "ssd_mobilenet-i8")
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
