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

#include "common/common_tools.h"
#include "neural_memory.h"
#include "api/CPP/memory.hpp"

#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/numeric/conversion/cast.hpp>
#include <boost/program_options.hpp>

#include <cstdint>
#include <iostream>
#include <regex>
#include <string>
#include <sstream>
#include <type_traits>


// --------------------------------------------------------------------------------------------------------------------
// Command-line parsing.
// --------------------------------------------------------------------------------------------------------------------

/// Application version.
static const unsigned long app_version = 0x10100L;


// Helper classes.
namespace
{
    /// Helper class that provides elements for command-line parsing and message presentation.
    class cmdline_options
    {
        boost::program_options::options_description _all_options;
        std::string _help_message;
        std::string _version_message;

    public:
        /// Gets description of all options that will be parsed by command-line parser.
        const boost::program_options::options_description& all_options() const
        {
            return _all_options;
        }

        /// Gets help message for command-line.
        const std::string& help_message() const
        {
            return _help_message;
        }

        /// Gets version message for command-line.
        const std::string& version_message() const
        {
            return _version_message;
        }


        /// Creates instance of helper class.
        ///
        /// @param all_options     All options that will be parsed by parser.
        /// @param help_message    Application help message (temporary).
        /// @param version_message Application version message (temporary).
        cmdline_options(const boost::program_options::options_description& all_options, std::string&& help_message, std::string&& version_message)
            : _all_options(all_options),
            _help_message(std::move(help_message)),
            _version_message(std::move(version_message))
        {}
    };
}

//// ADL-enabled parser / validators for some command-line options.
namespace cldnn
{
    /// int type (to properly order overloads).
    ///
    /// @param [in, out] outVar Output variable where parsing/verification result will be stored.
    /// @param values           Input strings representing tokens with values for specific outVar variable.
    ///
    /// @exception boost::program_options::validation_error Parsing/validation failed on value of an option.
    void validate(boost::any& outVar, const std::vector<std::string>& values, cldnn::engine_types*, int)
    {
        namespace bpo = boost::program_options;

        bpo::validators::check_first_occurrence(outVar);
        const auto& value = bpo::validators::get_single_string(values);

        std::regex val_gpu_pattern("^gpu$",
            std::regex_constants::ECMAScript | std::regex_constants::icase | std::regex_constants::optimize);

        if (std::regex_match(value, val_gpu_pattern))
        {
            outVar = boost::any(cldnn::engine_types::ocl);
        }
        else
            throw bpo::invalid_option_value(value);
    }

namespace backward_comp
{
    /// ADL-accessible validation/parsing route for cldnn::neural_memory::format::type enum.
    ///
    /// This function is specific to examples main command-line parser. Please do not move it to any
    /// headers (to avoid confict with different implementations in other source files).
    ///
    /// The last two parameters represents match type for validator (pointer to matched type) and
    /// int type (to properly order overloads).
    ///
    /// @param [in, out] outVar Output variable where parsing/verification result will be stored.
    /// @param values           Input strings representing tokens with values for specific outVar variable.
    ///
    /// @exception boost::program_options::validation_error Parsing/validation failed on value of an option.
    void validate(boost::any& outVar, const std::vector<std::string>& values, cldnn::backward_comp::neural_memory::format::type*, int)
    {
        namespace bpo = boost::program_options;

        bpo::validators::check_first_occurrence(outVar);
        const auto& value = bpo::validators::get_single_string(values);

        std::regex val_numeric_pattern("^[0-9]+$",
            std::regex_constants::ECMAScript | std::regex_constants::icase | std::regex_constants::optimize);

        // Underlying type of cldnn::neural_memory::format::type.
        using format_ut = std::underlying_type_t<backward_comp::neural_memory::format::type>;
        // Type used for conversion/lexical_cast for cldnn::neural_memory::format::type (to avoid problems with char types
        // in lexical_cast).
        using format_pt = std::common_type_t<format_ut, unsigned>;
        if (std::regex_match(value, val_numeric_pattern))
        {
            try
            {
                outVar = boost::any(static_cast<backward_comp::neural_memory::format::type>(
                    boost::numeric_cast<format_ut>(boost::lexical_cast<format_pt>(value))));
            }
            catch (...)
            {
                throw bpo::invalid_option_value(value);
            }
        }
        else
            throw bpo::invalid_option_value(value);
    }
} //namespace backward_comp
} // namespace cldnn

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
    standard_cmdline_options.add_options()
        ("input", bpo::value<std::string>()->value_name("<input-dir>"),
            "Path to input directory containing images to classify (mandatory when running classification).")
        ("batch", bpo::value<std::uint32_t>()->value_name("<batch-size>")->default_value(8),
            "Size of a group of images that are classified together (large batch sizes have better performance).")
        ("loop", bpo::value<std::uint32_t>()->value_name("<loop-count>")->default_value(1),
            "Number of iterations to run each execution. Can be used for robust benchmarking. (default 1)")
        ("model", bpo::value<std::string>()->value_name("<model-name>")->default_value("alexnet"),
            "Name of a neural network model that is used for classification.\n"
            "It can be one of:\n  \talexnet, vgg16, vgg16_face, googlenet, gender, squeezenet, microbench.")
        ("run_until_primitive", bpo::value<std::string>()->value_name("<primitive_name>"),
            "Runs topology until specified primitive.")
        ("engine", bpo::value<cldnn::engine_types>()->value_name("<eng-type>")->default_value(cldnn::engine_types::ocl, "gpu"),
            "Type of an engine used for classification.\nIt can be one of:\n  \treference, gpu.")
        ("dump_hidden_layers", bpo::bool_switch(),
            "Dump results from hidden layers of network to files.")
        ("dump_layer", bpo::value<std::string>()->value_name("<layer_name>"),
            "Dump results of specified network layer (or weights of the layer <name>_weights.nnd) to files.")
        ("dump_batch", bpo::value<std::uint32_t>()->value_name("<batch-id>"),
            "Dump results only for this specified batch.")
        ("dump_feature", bpo::value<std::uint32_t>()->value_name("<feature-id>"),
            "Dump results only for this specified feature.")
        ("weights", bpo::value<std::string>()->value_name("<weights-dir>"),
            "Path to directory containing weights used in classification.\n"
            "Non-absolute paths are computed in relation to <executable-dir> (not working directory).\n"
            "If not specified, the \"<executable-dir>/<model-name>\" path is used in first place with \"<executable-dir>/weights\" as fallback.")
        ("use_half", bpo::bool_switch(),
            "Uses half precision floating point numbers (FP16, halfs) instead of single precision ones (float) in "
            "computations of selected model.")
        ("meaningful_names", bpo::bool_switch(),
            "Use kernels' names derived from primitives' ids for easier identification while profiling.\n"
            "Note: this may disable caching and significantly increase compilation time as well as binary size!")
        ("profiling", bpo::bool_switch(),
            "Enables profiling and create profiling report.")
        ("print_type", bpo::value<std::uint32_t>()->value_name("<print_type>")->default_value(0),
            "0 = Verbose (default)\n"
            "1 = only print performance results\n"
            "2 = print topology primtives descritpion, print wrong/correct classification - used for broad correctness testing.")
        ("optimize_weights", bpo::bool_switch(),
            "Performs weights convertion to most desirable format for each network layer while building network.")
        ("perf_per_watt", bpo::bool_switch(),
            "Triggers power consumption measuring and outputing frames per second per watt.")
        ("version", "Show version of the application.")
        ("help", "Show help message and available command-line options.");

    // Conversions options.
    bpo::options_description weights_conv_cmdline_options("Weights conversion options");
    weights_conv_cmdline_options.add_options()
        ("convert", bpo::value<cldnn::backward_comp::neural_memory::format::type>()->value_name("<format-type>"),
            "Convert weights of a neural network to given format (<format-type> represents numeric value of "
            "cldnn::neural_memory::format enum).")
        ("convert_filter", bpo::value<std::string>()->value_name("<filter>"),
            "Name or part of the name of weight file(s) to be converted.\nFor example:\n"
            "  \"conv1\" - first convolution,\n  \"fc\" - every fully connected.");

    // All options.
    bpo::options_description all_cmdline_options;
    all_cmdline_options.add(standard_cmdline_options).add(weights_conv_cmdline_options);
    // All visible options.
    bpo::options_description all_visible_cmdline_options;
    all_visible_cmdline_options.add(standard_cmdline_options).add(weights_conv_cmdline_options);

    // ----------------------------------------------------------------------------------------------------------------
    // Version message.
    std::ostringstream version_msg_out;
    version_msg_out << exec_info->file_name_wo_ext() << "   (version: "
        << (app_version >> 16) << "." << ((app_version >> 8) & 0xFF) << "." << (app_version & 0xFF)
        << ")";

    // ----------------------------------------------------------------------------------------------------------------
    // Help message.
    std::ostringstream help_msg_out;
    help_msg_out << "Usage:\n  " << exec_info->file_name_wo_ext() << " [standard options]\n";
    help_msg_out << "  " << exec_info->file_name_wo_ext() << " [weights conversion options]\n\n";
    help_msg_out << "Executes classification on specified neural network (standard options),\n";
    help_msg_out << "or converts network weights (weights conversion options).\n\n";
    help_msg_out << "When conversion options are specified execution options are ignored.\n\n";
    help_msg_out << all_visible_cmdline_options;


    return {all_cmdline_options, help_msg_out.str(), version_msg_out.str()};
}

/// Parses command-line options.
///
/// Throws exception on parse errors.
///
/// @param options Options helper class with all options and basic messages.
/// @param argc    Main function arguments count.
/// @param argv    Main function argument values.
///
/// @return Variable map with parsed options.
static boost::program_options::variables_map parse_cmdline_options(
    const cmdline_options& options, int argc, const char* const argv[])
{
    namespace bpo = boost::program_options;

    bpo::variables_map vars_map;
    store(bpo::parse_command_line(argc, argv, options.all_options()), vars_map);
    notify(vars_map);

    return vars_map;
}

// --------------------------------------------------------------------------------------------------------------------

int main(int argc, char* argv[])
{
    namespace bpo = boost::program_options;
    namespace bfs = boost::filesystem;
    bool microbench = false;
    // TODO: create header file for all examples
    extern void alexnet(const execution_params &ep);
    extern void vgg16(const execution_params &ep);
    extern void googlenet_v1(const execution_params &ep);
    extern void convert_weights(cldnn::data_types dt, cldnn::format::type format, std::string);


    set_executable_info(argc, argv); // Must be set before using get_executable_info().

                                     // Parsing command-line and handling/presenting basic options.
    auto exec_info = get_executable_info();
    auto options = prepare_cmdline_options(exec_info);
    bpo::variables_map parsed_args;
    try
    {
        parsed_args = parse_cmdline_options(options, argc, argv);
        microbench = parsed_args["model"].as<std::string>() == "microbench";
        if (parsed_args.count("help"))
        {
            std::cerr << options.version_message() << "\n\n";
            std::cerr << options.help_message() << std::endl;
            return 0;
        }
        if (parsed_args.count("version"))
        {
            std::cerr << options.version_message() << std::endl;
            return 0;
        }
        if (!parsed_args.count("input") && !parsed_args.count("convert") && !microbench)
        {
            std::cerr << "ERROR: none of required options was specified (either --input or microbench\n";
            std::cerr << "       --convert is needed)!!!\n\n";
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

    // Execute network or convert weights.
    try
    {
        // Convert weights if convertion options are used.
        if (parsed_args.count("convert"))
        {
            auto convert_filter = parsed_args.count("convert_filter")
                ? parsed_args["convert_filter"].as<std::string>()
                : "";
            auto format = parsed_args["convert"].as<cldnn::backward_comp::neural_memory::format::type>();
            convert_weights(
                cldnn::backward_comp::neural_memory::to_data_type(format),
                cldnn::backward_comp::neural_memory::to_tensor_format(format),
                convert_filter);
            return 0;
        }

        // Execute network otherwise.
        execution_params ep;
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
            
            if (!microbench) // don't need weights for microbench.
            {
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
            }

            ep.input_dir = input_dir;
            ep.weights_dir = weights_dir;
        }
        else if (microbench)
        {
            ep.input_dir = "NA";
            ep.weights_dir = "NA";
        }

        std::string dump_layer = "";
        if (parsed_args.count("dump_layer"))
        {
            dump_layer = parsed_args["dump_layer"].as<std::string>();
            if (dump_layer.find("_weights.nnd"))
                ep.dump_weights = true;
            else
                ep.dump_weights = false;
        }


        std::string run_until_primitive = "";
        if (parsed_args.count("run_until_primitive"))
        {
            run_until_primitive = parsed_args["run_until_primitive"].as<std::string>();
        }
        ep.topology_name = parsed_args["model"].as<std::string>();
        ep.batch = parsed_args["batch"].as<std::uint32_t>();
        ep.meaningful_kernels_names = parsed_args["meaningful_names"].as<bool>();
        ep.profiling = parsed_args["profiling"].as<bool>();
        ep.optimize_weights = parsed_args["optimize_weights"].as<bool>();
        ep.use_half = parsed_args["use_half"].as<bool>();
        ep.run_until_primitive_name = run_until_primitive;
        ep.dump_hidden_layers = parsed_args["dump_hidden_layers"].as<bool>();
        ep.dump_layer_name = dump_layer;
        ep.dump_single_batch = parsed_args.count("dump_batch") != 0;
        ep.dump_batch_id = ep.dump_single_batch ? parsed_args["dump_batch"].as<uint32_t>() : 0;
        ep.dump_single_feature = parsed_args.count("dump_feature") != 0;
        ep.dump_feature_id = ep.dump_single_feature ? parsed_args["dump_feature"].as<uint32_t>() : 0;
        ep.perf_per_watt = parsed_args["perf_per_watt"].as<bool>();
        ep.loop = parsed_args["loop"].as<std::uint32_t>();

        std::uint32_t print = parsed_args["print_type"].as<std::uint32_t>();
        ep.print_type = (PrintType)((print >= (std::uint32_t)PrintType::PrintType_count) ? 0 : print);

        if (ep.topology_name == "alexnet" ||
            ep.topology_name == "vgg16" ||
            ep.topology_name == "vgg16_face" ||
            ep.topology_name == "googlenet" ||
            ep.topology_name == "gender" ||
            ep.topology_name == "squeezenet" ||
            ep.topology_name == "microbench")
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
