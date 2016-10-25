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

// ADL-enabled parser / validators for some command-line options.
namespace neural
{
    /// int type (to properly order overloads).
    ///
    /// @param [in, out] outVar Output variable where parsing/verification result will be stored.
    /// @param values           Input strings representing tokens with values for specific outVar variable.
    ///
    /// @exception boost::program_options::validation_error Parsing/validation failed on value of an option.
    void validate(boost::any& outVar, const std::vector<std::string>& values, neural::engine::type*, int)
    {
        namespace bpo = boost::program_options;

        bpo::validators::check_first_occurrence(outVar);
        const auto& value = bpo::validators::get_single_string(values);

        std::regex val_gpu_pattern("^gpu$",
            std::regex_constants::ECMAScript | std::regex_constants::icase | std::regex_constants::optimize);
        std::regex val_ref_pattern("^(ref|reference)$",
            std::regex_constants::ECMAScript | std::regex_constants::icase | std::regex_constants::optimize);

        if (std::regex_match(value, val_gpu_pattern))
        {
            outVar = boost::any(neural::engine::gpu);
        }
        else if (std::regex_match(value, val_ref_pattern))
        {
            outVar = boost::any(neural::engine::reference);
        }
        else
            throw bpo::invalid_option_value(value);
    }

    /// ADL-accessible validation/parsing route for neural::memory::format::type enum.
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
    void validate(boost::any& outVar, const std::vector<std::string>& values, neural::memory::format::type*, int)
    {
        namespace bpo = boost::program_options;

        bpo::validators::check_first_occurrence(outVar);
        const auto& value = bpo::validators::get_single_string(values);

        std::regex val_numeric_pattern("^[0-9]+$",
            std::regex_constants::ECMAScript | std::regex_constants::icase | std::regex_constants::optimize);

        // Underlying type of neural::memory::format::type.
        using format_ut = std::underlying_type_t<neural::memory::format::type>;
        // Type used for conversion/lexical_cast for neural::memory::format::type (to avoid problems with char types
        // in lexical_cast).
        using format_pt = std::common_type_t<format_ut, unsigned>;
        if (std::regex_match(value, val_numeric_pattern))
        {
            try
            {
                outVar = boost::any(static_cast<neural::memory::format::type>(
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
} // namespace neural

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
        ("model", bpo::value<std::string>()->value_name("<model-name>")->default_value("alexnet"),
            "Name of a neural network model that is used for classification.\n"
            "It can be one of:\n  \talexnet, vgg16.")
        ("engine", bpo::value<neural::engine::type>()->value_name("<eng-type>")->default_value(neural::engine::reference, "reference"),
            "Type of an engine used for classification.\nIt can be one of:\n  \treference, gpu.")
        ("dump_hidden_layers", bpo::bool_switch(),
            "Dump results from hidden layers of network to files.")
        ("weights", bpo::value<std::string>()->value_name("<weights-dir>"),
            "Path to directory containing weights used in classification.\n"
            "Non-absolute paths are computed in relation to <executable-dir> (not working directory).\n"
            "If not specified, the \"<executable-dir>/weights\" path is used.")
        ("profiling", bpo::bool_switch(),
            "Enable profiling and create profiling report.")
        ("optimize_weights", bpo::bool_switch(),
            "Perform weights convertion to most desirable format for each network layer while building network.")
        ("version", "Show version of the application.")
        ("help", "Show help message and available command-line options.");

    // Conversions options.
    bpo::options_description weights_conv_cmdline_options("Weights conversion options");
    weights_conv_cmdline_options.add_options()
        ("convert", bpo::value<neural::memory::format::type>()->value_name("<format-type>"),
            "Convert weights of a neural network to given format (<format-type> represents numeric value of "
            "neural::memory::format enum).")
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


    return{ all_cmdline_options, help_msg_out.str(), version_msg_out.str() };
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

    // TODO: create header file for all examples
    extern void alexnet(uint32_t, std::string, const std::string&, bool, bool, bool);
    extern void vgg16(uint32_t, std::string, const std::string&, bool, bool, bool);
    extern void convert_weights(neural::memory::format::type, std::string);


    set_executable_info(argc, argv); // Must be set before using get_executable_info().

                                     // Parsing command-line and handling/presenting basic options.
    auto exec_info = get_executable_info();
    auto options = prepare_cmdline_options(exec_info);
    bpo::variables_map parsed_args;
    try
    {
        parsed_args = parse_cmdline_options(options, argc, argv);

        if (parsed_args.count("help"))
        {
            std::cerr << options.version_message() << "\n\n";
            std::cerr << options.help_message() << std::endl;
            return 1;
        }
        if (parsed_args.count("version"))
        {
            std::cerr << options.version_message() << std::endl;
            return 1;
        }
        if (!parsed_args.count("input") && !parsed_args.count("convert"))
        {
            std::cerr << "ERROR: none of required options was specified (either --input or\n";
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
            convert_weights(parsed_args["convert"].as<neural::memory::format::type>(), convert_filter);
            return 0;
        }
        // Execute network otherwise.
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
            auto weights_dir = parsed_args.count("weights")
                ? bfs::absolute(parsed_args["weights"].as<std::string>(), exec_info->dir()).string()
                : join_path(exec_info->dir(), "weights");
            // Validate weights directory.
            if (!bfs::exists(weights_dir) || !bfs::is_directory(weights_dir))
            {
                std::cerr << "ERROR: specified network weights path (\"" << weights_dir
                    << "\") does not exist or does not point to directory (--weights option invald)!!!" << std::endl;
                return 1;
            }

            if (parsed_args["model"].as<std::string>() == "alexnet")
            {
                alexnet(
                    parsed_args["batch"].as<std::uint32_t>(),
                    input_dir,
                    weights_dir,
                    parsed_args["dump_hidden_layers"].as<bool>(),
                    parsed_args["profiling"].as<bool>(),
                    parsed_args["optimize_weights"].as<bool>());
                return 0;
            }
            else if (parsed_args["model"].as<std::string>() == "vgg16")
            {
                vgg16(
                    parsed_args["batch"].as<std::uint32_t>(),
                    input_dir,
                    weights_dir,
                    parsed_args["dump_hidden_layers"].as<bool>(),
                    parsed_args["profiling"].as<bool>(),
                    parsed_args["optimize_weights"].as<bool>());
                return 0;
            }

            std::cerr << "ERROR: model/topology (\"" << parsed_args["model"].as<std::string>()
                << "\") is not implemented!!!" << std::endl;
        }

        // No need for "else": We already handled when neither --input nor --convert is specified.
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
