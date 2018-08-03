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

#include "common_tools.h"
#include "neural_memory.h"
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
    namespace bpo = boost::program_options;

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

    uint16_t parse_help_version(bpo::variables_map &parsed_args, cmdline_options options)
    {
        if (parsed_args.count("help"))
        {
            std::cout << options.version_message() << "\n\n";
            std::cout << options.help_message() << std::endl;
            return 1;
        }
        if (parsed_args.count("version"))
        {
            std::cout << options.version_message() << std::endl;
            return 1;
        }
        return 0;
    }

    void parse_common_args(const bpo::variables_map &parsed_args, execution_params &ep)
    {
        std::string dump_layer = "";
        if (parsed_args.count("dump_layer"))
        {
            dump_layer = parsed_args["dump_layer"].as<std::string>();
            if (dump_layer.find("_weights.nnd") != std::string::npos)
                ep.dump_weights = true;
            else
                ep.dump_weights = false;
        }

        std::string run_until_primitive = "";
        if (parsed_args.count("run_until_primitive"))
        {
            run_until_primitive = parsed_args["run_until_primitive"].as<std::string>();
        }

        std::string single_layer_name = "";
        if (parsed_args.count("run_single_layer"))
        {
            single_layer_name = parsed_args["run_single_layer"].as<std::string>();
        }

        ep.run_until_primitive_name = run_until_primitive;
        ep.run_single_kernel_name = single_layer_name;
        ep.dump_layer_name = dump_layer;
        ep.serialization = parsed_args["serialization"].as<std::string>();
        ep.load_program = parsed_args["load_program"].as<std::string>();
        ep.batch = parsed_args["batch"].as<std::uint32_t>();
        ep.meaningful_kernels_names = parsed_args["meaningful_names"].as<bool>();
        ep.profiling = parsed_args["profiling"].as<bool>();
        ep.optimize_weights = parsed_args["optimize_weights"].as<bool>();
        ep.use_half = parsed_args["use_half"].as<bool>();
        ep.use_oooq = !parsed_args["no_oooq"].as<bool>();
        ep.dump_hidden_layers = parsed_args["dump_hidden_layers"].as<bool>();
        ep.dump_single_batch = parsed_args.count("dump_batch") != 0;
        ep.dump_batch_id = ep.dump_single_batch ? parsed_args["dump_batch"].as<uint32_t>() : 0;
        ep.dump_single_feature = parsed_args.count("dump_feature") != 0;
        ep.dump_feature_id = ep.dump_single_feature ? parsed_args["dump_feature"].as<uint32_t>() : 0;
        ep.dump_graphs = parsed_args["dump_graphs"].as<bool>();
        ep.log_engine = parsed_args["log_engine"].as<bool>();
        ep.dump_sources = parsed_args["dump_sources"].as<bool>();
        ep.perf_per_watt = parsed_args["perf_per_watt"].as<bool>();
        ep.loop = parsed_args["loop"].as<std::uint32_t>();
        ep.disable_mem_pool = parsed_args["memory_opt_disable"].as<bool>();

        std::uint32_t print = parsed_args["print_type"].as<std::uint32_t>();
        ep.print_type = (PrintType)((print >= (std::uint32_t)PrintType::PrintType_count) ? 0 : print);

        if (!ep.run_single_kernel_name.empty())
            ep.meaningful_kernels_names = true;
    }

    void add_base_options(bpo::options_description &options)
    {
        options.add_options()
            ("batch", bpo::value<std::uint32_t>()->value_name("<batch-size>")->default_value(1),
                "Size of a group of images that are classified together (large batch sizes have better performance).")
            ("loop", bpo::value<std::uint32_t>()->value_name("<loop-count>")->default_value(1),
                "Number of iterations to run each execution. Can be used for robust benchmarking. (default 1).\n"
                "For character-model topos it says how many characters will be predicted.")
            ("engine", bpo::value<cldnn::engine_types>()->value_name("<eng-type>")->default_value(cldnn::engine_types::ocl, "gpu"),
                "Type of an engine used for classification.\nIt can be one of:\n  \treference, gpu.")
            ("run_until_primitive", bpo::value<std::string>()->value_name("<primitive_name>"),
                "Runs topology until specified primitive.")
            ("run_single_layer", bpo::value<std::string>()->value_name("<primitive_name>"),
                "Run single layer from the topology. Provide ID of that layer.\n")
            ("dump_hidden_layers", bpo::bool_switch(),
                "Dump results from hidden layers of network to files.")
            ("dump_layer", bpo::value<std::string>()->value_name("<layer_name>"),
                "Dump results of specified network layer (or weights of the layer <name>_weights.nnd) to files.")
            ("dump_batch", bpo::value<std::uint32_t>()->value_name("<batch-id>"),
                "Dump results only for this specified batch.")
            ("dump_feature", bpo::value<std::uint32_t>()->value_name("<feature-id>"),
                "Dump results only for this specified feature.")
            ("dump_graphs", bpo::bool_switch(),
                "Dump informations about stages of graph compilation to files in .graph format.\n"
                "Dump informations about primitives in graph to .info format.\n"
                "GraphViz is needed for converting .graph files to pdf format. (example command line in cldnn_dumps folder: dot -Tpdf cldnn_program_1_0_init.graph -o outfile.pdf\n")
            ("log_engine", bpo::bool_switch(),
                "Log engine actions during execution of a network.")
            ("dump_sources", bpo::bool_switch(),
                "Dump ocl source code per compilation.")
            ("use_half", bpo::bool_switch(),
                "Uses half precision floating point numbers (FP16, halfs) instead of single precision ones (float) in "
                "computations of selected model.")
            ("no_oooq", bpo::bool_switch(),
                "Do not use out-of-order queue for ocl engine.")
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
            ("memory_opt_disable", bpo::bool_switch(),
                "Disables memory reuse within primitves.")
            ("perf_per_watt", bpo::bool_switch(),
                "Triggers power consumption measuring and outputing frames per second per watt.")
            ("serialization", bpo::value<std::string>()->value_name("<network name>")->default_value(""),
                "Name for serialization process.")
            ("load_program", bpo::value<std::string>()->value_name("<network name>")->default_value(""),
                "Name of load_program process.")
            ("version", "Show version of the application.")
            ("help", "Show help message and available command-line options.");
    }

    std::ostringstream print_version_message(const std::shared_ptr<const executable_info>& exec_info)
    {
        // ----------------------------------------------------------------------------------------------------------------
        // Version message.
        auto ver = cldnn::get_version();

        std::ostringstream version_msg_out;
        version_msg_out << exec_info->file_name_wo_ext() << "   (version: "
            << (app_version >> 16) << "." << ((app_version >> 8) & 0xFF) << "." << (app_version & 0xFF)
            << "; clDNN version: "
            << ver.major << "." << ver.minor << "." << ver.build << "." << ver.revision
            << ")";

        return version_msg_out;
    }

    std::ostringstream print_help_message(const std::shared_ptr<const executable_info>& exec_info)
    {
        // ----------------------------------------------------------------------------------------------------------------
        // Help message.
        std::ostringstream help_msg_out;
        help_msg_out << "Usage:\n  " << exec_info->file_name_wo_ext() << " [standard options]\n";
        help_msg_out << "  " << exec_info->file_name_wo_ext() << " [weights conversion options]\n\n";
        help_msg_out << "Executes classification on specified neural network (standard options),\n";
        help_msg_out << "or converts network weights (weights conversion options).\n\n";
        help_msg_out << "When conversion options are specified execution options are ignored.\n\n";

        return help_msg_out;
    }
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
    void validate(boost::any& outVar, const std::vector<std::string>& values, cldnn::backward_comp::neural_memory::nnd_layout_format::type*, int)
    {
        namespace bpo = boost::program_options;

        bpo::validators::check_first_occurrence(outVar);
        const auto& value = bpo::validators::get_single_string(values);

        std::regex val_numeric_pattern("^[0-9]+$",
            std::regex_constants::ECMAScript | std::regex_constants::icase | std::regex_constants::optimize);

        // Underlying type of cldnn::neural_memory::format::type.
        using format_ut = std::underlying_type_t<backward_comp::neural_memory::nnd_layout_format::type>;
        // Type used for conversion/lexical_cast for cldnn::neural_memory::format::type (to avoid problems with char types
        // in lexical_cast).
        using format_pt = std::common_type_t<format_ut, unsigned>;
        if (std::regex_match(value, val_numeric_pattern))
        {
            try
            {
                outVar = boost::any(static_cast<backward_comp::neural_memory::nnd_layout_format::type>(
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