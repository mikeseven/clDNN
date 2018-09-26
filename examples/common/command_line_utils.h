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

#pragma once

#include "common_tools.h"
#include "neural_memory.h"
#include <boost/program_options.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <iterator>
#include <memory>
#include <regex>
#include <stdexcept>
#include <string>
#include <sstream>
#include <type_traits>
#include <vector>


namespace cldnn
{
namespace utils
{
namespace examples
{
namespace cmdline
{
// --------------------------------------------------------------------------------------------------------------------
// Command-line parsing.
// --------------------------------------------------------------------------------------------------------------------

    namespace bpo = boost::program_options;

// --------------------------------------------------------------------------------------------------------------------
// External identifiers.

/// Application version (should be provided by every example-like application).
extern const unsigned long app_version;
// Application help message (should be provided by every example-like application).
//extern std::string help_message;

// --------------------------------------------------------------------------------------------------------------------
// Helper classes.

    bpo::options_description& add_common_options(bpo::options_description& options);
    bpo::options_description& add_version_help_options(bpo::options_description& options);
    std::ostringstream print_version_message(const std::shared_ptr<const executable_info>& exec_info);
    std::ostream& print_help_message(std::ostream& help_msg_out, const std::shared_ptr<const executable_info>& exec_info);

    /// @brief Helper class that provides elements for command-line parsing and message presentation.
    class cmdline_options
    {
        std::shared_ptr<const bpo::options_description> _all_options;
        std::string _help_message;
        std::string _version_message;

    public:
        /// @brief Type of formatter function that can be specified to format help message (usage header).
        using help_msg_formatter_type = std::ostream& (std::ostream&, const std::shared_ptr<const executable_info>&);


        /// @brief Default line length used when formatting options.
        static constexpr unsigned default_line_length      = 120;
        /// @brief Default minimum relative size of description during option formatting.
        static constexpr float default_desc_rel_min_length = 0.66f;


        /// @brief Options group.
        ///
        /// @details This class wraps boost::program_options::options_description to work around limitation
        ///          with regulating size of line once description was created.
        /// @n
        /// @n       Please note that the nested groups are not supported (options description is copied
        ///          to new one during command-line processing).
        /// @n
        /// @n       Base class does not have virtual destructor, so use only via std::shared_ptr
        ///          or similar "deduced-deleter" functionality.
        class opt_group: public bpo::options_description
        {
            std::string _caption;

        public:
            /// @brief Gets caption for current options group.
            const std::string& caption() const
            {
                return _caption;
            }

            /// @brief Creates new instance of options group.
            ///
            /// @details Please DO NOT use this constructor directly. Instead use cmdline_options::create_group().
            ///
            /// @param caption Caption for options group.
            explicit opt_group(const std::string& caption)
                : options_description(caption), _caption(caption)
            {}
        };


        /// @brief Gets description of all options that will be parsed by command-line parser.
        const boost::program_options::options_description& all_options() const
        {
            return *_all_options;
        }

        /// @brief Gets help message for command-line.
        const std::string& help_message() const
        {
            return _help_message;
        }

        /// @brief Gets version message for command-line.
        const std::string& version_message() const
        {
            return _version_message;
        }


        cmdline_options(const std::shared_ptr<const executable_info>& exec_info,
                        const std::shared_ptr<const opt_group>& std_options_group,
                        const bool add_examples_common_options                              = true,
                        const std::function<help_msg_formatter_type>& help_msg_formatter    = {},
                        const std::vector<std::shared_ptr<const opt_group>>& options_groups = {},
                        const std::vector<std::shared_ptr<const opt_group>>& visible_groups = {})
            : cmdline_options(exec_info, std_options_group, default_line_length, default_desc_rel_min_length,
                              add_examples_common_options, help_msg_formatter, options_groups, visible_groups)
        {}

        cmdline_options(const std::shared_ptr<const executable_info>& exec_info,
                        const std::shared_ptr<const opt_group>& std_options_group,
                        const unsigned line_length,
                        const float desc_rel_min_length              = default_desc_rel_min_length,
                        const bool add_examples_common_options                              = true,
                        const std::function<help_msg_formatter_type>& help_msg_formatter    = {},
                        const std::vector<std::shared_ptr<const opt_group>>& options_groups = {},
                        const std::vector<std::shared_ptr<const opt_group>>& visible_groups = {})
        {
            if (std_options_group == nullptr)
                throw std::invalid_argument("Standard options group was not specified (null).");

            // Calculate line length / description minimum length.
            const auto rel_min_length = std::max(0.1f, std::min(desc_rel_min_length, 0.9f)); 
            auto min_desc_length = static_cast<unsigned>(std::round(line_length * rel_min_length)); 

            // Prepare standard options.
            auto extended_std_options_group = apply_formatting(std_options_group, line_length, min_desc_length);
            if (add_examples_common_options)
                add_common_options(*extended_std_options_group);
            add_version_help_options(*extended_std_options_group);

            // All options.
            std::vector<std::shared_ptr<bpo::options_description>> formatted_options_groups;
            auto all_cmdline_options = std::make_shared<bpo::options_description>(line_length, min_desc_length);
            if (options_groups.empty())
                all_cmdline_options->add(*extended_std_options_group);
            else
            {
                auto std_options_group_found = false;
                for (const auto& options_group : options_groups)
                {
                    if (options_group == nullptr)
                        throw std::invalid_argument("One of options groups is not specified (null).");

                    if (options_group == std_options_group)
                    {
                        formatted_options_groups.push_back(extended_std_options_group);
                        std_options_group_found = true;
                    }
                    else
                    {
                        formatted_options_groups.push_back(
                            apply_formatting(options_group, line_length, min_desc_length));
                    }
                    all_cmdline_options->add(*formatted_options_groups.back());
                }
                if (!std_options_group_found)
                    throw std::invalid_argument("Standard options group was not included "
                                                "in list of options groups.");
            }

            // All visible options.
            std::shared_ptr<bpo::options_description> all_visible_cmdline_options;
            if (visible_groups.empty())
                all_visible_cmdline_options = all_cmdline_options;
            else
            {
                all_visible_cmdline_options = std::make_shared<bpo::options_description>(line_length, min_desc_length);
                for (const auto& visible_group : visible_groups)
                {
                    if (visible_group == nullptr)
                        throw std::invalid_argument("One of visible options groups is not specified (null).");

                    auto found_group = std::find(options_groups.cbegin(), options_groups.cend(), visible_group);
                    if (found_group == options_groups.cend())
                        throw std::invalid_argument("One of visible options groups is not added to options groups.");

                    const auto found_group_idx = static_cast<decltype(formatted_options_groups)::size_type>(
                        std::distance(options_groups.begin(), found_group));
                    all_visible_cmdline_options->add(*formatted_options_groups.at(found_group_idx));
                }
            }

            auto version_msg_out = print_version_message(exec_info);

            std::ostringstream help_msg_out;
            if (help_msg_formatter)
                // ReSharper disable once CppExpressionWithoutSideEffects
                help_msg_formatter(help_msg_out, exec_info);
            else
                print_help_message(help_msg_out, exec_info);
            help_msg_out << "\n\n" << *all_visible_cmdline_options;

            _all_options     = std::move(all_cmdline_options);
            _help_message    = help_msg_out.str();
            _version_message = version_msg_out.str();
        }

        /// @brief Creates instance of helper class.
        ///
        /// @param all_options     All options that will be parsed by parser.
        /// @param help_message    Application help message (temporary).
        /// @param version_message Application version message (temporary).
        cmdline_options(const boost::program_options::options_description& all_options,
                        std::string&& help_message, std::string&& version_message)
            : _all_options(std::make_shared<bpo::options_description>(all_options)),
              _help_message(std::move(help_message)),
              _version_message(std::move(version_message))
        {}


        /// @brief Creates instance of options group.
        ///
        /// @param caption Caption for options group.
        /// @return Options group (empty; as shared pointer).
        static std::shared_ptr<opt_group> create_group(const std::string& caption)
        {
            return std::make_shared<opt_group>(caption);
        }

    private:
        /// @brief Applies formatting for selected options group.
        ///
        /// @param options_group   Options group that will be extended with formatting.
        /// @param line_length     Line length to which options help information should be formatted.
        /// @param desc_min_length Minimum length of description of each option (text part after option
        ///                        name / specification). The value must be lower @p than line_length.
        /// @return                Options description that describe options group with formatting applied.
        static std::shared_ptr<bpo::options_description> apply_formatting(
            const std::shared_ptr<const opt_group>& options_group,
            const unsigned line_length, const unsigned desc_min_length)
        {
            assert(options_group != nullptr && "Options group is empty (null)!!!");

            auto group = std::make_shared<bpo::options_description>(options_group->caption(),
                                                                    line_length, desc_min_length);
            for (const auto& option : options_group->options())
                group->add(option);
            return group; // NRVO
        }
    };

    /// Parses command-line options.
    ///
    /// Throws exception on parse errors.
    ///
    /// @param options Options helper class with all options and basic messages.
    /// @param argc    Main function arguments count.
    /// @param argv    Main function argument values.
    ///
    /// @return Variable map with parsed options.
    inline boost::program_options::variables_map parse_cmdline_options(const cmdline_options& options,
                                                                       const int argc, const char* const argv[])
    {
        namespace bpo = boost::program_options;

        bpo::variables_map vars_map;
        store(parse_command_line(argc, argv, options.all_options()), vars_map);
        notify(vars_map);

        return vars_map;
    }

    /// @brief Provides handling for --help and --version options inside applications.
    ///
    /// @param parsed_args Pre-parsed arguments provided to application that will be interpreted.
    /// @param options     List of options supported by application.
    /// @return            @c true if parsed arguments contain --help or --version
    ///                    switches and the function provided handling for it;
    ///                    otherwise, @false.
    inline bool parse_version_help_options(bpo::variables_map& parsed_args, const cmdline_options& options)
    {
        if (parsed_args.count("help"))
        {
            std::cout << options.version_message() << "\n\n";
            std::cout << options.help_message() << std::endl;
            return true;
        }
        if (parsed_args.count("version"))
        {
            std::cout << options.version_message() << std::endl;
            return true;
        }
        return false;
    }

    /// @brief Provides handling for common clDNN-oriented options inside application.
    ///
    /// @param parsed_args Pre-parsed arguments provided to application that will be interpreted.
    /// @param exec_params Execution params structure (or derived structure) which will be
    ///                    filled with parsed parameters.
    /// @return            @c true if function provided handling for common parameters;
    ///                    otherwise, @c false.
    inline bool parse_common_options(const bpo::variables_map &parsed_args, execution_params& exec_params)
    {
        exec_params.batch                    = parsed_args["batch"].as<std::uint32_t>();
        exec_params.loop                     = parsed_args["loop"].as<std::uint32_t>();

        exec_params.run_until_primitive_name = parsed_args["run_until_primitive"].as<std::string>();
        exec_params.run_single_kernel_name   = parsed_args["run_single_layer"].as<std::string>();

        exec_params.dump_hidden_layers       = parsed_args["dump_hidden_layers"].as<bool>();
        exec_params.dump_layer_name          = parsed_args["dump_layer"].as<std::string>();
        exec_params.dump_single_batch        = parsed_args.count("dump_batch") != 0;
        exec_params.dump_batch_id            = exec_params.dump_single_batch
                                                   ? parsed_args["dump_batch"].as<std::uint32_t>()
                                                   : 0;
        exec_params.dump_single_feature      = parsed_args.count("dump_feature") != 0;
        exec_params.dump_feature_id          = exec_params.dump_single_feature
                                                   ? parsed_args["dump_feature"].as<std::uint32_t>()
                                                   : 0;
        exec_params.dump_graphs              = parsed_args["dump_graphs"].as<bool>();
        exec_params.dump_sources             = parsed_args["dump_sources"].as<bool>();
        exec_params.log_engine               = parsed_args["log_engine"].as<bool>();
        exec_params.profiling                = parsed_args["profiling"].as<bool>();
        exec_params.meaningful_kernels_names = parsed_args["meaningful_names"].as<bool>();
        exec_params.perf_per_watt            = parsed_args["perf_per_watt"].as<bool>();

        exec_params.use_half                 = parsed_args["use_half"].as<bool>();
        exec_params.optimize_weights         = parsed_args["optimize_weights"].as<bool>();
        exec_params.use_oooq                 = !parsed_args["no_oooq"].as<bool>();
        exec_params.disable_mem_pool         = parsed_args["memory_opt_disable"].as<bool>();
        exec_params.serialization            = parsed_args["serialization"].as<std::string>();
        exec_params.load_program             = parsed_args["load_program"].as<std::string>();

        exec_params.print_type               = parsed_args["print_type"].as<print_type>();;

        // Coercing of execution parameters.
        const std::regex dump_layer_wg_pattern(
            "^.+_weights\\.nnd$",
            std::regex_constants::ECMAScript | std::regex_constants::icase | std::regex_constants::optimize);
        exec_params.dump_weights = std::regex_match(exec_params.dump_layer_name, dump_layer_wg_pattern);

        if (!exec_params.run_single_kernel_name.empty())
            exec_params.meaningful_kernels_names = true;

        return true;
    }

    /// @brief Add common clDNN-oriented options to options set used by command-line parser.
    ///
    /// @param options Options set that will be extended with common options.
    inline bpo::options_description& add_common_options(bpo::options_description& options)
    {
        options.add_options()
            ("batch,b", bpo::value<std::uint32_t>()->value_name("<batch-size>")->default_value(1),
                "Size of a group of images that are classified together (large batch sizes have better performance; "
                "default: 1).")
            ("loop,l", bpo::value<std::uint32_t>()->value_name("<loop-count>")->default_value(1),
                "Number of iterations to run each execution. Can be used for robust benchmarking (default: 1).\n\n"
                "For character-model topologies, it says how many characters will be predicted.")
            ("engine,e", bpo::value<engine_types>()->value_name("<eng-type>")->default_value(engine_types::ocl, "gpu"),
                "Type of an engine used for classification.\nIt can be one of:\n  \tgpu.\n"
                "NOTE: This option is obsolete and it is not used anymore (provided for compatibility).")

            ("run_until_primitive", bpo::value<std::string>()->value_name("<primitive_name>")->default_value(""),
                "Run topology until specified primitive name / identifier.")
            ("run_single_layer", bpo::value<std::string>()->value_name("<primitive_name>")->default_value(""),
                "Run single layer / primitive from the topology specified by name / identifier provided "
                "by this option.")

            ("dump_hidden_layers,D", bpo::bool_switch(),
                "Dump results from hidden layers / primitives of a network to files.")
            ("dump_layer", bpo::value<std::string>()->value_name("<layer-name>")->default_value(""),
                "Dump results of specified network layer / primitive (or weights of the layer / primitive "
                "if name is in following format: \"<layer-name>_weights.nnd\") to files.")
            ("dump_batch", bpo::value<std::uint32_t>()->value_name("<batch-id>"),
                "Dump results only for specified batch (0-based).")
            ("dump_feature", bpo::value<std::uint32_t>()->value_name("<feature-id>"),
                "Dump results only for specified feature (0-based).")
            ("dump_graphs", bpo::bool_switch(),
                "Dump information about stages of a graph compilation to files in .graph format.\n"
                "Dump information about primitives in a graph to .info format.\n"
                "GraphViz is needed for converting .graph files to PDF format (an example command line "
                "in \"cldnn_dumps\" folder:\n  \tdot -Tpdf cldnn_program_1_0_init.graph -o outfile.pdf\n).")
            ("dump_sources", bpo::bool_switch(),
                "Dump engine source code per compilation. Currently only OpenCL sources are supported.")
            ("log_engine", bpo::bool_switch(),
                "Log engine actions during execution of a network.")
            ("profiling,p", bpo::bool_switch(),
                "Turn on profiling and create profiling report.")
            ("meaningful_names", bpo::bool_switch(),
                "Use kernels names derived from primitives identifiers / names for easier identification "
                "while profiling.\n"
                "NOTE: This may disable caching and significantly increase compilation time as well as binary size!")
            ("perf_per_watt", bpo::bool_switch(),
                "Turn on measurement of power consumption and output information about it "
                "(in frames per second per watt).")

            ("use_half", bpo::bool_switch(),
                "Use half-precision floating-point numbers (FP16, halfs) instead of single-precision ones (float) in "
                "computations of selected model.")
            ("optimize_weights", bpo::bool_switch(),
                "Perform weights conversion to most desirable format for each network layer / primitive while "
                "building network.")
            ("no_oooq", bpo::bool_switch(),
                "Do not use out-of-order queue for OpenCL engine.")
            ("memory_opt_disable", bpo::bool_switch(),
                "Disable memory reuse within primitives / layers.")
            ("serialization", bpo::value<std::string>()->value_name("<network-name>")->default_value(""),
                "Serialize currently built network and save it by specified name. Provide name for serialization "
                "process.")
            ("load_program", bpo::value<std::string>()->value_name("<network-name>")->default_value(""),
                "Load serialized network instead of building one. Provide name of load_program process.")

            ("print_type", bpo::value<print_type>()->value_name("<print-type>")->default_value(print_type::verbose,
                                                                                               "verbose"),
                "Configure volume of printing outputted from application. The following values are supported:\n"
                "  0, \"verbose\"          - \tVerbose output (the default).\n"
                "  1, \"performance\"      - \tOnly output performance results.\n"
                "  2, \"extended-testing\" - \tOutput topology primitives description, indication about "
                "wrong / correct results (used for broad correctness testing).");

        return options;
    }

    /// @brief Add common help and version options to options set used by command-line parser.
    ///
    /// @param options Options set that will be extended with help / version options.
    inline bpo::options_description& add_version_help_options(bpo::options_description& options)
    {
        options.add_options()
            ("version,v", "Show version of the application.")
            ("help,h",    "Show help message and available command-line options.");

        return options;
    }

    /// @brief Creates output string stream and writes to it version message.
    ///
    /// @param exec_info Information about executable.
    /// @return          String stream with message.
    inline std::ostringstream print_version_message(const std::shared_ptr<const executable_info>& exec_info)
    {
        // ------------------------------------------------------------------------------------------------------------
        // Version message.
        const auto ver = cldnn::get_version();

        std::ostringstream version_msg_out;
        version_msg_out << exec_info->file_name_wo_ext() << "   (version: "
            << (app_version >> 16) << "." << ((app_version >> 8) & 0xFF) << "." << (app_version & 0xFF)
            << "; clDNN version: "
            << ver.major << "." << ver.minor << "." << ver.build << "." << ver.revision
            << ")";

        return version_msg_out; // NRVO / move
    }

    /// @brief Creates output string stream and writes to it help message (header).
    ///
    /// @param [out] help_msg_out Stream to which help message will be written.
    /// @param exec_info          Information about executable.
    /// @return                   Stream with help message.
    inline std::ostream& print_help_message(std::ostream& help_msg_out, const std::shared_ptr<const executable_info>& exec_info)
    {
        // ------------------------------------------------------------------------------------------------------------
        // Help message.
        help_msg_out << "Usage:\n  " << exec_info->file_name_wo_ext() << " [standard options]\n";
        help_msg_out << "  " << exec_info->file_name_wo_ext() << " [weights conversion options]\n\n";
        help_msg_out << "Executes classification on specified neural network (standard options),\n";
        help_msg_out << "or converts network weights (weights conversion options).\n\n";
        help_msg_out << "When conversion options are specified execution options are ignored.";

        return help_msg_out;
    }
} // namespace cmdline
} // namespace examples
} // namespace utils

// --------------------------------------------------------------------------------------------------------------------
// ADL-enabled parser / validators for some command-line options.

    /// ADL-accessible validation/parsing route for cldnn::neural_memory::format::type enum.
    ///
    /// The last two parameters represents match type for validator (pointer to matched type) and
    /// int type (to properly order overloads).
    ///
    /// @param [in, out] out_var Output variable where parsing/verification result will be stored.
    /// @param values           Input strings representing tokens with values for specific out_var variable.
    ///
    /// @exception boost::program_options::validation_error Parsing/validation failed on value of an option.
    inline void validate(boost::any& out_var, const std::vector<std::string>& values, engine_types*, int)
    {
        namespace bpo = boost::program_options;

        bpo::validators::check_first_occurrence(out_var);
        const auto& value = bpo::validators::get_single_string(values);

        const std::regex val_gpu_pattern("^gpu$",
            std::regex_constants::ECMAScript | std::regex_constants::icase | std::regex_constants::optimize);

        if (std::regex_match(value, val_gpu_pattern))
            out_var = boost::any(cldnn::engine_types::ocl);
        else
            throw bpo::invalid_option_value(value);
    }

namespace backward_comp
{
    /// ADL-accessible validation/parsing route for cldnn::neural_memory::format::type enum.
    ///
    /// The last two parameters represents match type for validator (pointer to matched type) and
    /// int type (to properly order overloads).
    ///
    /// @param [in, out] out_var Output variable where parsing/verification result will be stored.
    /// @param values           Input strings representing tokens with values for specific out_var variable.
    ///
    /// @exception boost::program_options::validation_error Parsing/validation failed on value of an option.
    inline void validate(boost::any& out_var, const std::vector<std::string>& values,
        neural_memory::nnd_layout_format::type*, int)
    {
        namespace bpo = boost::program_options;

        bpo::validators::check_first_occurrence(out_var);
        const auto& value = bpo::validators::get_single_string(values);

        const std::regex val_numeric_pattern(
            "^[0-9]+$",
            std::regex_constants::ECMAScript | std::regex_constants::icase | std::regex_constants::optimize);

        // Underlying type of cldnn::neural_memory::format::type.
        using format_ut = std::underlying_type_t<neural_memory::nnd_layout_format::type>;
        // Type used for conversion/lexical_cast for neural_memory::nnd_layout_format::type (to avoid
        // problems with char types in lexical_cast).
        using format_pt = std::common_type_t<format_ut, unsigned>;
        if (std::regex_match(value, val_numeric_pattern))
        {
            try
            {
                out_var = boost::any(static_cast<neural_memory::nnd_layout_format::type>(
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

namespace utils
{
namespace examples
{
    /// ADL-accessible validation/parsing route for cldnn::neural_memory::format::type enum.
    ///
    /// The last two parameters represents match type for validator (pointer to matched type) and
    /// int type (to properly order overloads).
    ///
    /// @param [in, out] out_var Output variable where parsing/verification result will be stored.
    /// @param values           Input strings representing tokens with values for specific out_var variable.
    ///
    /// @exception boost::program_options::validation_error Parsing/validation failed on value of an option.
    inline void validate(boost::any& out_var, const std::vector<std::string>& values,
        print_type*, int)
    {
        static_assert(static_cast<unsigned>(print_type::_enum_count) == 3,
            "Some cases from print_type enum are not handled. Please provide handling in validate().");

        namespace bpo = boost::program_options;

        bpo::validators::check_first_occurrence(out_var);
        const auto& value = bpo::validators::get_single_string(values);

        const std::regex val_numeric_pattern(
            "^(?:[0-9]+|verbose|performance|extended[_\\-]?testing)$",
            std::regex_constants::ECMAScript | std::regex_constants::icase | std::regex_constants::optimize);

        // Underlying type of cldnn::neural_memory::format::type.
        using format_ut = std::underlying_type_t<print_type>;
        // Type used for conversion/lexical_cast for print_type (to avoid problems with char types
        // in lexical_cast).
        using format_pt = std::common_type_t<format_ut, unsigned>;
        if (std::regex_match(value, val_numeric_pattern))
        {
            try
            {
                if (value[0] == 'V' || value[0] == 'v')
                    out_var = boost::any(print_type::verbose);
                else if (value[0] == 'P' || value[0] == 'p')
                    out_var = boost::any(print_type::performance);
                else if (value[0] == 'E' || value[0] == 'e')
                    out_var = boost::any(print_type::extended_testing);
                else
                {
                    auto enum_num_val = boost::numeric_cast<format_ut>(boost::lexical_cast<format_pt>(value));
                    if (enum_num_val >= static_cast<format_ut>(print_type::_enum_count))
                        throw bpo::invalid_option_value(value);

                    out_var = boost::any(static_cast<print_type>(enum_num_val));
                }
            }
            catch (...)
            {
                throw bpo::invalid_option_value(value);
            }
        }
        else
            throw bpo::invalid_option_value(value);
    }
} // namespace examples
} // namespace utils
} // namespace cldnn
