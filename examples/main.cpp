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
    /// ADL-accessible validation/parsing route for cldnn::neural_memory::nnd_layout_format::type enum.
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

        // Underlying type of cldnn::neural_memory::nnd_layout_format::type.
        using format_ut = std::underlying_type_t<backward_comp::neural_memory::nnd_layout_format::type>;
        // Type used for conversion/lexical_cast for cldnn::neural_memory::nnd_layout_format::type (to avoid problems with char types
        // in lexical_cast).
        using format_pt = std::common_type_t<format_ut, unsigned>;
        if (std::regex_match(value, val_numeric_pattern))
        {
            try
            {
                auto parsed_val = static_cast<backward_comp::neural_memory::nnd_layout_format::type>(
                    boost::numeric_cast<format_ut>(boost::lexical_cast<format_pt>(value)));

                if (!backward_comp::neural_memory::nnd_layout_format::is_supported(parsed_val))
                    throw bpo::invalid_option_value(value);

                outVar = boost::any(parsed_val);
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
		("serialization", bpo::value<std::string>()->value_name("<network name>")->default_value(""),
			"Name for serialization process.")
        ("batch", bpo::value<std::uint32_t>()->value_name("<batch-size>")->default_value(1),
            "Size of a group of images that are classified together (large batch sizes have better performance).")
        ("loop", bpo::value<std::uint32_t>()->value_name("<loop-count>")->default_value(1),
            "Number of iterations to run each execution. Can be used for robust benchmarking. (default 1).\n"
            "For character-model topos it says how many characters will be predicted.")
        ("model", bpo::value<std::string>()->value_name("<model-name>")->default_value("alexnet"),
            "Name of a neural network model that is used for classification.\n"
            "It can be one of:\n  \talexnet, vgg16, vgg16_face, googlenet, gender, squeezenet, resnet50, resnet50-i8, microbench_conv, microbench_lstm, ssd_mobilenet, ssd_mobilenet-i8, lstm_char, lenet.")
        ("run_until_primitive", bpo::value<std::string>()->value_name("<primitive_name>"),
            "Runs topology until specified primitive.")
        ("run_single_layer", bpo::value<std::string>()->value_name("<primitive_name>"),
            "Run single layer from the topology. Provide ID of that layer.\n")
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
        ("dump_graphs", bpo::bool_switch(),
            "Dump informations about stages of graph compilation to files in .graph format.\n"
            "Dump informations about primitives in graph to .info format.\n"
            "GraphViz is needed for converting .graph files to pdf format. (example command line in cldnn_dumps folder: dot -Tpdf cldnn_program_1_0_init.graph -o outfile.pdf\n")
		("log_engine", bpo::bool_switch(),
            "Log engine actions during execution of a network.")
        ("dump_sources", bpo::bool_switch(),
            "Dump ocl source code per compilation.")
        ("weights", bpo::value<std::string>()->value_name("<weights-dir>"),
            "Path to directory containing weights used in classification.\n"
            "Non-absolute paths are computed in relation to <executable-dir> (not working directory).\n"
            "If not specified, the \"<executable-dir>/<model-name>\" path is used in first place with \"<executable-dir>/weights\" as fallback.")
        ("use_half", bpo::bool_switch(),
            "Uses half precision floating point numbers (FP16, halfs) instead of single precision ones (float) in "
            "computations of selected model.")
        ("use_calibration", bpo::bool_switch(),
            "Uses int8 precision and output calibration. Supported topologies: squeezenet")
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
        ("sequence_length", bpo::value<std::uint32_t>()->value_name("<sequence-length>")->default_value(0),
            "Used in RNN topologies (LSTM).")
        ("perf_per_watt", bpo::bool_switch(),
            "Triggers power consumption measuring and outputing frames per second per watt.")
        ("vocabulary_file", bpo::value<std::string>()->value_name("<vocabulary-file>"),
            "Path to vocabulary file (.txt format). File has to contain all the characters, which model recognizes.")
        ("temperature", bpo::value<float>()->value_name("<temperature-value>")->default_value(0.0f),
            "Temperature for character selection at the output <range from 0.0 to 1.0>. (default 0.0f)")        
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
            "LSTM use initial cell tensor.")
        ("image_number", bpo::value<std::uint32_t>()->value_name("<image_number>")->default_value(8),
            "Number of images that will be used for traning. Default value is 8.")
        ("image_offset", bpo::value<std::uint32_t>()->value_name("<image_offset>")->default_value(0),
            "How many images should be skipped in mnist data on execution.")
        ("use_existing_weights", bpo::bool_switch(),
            "Parameter used in learning, when it is set then model will use existing weights files")
        ("lr", bpo::value<float>()->value_name("<lr>")->default_value(0.00001f),
            "Base learning rate for network training. Default value is 0.00001f.")
        ("version", "Show version of the application.")
        ("help", "Show help message and available command-line options.");

    // Conversions options.
    bpo::options_description weights_conv_cmdline_options("Weights conversion options");
    weights_conv_cmdline_options.add_options()
        ("convert", bpo::value<cldnn::backward_comp::neural_memory::nnd_layout_format::type>()->value_name("<nnd_layout_format-type>"),
            "Convert weights of a neural network to given nnd_layout_format (<nnd_layout_format-type> represents numeric value of "
            "cldnn::neural_memory::nnd_layout_format enum).")
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
    auto ver = cldnn::get_version();

    std::ostringstream version_msg_out;
    version_msg_out << exec_info->file_name_wo_ext() << "   (version: "
        << (app_version >> 16) << "." << ((app_version >> 8) & 0xFF) << "." << (app_version & 0xFF)
        << "; clDNN version: "
        << ver.major << "." << ver.minor << "." << ver.build << "." << ver.revision
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
    bool microbench_conv = false;
    bool microbench_lstm = false;
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
        microbench_conv = parsed_args["model"].as<std::string>() == "microbench_conv";
        microbench_lstm = parsed_args["model"].as<std::string>() == "microbench_lstm";
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
        if (!parsed_args.count("input") && !parsed_args.count("convert") && !microbench_conv && !microbench_lstm)
        {
            std::cerr << "ERROR: none of required options was specified (either --input or microbench_conv or microbench_lstm\n";
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
            
            if (!microbench_conv && !microbench_lstm) // don't need weights for microbench_conv or microbench_lstm.
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

            auto seq_length = parsed_args["sequence_length"].as<std::uint32_t>();;
            if (seq_length != 0) ep.rnn_type_of_topology = true;
            ep.sequence_length = seq_length;
            if (ep.rnn_type_of_topology && parsed_args["model"].as<std::string>() != "lstm_char")
                std::cerr << "lstm type params are allowed for lstm type topologies" << std::endl;;
            if (seq_length > 0)
            {
                auto vocab_file = parsed_args["vocabulary_file"].as<std::string>();
                if (!bfs::exists(vocab_file) || !bfs::is_regular_file(vocab_file) || bfs::extension(vocab_file) != ".txt")
                {
                    std::cerr << "ERROR: specified vocabulary file (\"" << vocab_file
                        << "\") does not exist or is not .txt type of file !!!" << std::endl;
                    return 1;
                }
                ep.vocabulary_file = vocab_file;
            }

            ep.input_dir = input_dir;
            ep.weights_dir = weights_dir;
        }
        else if (microbench_conv || microbench_lstm)
        {
            ep.input_dir = "NA";
            ep.weights_dir = "NA";
        }

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

        ep.topology_name = parsed_args["model"].as<std::string>();
		ep.serialization = parsed_args["serialization"].as<std::string>();
        ep.batch = parsed_args["batch"].as<std::uint32_t>();
        ep.meaningful_kernels_names = parsed_args["meaningful_names"].as<bool>();
        ep.profiling = parsed_args["profiling"].as<bool>();
        ep.optimize_weights = parsed_args["optimize_weights"].as<bool>();
        ep.use_half = parsed_args["use_half"].as<bool>();
        ep.use_oooq = !parsed_args["no_oooq"].as<bool>();
        ep.run_until_primitive_name = run_until_primitive;
        ep.run_single_kernel_name = single_layer_name;
        ep.dump_hidden_layers = parsed_args["dump_hidden_layers"].as<bool>();
        ep.dump_layer_name = dump_layer;
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
        ep.calibration = parsed_args["use_calibration"].as<bool>();
        ep.sequence_length = parsed_args["sequence_length"].as<std::uint32_t>();
        ep.temperature = parsed_args["temperature"].as<float>();
        ep.image_number = parsed_args["image_number"].as<std::uint32_t>();
        ep.image_offset = parsed_args["image_offset"].as<std::uint32_t>();
        ep.use_existing_weights = parsed_args["use_existing_weights"].as<bool>();
        ep.learning_rate = parsed_args["lr"].as<float>();

        if (ep.rnn_type_of_topology) //we care about temperature for some rnn topologies.
        {
            if (ep.temperature > 1.0f)
            {
                std::cout << "[WARNING] Temperature too high. Lowered to max value = 1.0f." << std::endl;
                ep.temperature = 1.0f;
            }
            if (ep.temperature < 0.0f)
            {
                std::cout << "[WARNING] Temperature can't be negative. Higered to min. value = 0.0f." << std::endl;
                ep.temperature = 0.0f;
            }
        }

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

        if (ep.topology_name == "alexnet" ||
            ep.topology_name == "vgg16" ||
            ep.topology_name == "vgg16_train" ||
            ep.topology_name == "vgg16_face" ||
            ep.topology_name == "googlenet" ||
            ep.topology_name == "gender" ||
            ep.topology_name == "squeezenet" ||
            ep.topology_name == "lenet" ||
            ep.topology_name == "lenet_train" ||
            ep.topology_name == "resnet50" ||
            ep.topology_name == "resnet50-i8" ||
            ep.topology_name == "microbench_conv" ||
            ep.topology_name == "microbench_lstm" ||
            ep.topology_name == "lstm_char" || 
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
