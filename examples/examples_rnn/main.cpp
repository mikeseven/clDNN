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
#include "command_line_utils.h"
#include "lstm_utils.h"

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


  /// @brief Extended execution parameters (for transformation / segmentation).
namespace
{
struct rnn_execution_params : execution_params
{
    float       temperature;
    std::string src_vocab_file;
    std::string tgt_vocab_file;
    uint32_t    sequence_length;
    bool        word_type = false;
    bool        character_type = false;
    uint32_t    beam_size;
};
}

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
            "Path to input directory containing sentences to transform.")
        ("model,m", bpo::value<std::string>()->value_name("<model-name>")->default_value("fns-candy"),
            "Name of a recurrent neural network model.\n"
            "It can be one of:\n  \tlstm_char\n \tonmt_ger_to_eng_6_layers.")
        ("weights,w", bpo::value<std::string>()->value_name("<weights-dir>"),
            "Path to directory containing weights used in inference.\n"
            "Non-absolute paths are computed in relation to <executable-dir> (not working directory).\n"
            "If not specified, the \"<executable-dir>/<model-name>\" path is used in first place with "
            "\"<executable-dir>/weights\" as fallback.")
        ("sequence_length", bpo::value<std::uint32_t>()->value_name("<sequence-length>")->default_value(0),
            "Used in RNN topologies (LSTM).")
        ("src_vocab_file", bpo::value<std::string>()->value_name("<src-vocabulary-file>"),
            "Path to source vocabulary file (.txt format).")
        ("tgt_vocab_file", bpo::value<std::string>()->value_name("<tgt-vocabularyfile>"),
            "Path to target vocabulary file (.txt format).")
        ("temperature", bpo::value<float>()->value_name("<temperature-value>")->default_value(0.0f),
            "Temperature for character selection at the output <range from 0.0 to 1.0>. (default 0.0f)")
        ("beam_size", bpo::value<uint32_t>()->value_name("<beam_size-value>")->default_value(0),
            "Tooltip about beam size.");

    const auto help_msg_formatter =
        [](std::ostream& out, const std::shared_ptr<const executable_info>& exec_info) -> std::ostream&
    {
        // ------------------------------------------------------------------------------------------------------------
        // Help message.
        out << "Usage:\n  " << exec_info->file_name_wo_ext() << " [standard options]\n\n";
        out << "Executes recurrent neural network.\n";
        return out;
    };

    return {exec_info, standard_cmdline_options, true, help_msg_formatter };
}

fp_seconds_type execute_character_type_network(const cldnn::engine& selected_engine, const cldnn::topology& selected_topology,
    cldnn::memory& input, CIntelPowerGadgetLib& power_measure_lib, const execution_params& exec_params, char_rnn_utils* utils)
{
    float zero = 0;
    cldnn::layout zero_layout(cldnn::data_types::f32, cldnn::format::bfyx, cldnn::tensor(1));
    auto output = cldnn::memory::attach(zero_layout, &zero, 1);

    auto network = build_network(selected_engine, selected_topology, exec_params);

    //initial input data
    utils->fill_memory(input);

    network.set_input_data("input", input);
    bool log_energy = do_log_energy(exec_params, power_measure_lib);

    if (exec_params.print_type == print_type::verbose)
    {
        std::cout << "Start execution.";
        if (exec_params.loop > 1)
        {
            std::cout << " Will make " << exec_params.loop << " predictions per batch.";
        }
        std::cout << std::endl;
    }
    decltype(network.execute()) outputs;
    cldnn::instrumentation::timer<> timer_execution;

    for (decltype(exec_params.loop) i = 0; i < exec_params.loop; i++)
    {
        outputs = network.execute();
        utils->post_process_output(outputs.at("output").get_memory());

        utils->fill_memory(input);

        network.set_input_data("input", input);

        if (log_energy)
            power_measure_lib.ReadSample();

    }
    return get_execution_time(timer_execution, exec_params, output, outputs, log_energy, power_measure_lib);

}

fp_seconds_type execute_word_type_network(const cldnn::engine& selected_engine, const cldnn::topology& selected_encoder, 
    const cldnn::topology& selected_decoder, const std::vector<cldnn::layout>& input_layouts, std::vector<cldnn::memory>& inputs, CIntelPowerGadgetLib& power_measure_lib, const execution_params& exec_params, nmt_utils* utils)
{
    float zero = 0;
    cldnn::layout zero_layout(cldnn::data_types::f32, cldnn::format::bfyx, cldnn::tensor(1));
    auto output = cldnn::memory::attach(zero_layout, &zero, 1);


    std::vector<std::string> encoder_outputs;
    for (auto const& inp_lay : input_layouts)
    {
        std::string prefix = "b_" + std::to_string(inp_lay.size.batch[0]) + "_s_" + std::to_string(inp_lay.size.spatial[0]) + "_";
        for (size_t b = 0; b < inp_lay.size.batch[0]; b++)
        {
            encoder_outputs.push_back(prefix + "initial_hidden_to_decoder_0_batch_" + std::to_string(b));
            encoder_outputs.push_back(prefix + "initial_hidden_to_decoder_1_batch_" + std::to_string(b));
            encoder_outputs.push_back(prefix + "initial_cell_to_decoder_0_batch_" + std::to_string(b));
            encoder_outputs.push_back(prefix + "initial_cell_to_decoder_1_batch_" + std::to_string(b));
            encoder_outputs.push_back("padded_" + prefix + "memory_bank_batch_" + std::to_string(b));
            encoder_outputs.push_back(prefix + "encoder_concated_all_hidden_states");
        }
    }
    std::vector<std::string> decoder_outputs =
    {
        "cropped_arg_max", "permute_linear_out",
        "dec_crop00000:hidden", "1_dec_crop00000:hidden",
        "dec_crop00000:cell", "1_dec_crop00000:cell",
        "reshaped_best_scores", "reshaped_flattened_best_scores",
        "aligned_vectors", "flattened_arg_max", "output"
    };

    auto encoder_network = build_network(selected_engine, selected_encoder, exec_params, encoder_outputs);

    auto decoder_network = build_network(selected_engine, selected_decoder, exec_params, decoder_outputs);
   
    bool log_energy = do_log_energy(exec_params, power_measure_lib);


    decltype(encoder_network.execute()) enc_output;
    decltype(decoder_network.execute()) dec_output;
    cldnn::instrumentation::timer<> timer_execution;

    auto beam_size = utils->get_beam_size();
    auto embedding_size = utils->get_embedding_size();

    for (auto& input : inputs)
    {
        size_t seq_len = input.get_layout().size.spatial[0];
        utils->fill_memory(input, seq_len);
        std::string prefix = "b_" + std::to_string(input.get_layout().size.batch[0]) + "_s_" + std::to_string(input.get_layout().size.spatial[0]) + "_";
        encoder_network.set_input_data(prefix + "input", input);
    }

    enc_output = encoder_network.execute();

    for (size_t i = 0; i < inputs.size(); i++)
    {
        auto& input = inputs.at(i);
        uint32_t current_seq_len = input.get_layout().size.spatial[0];

        for (size_t b = 0; b < input.get_layout().size.batch[0]; b++)
        {
            std::string prefix = "b_" + std::to_string(input.get_layout().size.batch[0]) + "_s_" + std::to_string(input.get_layout().size.spatial[0]) + "_";
            auto hidden_to_decoder_lstm_0_memory = enc_output.at(prefix + "initial_hidden_to_decoder_0_batch_" + std::to_string(b)).get_memory();
            auto hidden_to_decoder_lstm_1_memory = enc_output.at(prefix + "initial_hidden_to_decoder_1_batch_" + std::to_string(b)).get_memory();
            auto cell_to_decoder_lstm_0_memory = enc_output.at(prefix + "initial_cell_to_decoder_0_batch_" + std::to_string(b)).get_memory();
            auto cell_to_decoder_lstm_1_memory = enc_output.at(prefix + "initial_cell_to_decoder_1_batch_" + std::to_string(b)).get_memory();
            auto memory_bank = enc_output.at("padded_" + prefix + "memory_bank_batch_" + std::to_string(b)).get_memory();
            auto padded_tokens_memory = cldnn::memory::allocate(selected_engine, { cldnn::data_types::f32, cldnn::format::bfyx,{ 1, 1, beam_size, 1 } });
            memory_filler::fill_memory<float>(padded_tokens_memory, std::vector<float>({ 2, 1, 1, 1, 1 }));
            auto input_feed_memory = cldnn::memory::allocate(selected_engine, { cldnn::data_types::f32, cldnn::format::bfyx,{ 1, beam_size, embedding_size, 1 } }); //default values are 0, so its ok

            auto initial_best_scores = cldnn::memory::allocate(selected_engine, { cldnn::data_types::f32, cldnn::format::bfyx,{ beam_size, 1, 1, 1 } });

            auto indicies_to_index_select_memory = cldnn::memory::allocate(selected_engine, { cldnn::data_types::i32, cldnn::format::bfyx,{ 1, 1, beam_size, 1 } });
            memory_filler::fill_memory<int32_t>(indicies_to_index_select_memory, std::vector<int32_t>({ 0, 1, 2, 3, 4 })); //initial run do nothing

            for (size_t j = 0; j < 100; j++)
            {
                if (j == 0) //First iteration, connect encoder to decoder
                {
                    decoder_network.set_input_data("memory_bank", memory_bank);
                    decoder_network.set_input_data("padded_tokens", padded_tokens_memory);
                    decoder_network.set_input_data("input_feed", input_feed_memory);
                    decoder_network.set_input_data("input_best_scores", initial_best_scores);
                    decoder_network.set_input_data("indices_to_index_select", indicies_to_index_select_memory);
                    decoder_network.set_input_data("initital_hidden_0", hidden_to_decoder_lstm_0_memory);
                    decoder_network.set_input_data("initital_hidden_1", hidden_to_decoder_lstm_1_memory);
                    decoder_network.set_input_data("initital_cell_0", cell_to_decoder_lstm_0_memory);
                    decoder_network.set_input_data("initital_cell_1", cell_to_decoder_lstm_1_memory);
                }
                else if (j == 1) //first run of decoder, feed decoder with previous iteration outputs
                {
                    decoder_network.set_input_data("memory_bank", memory_bank);
                    decoder_network.set_input_data("padded_tokens", dec_output.at("cropped_arg_max").get_memory());
                    decoder_network.set_input_data("input_feed", dec_output.at("permute_linear_out").get_memory());
                    decoder_network.set_input_data("indices_to_index_select", indicies_to_index_select_memory);
                    decoder_network.set_input_data("initital_hidden_0", dec_output.at("dec_crop00000:hidden").get_memory());
                    decoder_network.set_input_data("initital_hidden_1", dec_output.at("1_dec_crop00000:hidden").get_memory());
                    decoder_network.set_input_data("initital_cell_0", dec_output.at("dec_crop00000:cell").get_memory());
                    decoder_network.set_input_data("initital_cell_1", dec_output.at("1_dec_crop00000:cell").get_memory());
                    decoder_network.set_input_data("input_best_scores", dec_output.at("reshaped_best_scores").get_memory());
                }
                else
                {
                    decoder_network.set_input_data("memory_bank", memory_bank);
                    decoder_network.set_input_data("padded_tokens", padded_tokens_memory);
                    decoder_network.set_input_data("input_feed", dec_output.at("permute_linear_out").get_memory());
                    decoder_network.set_input_data("indices_to_index_select", indicies_to_index_select_memory);
                    decoder_network.set_input_data("initital_hidden_0", dec_output.at("dec_crop00000:hidden").get_memory());
                    decoder_network.set_input_data("initital_hidden_1", dec_output.at("1_dec_crop00000:hidden").get_memory());
                    decoder_network.set_input_data("initital_cell_0", dec_output.at("dec_crop00000:cell").get_memory());
                    decoder_network.set_input_data("initital_cell_1", dec_output.at("1_dec_crop00000:cell").get_memory());
                    decoder_network.set_input_data("input_best_scores", dec_output.at("reshaped_flattened_best_scores").get_memory());
                }

                dec_output = decoder_network.execute();

                if (j == 0)
                {
                    utils->add_attention_output(
                        current_seq_len, b,
                        dec_output.at("aligned_vectors").get_memory(),
                        dec_output.at("cropped_arg_max").get_memory(),
                        dec_output.at("reshaped_best_scores").get_memory()
                    );
                }
                else
                {
                    utils->add_attention_output(
                        current_seq_len, b,
                        dec_output.at("aligned_vectors").get_memory(),
                        dec_output.at("flattened_arg_max").get_memory(),
                        dec_output.at("reshaped_flattened_best_scores").get_memory()
                    );
                }
                auto prev_ys = utils->get_batches().at(current_seq_len).get_mini_batches().at(b)->get_last_prev_ys();
                memory_filler::fill_memory<int32_t>(indicies_to_index_select_memory, prev_ys);
                memory_filler::fill_memory<float>(padded_tokens_memory, utils->get_input_to_iteration(current_seq_len, b));
                if (utils->found_eos_token(current_seq_len, b))
                {
                    break;
                }

            }
        }
    }

    if (log_energy)
        power_measure_lib.ReadSample();

     return get_execution_time(timer_execution, exec_params, output, dec_output, log_energy, power_measure_lib);
}

void run_topology(const rnn_execution_params& exec_params)
{
    auto input_list = list_input_files(exec_params.input_dir, input_file_type::text);
    if (input_list.empty())
    {
        throw std::runtime_error("specified input images directory is empty (does not contain image data)");
    }
    if (exec_params.word_type && input_list.size() != 1)
    {
        throw std::runtime_error("[ERROR] NMT topologies support one file input.Batches taken from this file.");
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

    if (exec_params.print_type == print_type::verbose)
        std::cout << "Building of \"" << exec_params.topology_name << "\" started..." << std::endl;
    else if (exec_params.print_type == print_type::extended_testing)
        std::cout << "Extended testing of \"" << exec_params.topology_name << "\" started..." << std::endl;

    // --- Measurement: build time (start).
    cldnn::instrumentation::timer<> timer_build;

    std::shared_ptr<rnn_base> rnn_utils;
    uint32_t seq_len = 0;
    if (exec_params.word_type)
    {
        rnn_utils = std::make_shared<nmt_utils>(
            exec_params.src_vocab_file,
            exec_params.tgt_vocab_file,
            exec_params.beam_size,
            500
            );
    }
    else
    {
        rnn_utils = std::make_shared<char_rnn_utils>(
            exec_params.sequence_length,
            (uint32_t)exec_params.loop,
            exec_params.temperature,
            exec_params.src_vocab_file);
    }
    
    //do the pre processing
    rnn_utils->pre_process(input_list);
    auto batch_size = rnn_utils->get_batch_size();

    std::vector<cldnn::layout> input_layouts;
    std::vector<cldnn::memory> inputs;
    std::vector<cldnn::topology> topologies;

    if (exec_params.character_type)
    {
        input_layouts.push_back(cldnn::layout(exec_params.use_half ? cldnn::data_types::f16 : cldnn::data_types::f32, cldnn::format::byxf, {}));
        if (exec_params.topology_name == "shakespears_generator")
        {
            topologies.emplace_back(build_char_level(exec_params.weights_dir, selected_engine, input_layouts.at(0), exec_params.batch, exec_params.sequence_length));
        }
        inputs.push_back(cldnn::memory::allocate(selected_engine, input_layouts.back()));
    }
    else if (exec_params.word_type)
    {
        //WORK IN PROGRESS
        auto utils = dynamic_cast<nmt_utils*>(rnn_utils.get());
        auto batch_and_seq_lens = utils->batches_and_seq_lens();
        if (exec_params.topology_name == "onmt_ger_to_eng_6_layers")
        {
            for (auto const& mb : batch_and_seq_lens)
            {
                input_layouts.push_back(cldnn::layout(exec_params.use_half ? cldnn::data_types::f16 : cldnn::data_types::f32, cldnn::format::bfyx, {}));
                input_layouts.back().size = { static_cast<int32_t>(mb.first), 1, static_cast<int32_t>(mb.second), 1 };
                inputs.push_back(cldnn::memory::allocate(selected_engine, input_layouts.back()));
            }
            topologies.push_back(build_onmt_ger_to_eng_6_layers_encoder(exec_params.weights_dir, selected_engine, input_layouts, batch_and_seq_lens, utils->get_max_seq_len(), exec_params.beam_size));
            topologies.push_back(build_onmt_ger_to_eng_6_layers_decoder(exec_params.weights_dir, selected_engine, utils->get_max_seq_len(), exec_params.beam_size));
        }
    }
    else
    {
        throw std::runtime_error("Topology \"" + exec_params.topology_name + "\" not implemented!");
    }

    const auto build_time = timer_build.uptime();
    if (exec_params.print_type == print_type::verbose)
    {
        std::cout << "Building of \"" << exec_params.topology_name << "\" finished (time: "
            << instrumentation::to_string(build_time) << ")." << std::endl;
    }

    fp_seconds_type time{ 0.0 };
    if (exec_params.character_type)
    {
        time = execute_character_type_network(
            selected_engine,
            topologies.at(0),
            inputs.at(0),
            power_measure_lib,
            exec_params,
            dynamic_cast<char_rnn_utils*>(rnn_utils.get()));
    }
    else if (exec_params.word_type)
    {
        time = execute_word_type_network(
            selected_engine,
            topologies.at(0),
            topologies.at(1),
            input_layouts,
            inputs,
            power_measure_lib,
            exec_params,
            dynamic_cast<nmt_utils*>(rnn_utils.get()));
    }

    if (exec_params.run_until_primitive_name.empty() && exec_params.run_single_kernel_name.empty())
    {
        //show the output
        std::string result = rnn_utils->get_predictions();
        std::cout << "Result of execution: \n"""
            << result << std::endl;
    }
    if (!exec_params.run_until_primitive_name.empty())
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

// --------------------------------------------------------------------------------------------------------------------

int main(int argc, char* argv[])
{
    namespace bfs = boost::filesystem;

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

        if (!parsed_args.count("input"))
        {
            std::cerr << "ERROR: input has not been provided !!!" << std::endl;
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
        // Execute network otherwise.
        rnn_execution_params ep;
        parse_common_options(parsed_args, ep);

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
            std::string model_name = get_model_name(parsed_args["model"].as<std::string>());
            weights_dir = join_path(exec_info->dir(), model_name);

            auto lr_string = std::to_string(parsed_args["lr"].as<float>());
            lr_string = lr_string.substr(lr_string.find_last_of(".") + 1);
            weights_dir += "_lr" + lr_string + "_b" + std::to_string(ep.batch);
            boost::filesystem::create_directories(weights_dir);
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

        auto src_vocab_file = parsed_args["src_vocab_file"].as<std::string>();
        if (!bfs::exists(src_vocab_file) || !bfs::is_regular_file(src_vocab_file) || bfs::extension(src_vocab_file) != ".txt")
        {
            std::cerr << "ERROR: specified source vocabulary file (\"" << src_vocab_file
                << "\") does not exist or is not .txt type of file !!!" << std::endl;
            return 1;
        }
        ep.src_vocab_file = src_vocab_file;
        if (parsed_args.count("tgt_vocab_file"))
        {
            auto tgt_vocab_file = parsed_args["tgt_vocab_file"].as<std::string>();
            if (tgt_vocab_file != "" &&
                (!bfs::exists(tgt_vocab_file) || !bfs::is_regular_file(tgt_vocab_file) || bfs::extension(tgt_vocab_file) != ".txt"))
            {
                std::cerr << "ERROR: specified target vocabulary file (\"" << tgt_vocab_file
                    << "\") does not exist or is not .txt type of file !!!" << std::endl;
                return 1;
            }
            ep.tgt_vocab_file = tgt_vocab_file;
        }

        //decide what type of model 
        ep.topology_name = parsed_args["model"].as<std::string>();
        if (ep.topology_name == "onmt_ger_to_eng_6_layers")
        {
            ep.word_type = true;
        }
        if (ep.topology_name == "shakespears_generator")
        {
            ep.character_type = true;
        }

        ep.temperature = parsed_args["temperature"].as<float>();
        if (ep.character_type)
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
        else if (ep.word_type && ep.temperature > 0.0f)
        {
            std::cout << "[WARNING] Temperature not used for word types topologies." << std::endl;
        }

        ep.beam_size = parsed_args["beam_size"].as<uint32_t>();
        if (ep.beam_size != 0 && ep.character_type)
        {
            std::cout << "[WARNING] Beam size is not used for character types topologies." << std::endl;
        }
        ep.sequence_length = parsed_args["sequence_length"].as<uint32_t>();
        
        if (ep.topology_name == "onmt_ger_to_eng_6_layers" ||
            ep.topology_name == "shakespears_generator")
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
