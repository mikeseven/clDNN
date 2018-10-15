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
#include "file.h"

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
#include <random>

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
        ("model", bpo::value<std::string>()->value_name("<model-name>")->default_value("lenet"),
            "Name of a neural network model that is used for classification.\n"
            "It can be one of:\n  \tlenet, lenet_train, vgg16_train, vgg16_test, resnet50_train.")
        ("weights", bpo::value<std::string>()->value_name("<weights-dir>"),
            "Path to directory containing weights used in classification.\n"
            "Non-absolute paths are computed in relation to <executable-dir> (not working directory).\n"
            "If not specified, the \"<executable-dir>/<model-name>\" path is used in first place with \"<executable-dir>/weights\" as fallback.")
        ("image_number", bpo::value<std::uint32_t>()->value_name("<image_number>")->default_value(8),
            "Number of images that will be used for traning. Default value is 8.")
        ("image_offset", bpo::value<std::uint32_t>()->value_name("<image_offset>")->default_value(0),
            "How many images should be skipped in mnist data on execution.")
        ("use_existing_weights", bpo::bool_switch(),
            "Parameter used in learning, when it is set then model will use existing weights files")
        ("compute_imagemean", bpo::bool_switch(),
            "Parameter used in learning, when it is set then imagemean value will be computed for the requested image set")
        ("lr", bpo::value<float>()->value_name("<lr>")->default_value(0.00001f),
            "Base learning rate for network training. Default value is 0.00001f.")
        ("train_snapshot", bpo::value<std::uint32_t>()->value_name("<train_snapshot>")->default_value(100),
            "After how many iterations, the weights and biases files will be updated on disk. Default value is 100.")
        ("continue_training", bpo::bool_switch(),
            "Continue training with the data provided in train_iteration.txt file.")
        ("image_set", bpo::value<std::string>()->value_name("<image_set>")->default_value("imagenet"),
            "Imageset that will be used. Currently supported: mnist (lenet), imagenet (vgg16), cifar10.")
        ("epoch", bpo::value<std::uint32_t>()->value_name("<epoch>")->default_value(1),
            "Number of epochs that will be done for traning. Default value is 1.");

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

template<typename MemElemTy>
void generate_bernoulli(cldnn::memory& mem, const float threshold, unsigned int seed)
{
    auto memory_layout = mem.get_layout();
    auto dst_ptr = mem.pointer<MemElemTy>();
    std::default_random_engine generator(seed);
    float scale = 1.f / (1.f - threshold);

    std::bernoulli_distribution distribution(threshold);
    for (uint32_t i = 0; i < (uint32_t)memory_layout.count(); i++)
        dst_ptr[i] = distribution(generator) * scale;
}

uint32_t get_output_index(uint32_t label_idx, uint32_t batch, uint32_t labels_num, uint32_t batch_size, cldnn::format layout)
{
    uint32_t index = 0;
    if (layout == cldnn::format::bfyx || layout == cldnn::format::byxf)
        index = batch * labels_num + label_idx;
    else
        index = batch + label_idx * batch_size;

    return index;
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
    std::vector<cldnn::primitive_id> outputs(0);
    if (ep.topology_name == "vgg16_test")
        primitives = build_vgg16(ep.weights_dir, engine, input_layout, gpu_batch_size);
    else if (ep.topology_name == "vgg16_train")
    {
        if (ep.compute_imagemean)
            compute_image_mean(ep, engine, ep.image_set == "cifar10");

        primitives = build_vgg16_train(ep.weights_dir, engine, input_layout, gpu_batch_size, ep.use_existing_weights, outputs);
    }
    else if (ep.topology_name == "lenet")
        primitives = build_lenet(ep.weights_dir, engine, input_layout, gpu_batch_size);
    else if (ep.topology_name == "lenet_train")
        primitives = build_lenet_train(ep.weights_dir, engine, input_layout, gpu_batch_size, ep.use_existing_weights, outputs);
    else if (ep.topology_name == "resnet50_train")
    {
        if (ep.compute_imagemean)
            compute_image_mean(ep, engine, ep.image_set == "cifar10");

        primitives = build_resnet50_train(ep.weights_dir, engine, input_layout, gpu_batch_size, true, ep.use_existing_weights, outputs);
    }
    else
        throw std::runtime_error("Topology \"" + ep.topology_name + "\" not implemented!");

    //load images in fp32 and convert them in topology
    input_layout.data_type = cldnn::data_types::f32;

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
    auto network = build_network(engine, primitives, ep, outputs);
    //TODO check if we can define the 'empty' memory
    float zero = 0;
    cldnn::layout zero_layout(cldnn::data_types::f32, cldnn::format::bfyx, { 1,1,1,1 });
    auto output = cldnn::memory::attach(zero_layout, &zero, 1);

        auto input = cldnn::memory::allocate(engine, input_layout);
        auto neurons_list_filename = "lenet.txt";
        if (ep.topology_name == "vgg16_test")
            neurons_list_filename = "names.txt";

        auto input_list = list_input_files(ep.input_dir);
        if (input_list.empty())
            throw std::runtime_error("specified input images directory is empty (does not contain image data)");

        auto number_of_batches = (input_list.size() % batch_size == 0)
            ? input_list.size() / batch_size : input_list.size() / batch_size + 1;
        std::vector<std::string> input_files_in_batch;
        auto input_list_iterator = input_list.begin();
        auto input_list_end = input_list.end();

        if (ep.topology_name == "lenet" || ep.topology_name == "lenet_train" || ep.topology_name == "vgg16_train" || ep.topology_name == "vgg16_test" || ep.topology_name == "resnet50_train")
            number_of_batches = 1;

        for (decltype(number_of_batches) batch = 0; batch < number_of_batches; batch++)
        {
            input_files_in_batch.clear();
            for (uint32_t i = 0; i < batch_size && input_list_iterator != input_list_end; ++i, ++input_list_iterator)
            {
                input_files_in_batch.push_back(*input_list_iterator);
            }
            double time_in_sec = 0.0;
            // load croped and resized images into input
            if (ep.topology_name != "lenet" && ep.topology_name != "lenet_train" && ep.topology_name != "vgg16_train" && ep.topology_name != "vgg16_test" && ep.topology_name != "resnet50_train")
            {
                load_image_files(input_files_in_batch, input);
            }
            else if (ep.topology_name == "lenet")
            {
                if (ep.image_set != "mnist")
                    throw std::runtime_error("Lenet only support mnist images. Please use --image_set=mnist!");

                float acc = 0;
                uint32_t labels_num = 10;
                auto labels = cldnn::memory::allocate(engine, { cldnn::data_types::f32, cldnn::format::bfyx,{ input_layout.size.batch[0],1,1,1 } });
                for (uint32_t i = ep.image_offset; i < ep.image_number + ep.image_offset; i += batch_size)
                {
                    load_data_from_file_list_lenet(input_list, input, i, batch_size, false, labels);

                    network.set_input_data("input", input);
                    auto outputs = network.execute();
                    auto o = outputs.at("output").get_memory().pointer<float>();
                    auto vals_layout_format = outputs.at("output").get_memory().get_layout().format;

                    auto l = labels.pointer<float>();
                    for (uint32_t b = 0; b < batch_size; b++)
                    {
                        auto e = l[b];
                        std::vector< std::pair<float, uint32_t> > output_vec;

                        //check if true label is on top of predictions
                        for (uint32_t j = 0; j < labels_num; j++)
                            output_vec.push_back(std::make_pair(o[get_output_index(j, b, labels_num, batch_size, vals_layout_format)], j));

                        std::sort(output_vec.begin(), output_vec.end(), std::greater<std::pair<float, uint32_t> >());

                        if (output_vec[0].second == e)
                            acc++;
                    }
                }
                std::cout << "Images processed = " << ep.image_number << std::endl;
                std::cout << "Accuracy = " << acc / ep.image_number << std::endl;
                continue;
            }
            else if (ep.topology_name == "lenet_train")
            {
                if (ep.image_set != "mnist")
                    throw std::runtime_error("Lenet only support mnist images. Please use --image_set=mnist!");

                uint32_t labels_num = 10;
                auto labels = cldnn::memory::allocate(engine, { cldnn::data_types::f32, cldnn::format::bfyx,{ input_layout.size.batch[0],1,1,1 } });
                float base_learning_rate = ep.learning_rate;
                float learning_rate = base_learning_rate;
                for (uint32_t epoch_it = 0; epoch_it < ep.epoch_number; epoch_it++)
                {
                    for (uint32_t learn_it = ep.image_offset; learn_it < ep.image_number + ep.image_offset; learn_it += batch_size)
                    {
                        double loss = 0;
                        //update learning rate, policy "inv", gamma=0.0001, power=0.75.
                        //TODO: enable getting learning rate params from command line
                        learning_rate = base_learning_rate * pow(1.f + 0.0001f * learn_it, -0.75f);
                        loss = 0;
                        network.set_learning_rate(learning_rate);

                        load_data_from_file_list_lenet(input_list, input, learn_it, batch_size, true, labels);

                        network.set_input_data("input", input);
                        network.set_input_data("labels", labels);
                        auto time = execute_cnn_topology(network, ep, energyLib, output, learn_it, ep.image_offset + ep.image_number / batch_size - 1);
                        time_in_sec = std::chrono::duration_cast<std::chrono::duration<double, std::chrono::seconds::period>>(time).count();
                        auto expected = labels.pointer<float>();
                        auto vals = output.pointer<float>();
                        auto vals_layout_format = output.get_layout().format;

                        for (uint32_t b = 0; b < batch_size; b++)
                        {
                            auto e = (uint32_t)expected[b];
                            loss -= log(std::max(vals[get_output_index(e, b, labels_num, batch_size, vals_layout_format)], std::numeric_limits<float>::min()));
                        }

                        loss = loss / batch_size;
                        std::cout << "Epoch: " << epoch_it << ", Iter: " << learn_it - ep.image_offset << std::endl;
                        std::cout << "Loss = " << loss << std::endl;
                        std::cout << "Learning Rate = " << learning_rate << std::endl;
                    }
                }
                continue;
            }
            else if (ep.topology_name == "vgg16_train" || ep.topology_name == "resnet50_train")
            {
                if (ep.image_set != "imagenet")
                    throw std::runtime_error("Vgg16 and resnet50 support only imagenet images!");

                float acc = 0;
                auto labels = cldnn::memory::allocate(engine, { cldnn::data_types::f32, cldnn::format::bfyx,{ input_layout.size.batch[0],1,1,1 } });
                auto fc6_dropout_mem = cldnn::memory::allocate(engine, { input_layout.data_type, cldnn::format::bfyx,{ input_layout.size.batch[0],1,4096,1 } });
                auto fc7_dropout_mem = cldnn::memory::allocate(engine, { input_layout.data_type, cldnn::format::bfyx,{ input_layout.size.batch[0],1,4096,1 } });
                float base_learning_rate = ep.learning_rate;
                float learning_rate = base_learning_rate;
                for (uint32_t epoch_it = 0; epoch_it < ep.epoch_number; epoch_it++)
                {
                    for (uint32_t learn_it = ep.image_offset; learn_it < ep.image_number + ep.image_offset; learn_it += batch_size)
                    {
                        double loss = 0;
                        //TODO: enable getting learning rate params from command line
                        if (ep.topology_name == "vgg16_train")
                        {
                            //update learning rate, policy "step", gamma=0.001, stepsize=1000.
                            learning_rate = base_learning_rate * pow(0.001f, ((epoch_it * ep.image_number + learn_it) / 128.f) / 1000.f);
                        }
                        else //resnet50
                        {
                            //update learning rate, policy "step", gamma=0.1, stepsize=200000.
                            learning_rate = base_learning_rate * pow(0.1f, (epoch_it * ep.image_number + learn_it) / 32500.f);
                        }
                        network.set_learning_rate(learning_rate);

                        load_data_from_file_list_imagenet(input_list, ep.input_dir, input, learn_it, batch_size, true, labels);

                        //add dropout layers to VGG16
                        if (ep.topology_name == "vgg16_train")
                        {
                            generate_bernoulli<float>(fc6_dropout_mem, 0.5f, learn_it);
                            generate_bernoulli<float>(fc7_dropout_mem, 0.5f, learn_it + 1);
                            network.set_input_data("fc6_dropout_mask", fc6_dropout_mem);
                            network.set_input_data("fc7_dropout_mask", fc7_dropout_mem);
                        }
                        network.set_input_data("input", input);
                        network.set_input_data("labels", labels);

                        auto time = execute_cnn_topology(network, ep, energyLib, output, learn_it, ep.image_offset + ep.image_number / batch_size - 1);
                        time_in_sec = std::chrono::duration_cast<std::chrono::duration<double, std::chrono::seconds::period>>(time).count();
                        auto expected = labels.pointer<float>();
                        auto vals = output.pointer<float>();
                        auto vals_layout_format = output.get_layout().format;

                        float acc = 0;
                        uint32_t labels_num = 1000;

                        for (uint32_t b = 0; b < batch_size; b++)
                        {
                            auto e = (uint32_t)expected[b];
                            std::vector< std::pair<float, uint32_t> > output_vec;
                            auto index = get_output_index(e, b, labels_num, batch_size, vals_layout_format);

                            loss -= log(std::max(vals[index], std::numeric_limits<float>::min()));

                            std::cout << "Prob of correct result: " << vals[index] << std::endl;

                            //check if true label is on top of predictions
                            for (uint32_t j = 0; j < labels_num; j++)
                                output_vec.push_back(std::make_pair(vals[get_output_index(j, b, labels_num, batch_size, vals_layout_format)], j));

                            std::sort(output_vec.begin(), output_vec.end(), std::greater<std::pair<float, uint32_t> >());

                            if (output_vec[0].second == e)
                                acc++;
                        }
                        loss = loss / batch_size;
                        std::cout << "Train accuracy: " << acc / batch_size << std::endl;
                        std::cout << "Epoch: " << epoch_it << ", Iter: " << learn_it - ep.image_offset << std::endl;
                        std::cout << "Loss = " << loss << std::endl;
                        std::cout << "Learning Rate = " << learning_rate << std::endl;
                    }
                }
                continue;
            }
            else if (ep.topology_name == "vgg16_test")
            {
                if (ep.image_set != "imagenet")
                    throw std::runtime_error("Vgg16 and resnet50 supports only imagenet images!");

                float acc = 0;
                uint32_t labels_num = 1000;
                auto labels = cldnn::memory::allocate(engine, { cldnn::data_types::f32, cldnn::format::bfyx,{ input_layout.size.batch[0],1,1,1 } });
                for (uint32_t learn_it = ep.image_offset; learn_it < ep.image_number + ep.image_offset; learn_it += batch_size)
                {
                    if (ep.use_half)
                        load_data_from_file_list_imagenet<half_t>(input_list, ep.input_dir, input, learn_it, batch_size, false, labels);
                    else
                        load_data_from_file_list_imagenet(input_list, ep.input_dir, input, learn_it, batch_size, false, labels);

                    network.set_input_data("input", input);
                    auto time = execute_cnn_topology(network, ep, energyLib, output, learn_it, ep.image_offset + ep.image_number / batch_size - 1);
                    time_in_sec = std::chrono::duration_cast<std::chrono::duration<double, std::chrono::seconds::period>>(time).count();
                    auto expected = labels.pointer<float>();
                    auto vals = output.pointer<float>();
                    auto vals_layout_format = output.get_layout().format;

                    for (uint32_t b = 0; b < batch_size; b++)
                    {
                        auto e = (uint32_t)expected[b];
                        std::vector< std::pair<float, uint32_t> > output_vec;

                        //check if true label is on top of predictions
                        for (uint32_t j = 0; j < labels_num; j++)
                            output_vec.push_back(std::make_pair(vals[get_output_index(j, b, labels_num, batch_size, vals_layout_format)], j));

                        std::sort(output_vec.begin(), output_vec.end(), std::greater<std::pair<float, uint32_t> >());

                        if (output_vec[0].second == e)
                            acc++;
                    }
                }
                std::cout << "Images processed = " << ep.image_number << std::endl;
                std::cout << "Accuracy = " << acc / ep.image_number << std::endl;
                continue;
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
        }

        ep.topology_name = parsed_args["model"].as<std::string>();
        ep.image_number = parsed_args["image_number"].as<std::uint32_t>();
        ep.use_existing_weights = parsed_args["use_existing_weights"].as<bool>();
        ep.image_set = parsed_args["image_set"].as<std::string>();
        ep.epoch_number = parsed_args["epoch"].as<std::uint32_t>();

        if(ep.image_set != "mnist" && ep.image_set != "imagenet" && ep.image_set != "cifar10")
            std::cerr << "ERROR: image_set (\"" << ep.topology_name << "\") is not implemented!!!" << std::endl;

        if (parsed_args["continue_training"].as<bool>())
        {
            ep.use_existing_weights = true;
            ep.image_offset = file::get_train_iteration(join_path(ep.weights_dir, "train_iteration.txt"));
        }
        else
            ep.image_offset = parsed_args["image_offset"].as<std::uint32_t>();
        ep.train_snapshot = parsed_args["train_snapshot"].as<std::uint32_t>();

        ep.compute_imagemean = parsed_args["compute_imagemean"].as<bool>();
        ep.learning_rate = parsed_args["lr"].as<float>();

        if (ep.topology_name == "vgg16_train" ||
            ep.topology_name == "vgg16_test" ||
            ep.topology_name == "lenet" ||
            ep.topology_name == "lenet_train" ||
            ep.topology_name == "resnet50_train")
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
