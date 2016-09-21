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

#include "api/neural.h"

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <regex>
#include <string>
#include <type_traits>


// parses parameters stored as vector of strings and insersts them into map
void parse_parameters(std::map<std::string, std::string> &config, std::vector<std::string> &input) {
    std::regex regex[] = {
        std::regex("(--|-|\\/)([a-z][a-z\\-_]*)=(.*)")    // key-value pair
        , std::regex("(--|-|\\/)([a-z][a-z\\-_]*)")         // key only
        , std::regex(".+")                                  // this must be last in regex[]; what is left is a filename
    };
    for (std::string &in : input) {
        std::cmatch result;
        const auto regex_count = sizeof(regex) / sizeof(regex[0]);
        const auto regex_last = regex_count - 1;
        for (size_t index = 0; index<regex_count; ++index)
            if (std::regex_match(in.c_str(), result, regex[index])) {
                std::string key = index == regex_last ? std::string("input") : result[2];
                std::string value = result[index == regex_last ? 0 : 3];
                config.insert(std::make_pair(key, value));
                break;
            }
    }
}

boost::program_options::variables_map parse_cmdline_options(int argc, const char* const argv[])
{
    namespace bpo = boost::program_options;
    namespace bfs = boost::filesystem;

    bpo::options_description standard_cmdline_options("Standard options");
    standard_cmdline_options.add_options()
        ("batch", bpo::value<std::uint32_t>()->value_name("<batch-size>")->default_value(32),
            "Size of a group of images that are classified together (large batch sizes have better performance).")
        ("model", bpo::value<std::string>()->value_name("<model-name>")->default_value("alexnet"),
            "Name of a neural network model that is used for classification.\n"
            "It can be one of:\n  \talexnet, caffenet_float, caffenet_int16, lenet_float.")
        ("engine", bpo::value<std::string>()->value_name("<engine-type>")->default_value("reference"),
            "Type of an engine used for classification.\nIt can be one of:\n  \treference, gpu.")
        ("dump_hidden_layers", bpo::bool_switch(),
            "Dump results from hidden layers of network to files.")
        ("input", bpo::value<std::string>()->value_name("<input-dir>")->default_value("."),
            "Path to input directory containing images to classify.")
        ("profiling", bpo::bool_switch(),
            "Enable profiling and create profiling report.")
        ("help", "Show help message and available command-line options.");

    bpo::options_description weights_conv_cmdline_options("Weights conversion options");
    weights_conv_cmdline_options.add_options()
        ("convert", bpo::value<std::underlying_type_t<neural::memory::format::type>>()->value_name("<format-type>"),
            "Convert weights of a neural network to given format (<format-type> represents numeric value of "
            "neural::memory::format enum).")
        ("convert_filter", bpo::value<std::string>()->value_name("<filter>"),
            "Name or part of the name of weight file(s) to be converted.\nFor example:\n  \"conv1\" - first convolution,\n  \"fc\" - every fully connected.");


    bpo::variables_map vars_map;
    store(bpo::command_line_parser(argc, argv).options(standard_cmdline_options).options(weights_conv_cmdline_options).run(), vars_map);
    notify(vars_map);

    auto exec_abs_path = bfs::system_complete(argv[0]);

    std::cout << "Usage:\n  " << exec_abs_path.stem().string() << " [standard options]\n";
    std::cout << "  " << exec_abs_path.stem().string() << " [weights conversion options]\n" << std::endl;
    std::cout << standard_cmdline_options << std::endl;
    std::cout << weights_conv_cmdline_options << std::endl;

    return vars_map;
}


int main(int argc, char *argv[])
{
    // TODO: create header file for all examples
    extern void alexnet(uint32_t, std::string, neural::engine::type,bool,bool);
    extern void convert_weights(neural::memory::format::type, std::string);

    parse_cmdline_options(argc, argv);
    return 0;

    if (argc <= 1)
    {
        std::cout <<
            R"_help_(<parameters> include:
            --batch=<value>
            size of group of images that are classified together;  large batch
            sizes have better performance
            --model=<name>
            name of network model that is used for classfication
            can be : alexnet, caffenet_float, caffenet_int16 or lenet_float
            --engine=<type>
            engine type: can be referenced or gpu
            --dump_hidden_layers
            dumps results from hidden layers
            --convert=<format_num> 
            converts weights to given format (format_num represents format enum)
            --convertion_path=<path>
            name, or substring in file name to be converte. i.e. "conv1" - first convolution, "fc" every fully connected
            --input=<directory>
            path to directory that contains images to be classfied
            --profiling
            enable profiling report)_help_";
    }
    else
    {
        bool dump_results = false;
        bool profiling = false;
        std::vector<std::string> arg;
        for (int n = 1; n<argc; ++n) arg.push_back(argv[n]);
        // parse configuration (command line and from file)
        using config_t = std::map<std::string, std::string>;
        config_t config;
        parse_parameters(config, arg);
        // Validate params and set defaults
        auto not_found = std::end(config);
        if (config.find("convert") != not_found) {
            if (config.find("convertion_path") == not_found)
                config["convertion_path"] = "";
            convert_weights(static_cast<neural::memory::format::type>(std::stoi(config["convert"])),
                config["convertion_path"]);
            return 0;
        }
        if (config.find("batch") == not_found) config["batch"] = "32";
        if (config.find("model") == not_found) config["model"] = "alexnet";
        if (config.find("engine") == not_found) config["engine"] = "referenced";
        if (config.find("dump_hidden_layers") != not_found) dump_results = true;
        if (config.find("profiling") != not_found) profiling = true;
        if (config.find("input") == not_found)
        {
            std::cout << "Directory path has to be defined using current dir"<<std::endl;
            config["input"] = "./";
        }
        if (config["model"].compare("alexnet") == 0)
        {
            try {
                alexnet(
                    std::stoi(config["batch"]),
                    config["input"],
                    config["engine"].compare("gpu") == 0 ? neural::engine::gpu : neural::engine::reference,
                    dump_results,
                    profiling);
            }
            catch (std::exception &e) {
                std::cerr << e.what() << std::endl;
            }
            catch (...) {
                std::cerr << "Unknown exceptions." << std::endl;
            }
        }
        else
        {
            std::cout << "Topology not implemented" << std::endl;
        }
    }
    return 0;
}
