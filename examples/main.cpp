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
#include <iostream>
#include <string>
#include <algorithm>
#include <regex> 

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

int main(int argc, char *argv[])
{
    extern void Alexnet(uint32_t batch_size, std::string img_dir);

    if (argc <= 1)
    {
        std::cout <<
            R"_help_(<parameters> include:
            --batch = <value>
            size of group of images that are classified together;  large batch
            sizes have better performance
            --model = <name>
            name of network model that is used for classfication
            can be : alexnet, caffenet_float, caffenet_int16 or lenet_float
            --input = <directory>
            path to directory that contains images to be classfied)_help_";
    }
    else
    {
        std::vector<std::string> arg;
        for (int n = 1; n<argc; ++n) arg.push_back(argv[n]);
        // parse configuration (command line and from file)
        using config_t = std::map<std::string, std::string>;
        config_t config;
        parse_parameters(config, arg);
        // Validate params and set defaults
        auto not_found = std::end(config);
        if (config.find("batch") == not_found) config["batch"] = "32";
        if (config.find("model") == not_found) config["model"] = "alexnet";
        if (config.find("input") == not_found)
        {
            std::cout << "Directory path has to be defined using current dir"<<std::endl;
            config["input"] = "./";
        }
        if (config["model"].compare("alexnet") == 0)
        {
            try {
                Alexnet(std::stoi(config["batch"]), config["input"]);
            }
            catch (std::exception &e) {
                std::cerr << e.what();
            }
            catch (...) {
                std::cerr << "Unknown exceptions.";
            }
        }
        else
        {
            std::cout << "Topology not implemented" << std::endl;
        }
    }
    return 0;
}
