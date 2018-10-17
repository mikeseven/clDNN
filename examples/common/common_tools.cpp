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
#include "instrumentation.h"
#include "common_tools.h"
#include "image_toolkit.h"
#include "output_parser.h"

#include <boost/filesystem.hpp>
#include <boost/optional.hpp>

#include <iostream>

#include <algorithm>
#include <cstdint>
#include <regex>
#include <string>
#include <limits>
#include <api/CPP/data.hpp>
#include <api/CPP/network.hpp>
#include "file.h"

using namespace boost::filesystem;
using namespace cldnn::utils::examples;


//get_model_name (subtract _train or _test from topology string)
std::string get_model_name(const std::string& topology_name)
{
    std::string model_name = topology_name;
    if (model_name.find("_train") != std::string::npos)
        model_name = model_name.substr(0, model_name.find("_train"));
    else if (model_name.find("_test") != std::string::npos)
        model_name = model_name.substr(0, model_name.find("_test"));
    return model_name;
}

void compute_image_mean(const execution_params &ep, cldnn::engine& engine, bool use_cifar10)
{
    const uint32_t channels_num = 3;
    const uint32_t size_x = use_cifar10 ? 32 : 256;
    const uint32_t size_y = use_cifar10 ? 32 : 256;

    auto input_list = list_input_files(ep.input_dir);

    std::vector<std::string> requested_images;
    if(!use_cifar10)
        for (uint32_t i = ep.image_offset; i < ep.image_offset + ep.image_number; i++)
            requested_images.push_back(input_list[i]);

    auto memory_layout = cldnn::layout({ cldnn::data_types::f32, cldnn::format::byxf,cldnn::tensor{ 1, (cldnn::tensor::value_type)channels_num, (cldnn::tensor::value_type)size_x, (cldnn::tensor::value_type)size_y } });

    auto memory = cldnn::memory::allocate(engine, memory_layout);

    auto dst_ptr = memory.pointer<float>();

    if (memory_layout.format != cldnn::format::byxf) throw std::runtime_error("Only byxf format is supported as input to images from files");

    if (!cldnn::data_type_match<float>(memory_layout.data_type))
        throw std::runtime_error("Memory format expects different type of elements than specified");
    
    const uint32_t spatial_size = size_x * size_y;
    auto single_image_size = spatial_size * channels_num;
    std::vector<float> img_sum(single_image_size, 0);
    std::vector<float> img_tmp(single_image_size, 0);
    auto img_tmp_it = img_tmp.begin();

    if (!use_cifar10)
    {
        for (auto img : requested_images)
        {
            // "false" because we want to load images in BGR format because weights are in BGR format and we don't want any conversions between them.
            itk::load_image_data(img, img_tmp_it, false, size_x, 256);

            for (uint32_t i = 0; i < img_sum.size(); i++)
                img_sum[i] += img_tmp[i];
        }
    }
    else
    {
        std::ifstream rfile(get_image_file("data_batch.bin", input_list), std::ios::binary);

        const uint32_t img_spatial = 32;
        const uint32_t img_size = 1 + img_spatial * img_spatial * 3; //1-byte for label, 32*32*3 bytes for image data;

        if (rfile)
        {
            auto images_number = ep.image_number;
            std::vector<unsigned char> tmpBuffer(img_size * images_number);

            rfile.seekg(ep.image_offset * img_size, rfile.cur);
            rfile.read(reinterpret_cast<char *>(&tmpBuffer[0]), img_size * images_number);
            rfile.close();

            //read in image data
            for (uint32_t j = 0; j < images_number; ++j)
            {
                for (uint32_t y = 0u; y < img_spatial; ++y)
                {
                    for (uint32_t x = 0u; x < img_spatial; ++x)
                    {
                        img_tmp[y * 3 * img_spatial + x * 3 + 2] = static_cast<float>(tmpBuffer[j * img_size + 1 + y * img_spatial + x + 0]);
                        img_tmp[y * 3 * img_spatial + x * 3 + 1] = static_cast<float>(tmpBuffer[j * img_size + 1 + y * img_spatial + x + 1024]);
                        img_tmp[y * 3 * img_spatial + x * 3 + 0] = static_cast<float>(tmpBuffer[j * img_size + 1 + y * img_spatial + x + 2048]);
                    }
                }

                for (uint32_t i = 0; i < img_sum.size(); i++)
                    img_sum[i] += img_tmp[i];
            }
        }
        else
            throw std::runtime_error("Cannot read image cifar10 image file.");
    }

    for (uint32_t i = 0; i < img_sum.size(); i++)
        img_sum[i] /= ep.image_number;

    //per channel mean
    std::vector<float> mean_values(channels_num, 0);
    for (uint32_t i = 0; i < channels_num; i++)
    {
        for (uint32_t j = 0; j < spatial_size; j++)
            mean_values[i] += img_sum[i + j * channels_num];

        mean_values[i] /= spatial_size;

        for (uint32_t j = 0; j < spatial_size; j++)
            dst_ptr[i * spatial_size + j] = mean_values[i];

    }

    file::serialize_train(memory, join_path(ep.weights_dir, "imagenet_mean.nnd"));
}

void print_profiling_table(std::ostream& os, const std::vector<cldnn::instrumentation::profiling_info>& profiling_info) {
    if (profiling_info.size() == 0)
        return;

    const size_t numbers_width = 10;

    os << "Kernels profiling info (in microseconds): \n\n";

    // build column headers
    std::vector<std::string> column_headers;
    for (auto& info : profiling_info) {
        for (auto& interval : info.intervals) {
            if (std::count(column_headers.begin(), column_headers.end(), interval.name) == 0) {
                column_headers.push_back(interval.name);
            }
        }
    }

    size_t action_column_len = 0;
    for (auto& info : profiling_info) {
        action_column_len = std::max(action_column_len, info.name.length());
    }

    // print column headers
    auto column_width = std::max(action_column_len, numbers_width);
    std::string separation_line(column_width, '-');
    os << std::setw(column_width) << std::left << "Action";
    for (auto& header : column_headers) {
        column_width = std::max(header.length(), numbers_width);
        separation_line += "+" + std::string(column_width, '-');
        os << "|"
            << std::setw(column_width) << std::right
            << header;
    }
    os << "\n";

    std::chrono::nanoseconds total(0);

    // print rows
    size_t row_num = 0;
    for (auto& info : profiling_info) {
        if ((row_num++) % 4 == 0) {
            os << separation_line << "\n";
        }
        os << std::setw(action_column_len) << std::left << info.name;
        // prepare values per column
        std::vector<double> values(column_headers.size(), 0.0);
        for (auto& interval : info.intervals) {
            auto value = interval.value->value();
            total += value;
            auto value_d = std::chrono::duration_cast<std::chrono::duration<double, std::chrono::microseconds::period>>(value).count();
            auto column_index = std::find(column_headers.begin(), column_headers.end(), interval.name) - column_headers.begin();
            values[column_index] = value_d;
        }
        // print values in columns
        for (size_t i = 0; i < values.size(); ++i)
        {
            auto& header = column_headers[i];
            os << "|"
                << std::setw(std::max(header.length(), numbers_width)) << std::right
                << std::setprecision(3) << std::fixed << values[i];
        }
        os << "\n";
    }
    os << "\nTotal profiled time: " << instrumentation::to_string(total) << std::endl;
}

// Create worker
cldnn::network build_network(const cldnn::engine& engine, const cldnn::topology& topology, const execution_params &ep, const std::vector<cldnn::primitive_id> &output_ids)
{
    if (ep.print_type == print_type::verbose)
    {
        std::cout << "GPU Program compilation started" << std::endl;
    }

    cldnn::instrumentation::timer<> timer_compilation;

    cldnn::build_options options;

    //TODO set proper network build options
    if(ep.topology_name == "vgg16_train")
        options.set_option(cldnn::build_option::optimize_data(false));
    else
        options.set_option(cldnn::build_option::optimize_data(true));
    options.set_option(cldnn::build_option::debug(ep.dump_hidden_layers || ep.profiling));
    options.set_option(cldnn::build_option::serialize_network(ep.serialization));

    if (ep.dump_graphs)
    {
        std::string err;
        auto graphs_dumps_dir = instrumentation::logger::create_graphs_dumps_dir(err);
        if (err.empty())
            options.set_option(cldnn::build_option::graph_dumps_dir(graphs_dumps_dir));
        else
        {
            std::cout << "Could not create requested directory for graph dumps: '" << graphs_dumps_dir << "'\n    error:\n"
                << err << "\n    -- dumping will be disabled" << std::endl;
        }
    }

    std::vector<cldnn::primitive_id> outputs(output_ids);

    if (!ep.run_until_primitive_name.empty())
    {
        outputs.push_back(ep.run_until_primitive_name); //set the user custom primitive as output (works only while not in debug mode, because in debug mode every primitive is an output)
        if(ep.dump_hidden_layers)
            throw std::runtime_error("ERROR: Can't dump hidden layers when custom output is set.");
    }

    if (!ep.dump_layer_name.empty())
    {
        if (ep.topology_name == "microbench_conv")
        {
            for (auto prim_id : topology.get_primitive_ids())
            {
                if (prim_id.find("_weights") == std::string::npos &&
                    prim_id.find("_bias") == std::string::npos &&
                    prim_id.find("_input") == std::string::npos)
                    outputs.push_back(prim_id);
            }
        }
        else
            outputs.emplace_back("output");

        outputs.push_back(ep.dump_layer_name);
    }

    options.set_option(cldnn::build_option::outputs(outputs));

    try
    {
        cldnn::program program(engine, topology, options);
        auto compile_time = timer_compilation.uptime();

        if (ep.print_type == print_type::verbose)
        {
            std::cout << "GPU Program compilation finished in " << instrumentation::to_string(compile_time) << std::endl;
            std::cout << "Network allocation started" << std::endl;
        }

        cldnn::network network(program);

        auto allocation_time = timer_compilation.uptime() - compile_time;

        if (ep.print_type == print_type::verbose)
        {
            std::cout << "Network allocation finished in " << instrumentation::to_string(allocation_time) << std::endl;
        }

        if (ep.print_type == print_type::extended_testing)
        {
            std::cout << "All primitives information: " << std::endl;
            std::vector<std::string> primitives_id = topology.get_primitive_ids();
            std::string primitive_info = "";
            for (auto& prim : primitives_id) //loop through primitives_id vector, so we print information about all primitives
            {
                primitive_info = network.get_primitive_info(prim);
                std::cout << primitive_info << std::endl;
            }

        }

        return network;
    }
    catch (const cldnn::error &err)
    {
        std::cout << "ERROR: " << err.what() << std::endl;
        switch (err.status())
        {
        case CLDNN_OUT_OF_RESOURCES:
            std::cout << "HINT: Try to use smaller batch size" << std::endl;
            break;
        case CLDNN_ALLOC_SIZE_EXCEEDED:
            std::cout << "HINT: Try to use smaller buffers. Max memory alloc size per object (CL_DEVICE_MAX_MEM_ALLOC_SIZE) is " << engine.get_info().max_alloc_mem_size << " in bytes." << std::endl;
            break;
        case CLDNN_GLOBAL_SIZE_EXCEEDED:
            std::cout << "HINT: Try to use smaller amount of data. Size of global device memory (CL_DEVICE_GLOBAL_MEM_SIZE) is " << engine.get_info().max_global_mem_size << " in bytes." << std::endl;
            break;
        default:
            break;
        }
        throw;
    }
    catch (...)
    {
        std::cout << "ERROR: Network build failed" << std::endl;
        throw;
    } 
}

uint32_t get_next_nearest_power_of_two(int number)
{
    int tmp_number = number;
    uint32_t power = 1;
    while (tmp_number >>= 1) power <<= 1;
    if (number % power == 0)
        return power;
    return power << 1;
}

uint32_t get_gpu_batch_size(int number)
{
    uint32_t nearest_power_of_two = get_next_nearest_power_of_two(number);
    // we do not support batch of size 2 or 4 so we need to get rid of those
    if (nearest_power_of_two < 8 && nearest_power_of_two > 1)
        return 8;
    return nearest_power_of_two;
}

bool do_log_energy(const execution_params &ep, CIntelPowerGadgetLib& energyLib) 
{ 
    bool log_energy = ep.perf_per_watt && energyLib.IntelEnergyLibInitialize();
    if (log_energy)
    {
        try {
            wchar_t fileName[] = L"power_log.csv";
            energyLib.StartLog(fileName);
        }
        catch (...)
        {
            throw std::runtime_error("ERROR: can't open power_log.csv file");
        }
    }
    return log_energy;
}

std::chrono::nanoseconds execute_cnn_topology(cldnn::network network,
                                                const execution_params &ep,
                                                CIntelPowerGadgetLib& energyLib,
                                                cldnn::memory& output,
                                                const uint32_t iteration,
                                                const uint32_t execution_count)
{
    bool log_energy = do_log_energy(ep, energyLib);

    if (ep.print_type == print_type::verbose)
    {
        std::cout << "Start execution";
        if (ep.loop > 1)
        {
            std::cout << " in a loop " << ep.loop << " times:";
        }
        std::cout << std::endl;
    }
    decltype(network.execute()) outputs;
    cldnn::instrumentation::timer<> timer_execution;

    for (decltype(ep.loop) i = 0; i < ep.loop; i++)
    {
        outputs = network.execute();
        if (log_energy)
            energyLib.ReadSample();
    }

    return get_execution_time(timer_execution, ep, output, outputs, log_energy, energyLib, iteration, execution_count);
}

std::chrono::nanoseconds get_execution_time(cldnn::instrumentation::timer<>& timer_execution,
                                          const execution_params &ep,
                                          cldnn::memory& output,
                                          cldnn_output& outputs,
                                          bool log_energy,
                                          CIntelPowerGadgetLib& energyLib,
                                          const uint32_t iteration,
                                          const uint32_t execution_count)
{

    //GPU primitives scheduled in unblocked manner
    auto scheduling_time(timer_execution.uptime());

    //OCL buffers mapping blocks until all primitives are completed
    if (ep.topology_name != "microbench_conv" && ep.topology_name != "microbench_lstm")
    {
        std::string output_primitve_id = ep.run_until_primitive_name.empty() ? "output" : ep.run_until_primitive_name;
        output = outputs.at(output_primitve_id).get_memory();
    }

    if (ep.topology_name == "lenet_train" || ep.topology_name == "vgg16_train" || ep.topology_name == "resnet50_train")
    {
        if (iteration % ep.train_snapshot == 0 || iteration == execution_count)
        {
            for (auto& p : outputs)
            {
                file::serialize_train(p.second.get_memory(), join_path(ep.weights_dir, p.first));
            }
            std::cout << "Weights snapshot done." << std::endl;

            //make snapshot of execution data, from which training can be continued
            auto lr_string = std::to_string(ep.learning_rate);
            lr_string = lr_string.substr(lr_string.find_last_of(".") + 1);
            file::save_train_iteration(join_path(ep.weights_dir, "train_iteration.txt"), iteration);
        }
        output = outputs.at("softmax_fp32").get_memory();
    }

    auto execution_time(timer_execution.uptime());

    if (log_energy)
    {
        energyLib.ReadSample();
        energyLib.StopLog();
    }

    if (ep.print_type == print_type::verbose)
    {
        std::cout << ep.topology_name << " scheduling finished in " << instrumentation::to_string(scheduling_time) << std::endl;
        std::cout << ep.topology_name << " execution finished in " << instrumentation::to_string(execution_time) << std::endl;
    }

    make_instrumentations(ep, output, outputs);

    return std::chrono::duration_cast<std::chrono::nanoseconds>(execution_time);
}

void make_instrumentations(const execution_params& ep, cldnn::memory& output, std::map<cldnn::primitive_id, cldnn::network_output>& outputs)
{

    if (ep.dump_hidden_layers)
    {
        auto input = outputs.at("input").get_memory(); 
        instrumentation::logger::log_memory_to_file(input, "input0");
        for (auto& p : outputs)
        {
            instrumentation::logger::log_memory_to_file(p.second.get_memory(), p.first, ep.dump_single_batch, ep.dump_batch_id, ep.dump_single_feature, ep.dump_feature_id);
        }
        // for now its enough. rest will be done when we have equals those values
        for (auto& p : outputs)
        {
            if(p.first.find("output") != std::string::npos)
                instrumentation::logger::log_memory_to_file(p.second.get_memory(), p.first, ep.dump_single_batch, ep.dump_batch_id, ep.dump_single_feature, ep.dump_feature_id);
        }
    }
    else if (!ep.dump_layer_name.empty())
    {
        auto it = outputs.find(ep.dump_layer_name);
        if (it != std::end(outputs))
        {
            if (!ep.dump_weights)
                instrumentation::logger::log_memory_to_file(it->second.get_memory(), it->first, ep.dump_single_batch, ep.dump_batch_id, ep.dump_single_feature, ep.dump_feature_id);
            else
                instrumentation::logger::log_weights_to_file(it->second.get_memory(), it->first);
        }
        else
        {
            std::cout << "WARNING: " << ep.topology_name << " does not contain " << ep.dump_layer_name << " layer!" << std::endl;
        }
    }
    else
    {
        // We do not log results for microbench_conv.
        if (ep.topology_name != "microbench_conv" && ep.topology_name != "microbench_lstm")
        {
            instrumentation::logger::log_memory_to_file(output, "final_result");
        }
    }

    if (ep.profiling)
    {
        std::vector<cldnn::instrumentation::profiling_info> profiling_table;
        for (auto& p : outputs)
        {
            profiling_table.push_back({ p.first, p.second.get_event().get_profiling_info() });
        }
        print_profiling_table(std::cout, profiling_table);
    }
}
