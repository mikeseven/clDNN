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
#include "FreeImage_wraps.h"
#include "output_parser.h"

#include "topologies.h"

#include <boost/filesystem.hpp>

#include <iostream>

#include <regex>
#include <string>
#include <api/primitives/data.hpp>
#include <api/network.hpp>

using namespace boost::filesystem;



/// Global weak pointer to executable information.
///
/// Used to detect misuses:
///  * Using get_executable_info() before set_executable_info().
///  * Using get_executable_info() after destructon of global info object (during global destruction).
static std::weak_ptr<const executable_info> exec_info_ptr;

/// Sets information about executable based on "main"'s command-line arguments.
///
/// It works only once (if successful). Next calls to this function will not modify
/// global executable's information object.
///
/// @param argc Main function arguments count.
/// @param argv Main function argument values.
///
/// @exception std::runtime_error Main function arguments do not contain executable name.
/// @exception boost::filesystem::filesystem_error Cannot compute absolute path to executable.
void set_executable_info(int argc, const char* const argv[])
{
    if (argc <= 0)
        throw std::runtime_error("Arguments of \"main\" function do not contain executable name.");

    const std::string exec_name_arg = argv[0];
    if (exec_name_arg.empty())
        throw std::runtime_error("Arguments of \"main\" function do not contain executable name.");

    auto exec_abs_path = system_complete(exec_name_arg);

    // Safe (guarded call-once) creation of information object.
    static auto info = std::make_shared<executable_info>(
        exec_abs_path.string(), exec_abs_path.stem().string(), exec_abs_path.parent_path().string());
    exec_info_ptr = info;
}

/// Gets information about executable.
///
/// Information is fetched only if information was set using set_executable_info() and not yet
/// destroyed (during global destruction). Otherwise, exception is thrown.
///
/// @return Shared pointer pointing to valid executable information.
///
/// @exception std::runtime_error Executable information was not set or it is no longer valid.
std::shared_ptr<const executable_info> get_executable_info()
{
    auto exec_info = exec_info_ptr.lock();
    if (exec_info == nullptr)
        throw std::runtime_error("Executable information was not set or it is already destroyed.");

    return exec_info; // NRVO
}


/// Joins path using native path/directory separator.
///
/// @param parent Parent path.
/// @param child  Child part of path.
///
/// @return Joined path.
std::string join_path(const std::string& parent, const std::string& child)
{
    return (path(parent) / child).string();
}

// returns list of files (path+filename) from specified directory
static inline std::vector<std::string> get_directory_files(const std::string& images_path, const std::regex& extension)
{
    std::vector<std::string> result;

    for (const directory_entry& dir_entry : directory_iterator(images_path))
    {
        if (dir_entry.status().type() == file_type::regular_file && std::regex_match(dir_entry.path().extension().string(), extension))
        {
            result.push_back(absolute(dir_entry.path()).string());
        }
    }
    return result;
}

// returns list of files (path+filename) from specified directory
std::vector<std::string> get_directory_images(const std::string& images_path)
{
    std::regex allowed_exts("^\\.(jpe?g|png|bmp|gif|j2k|jp2|tiff)$",
                            std::regex_constants::ECMAScript | std::regex_constants::icase | std::regex_constants::optimize);
    return get_directory_files(images_path, allowed_exts);
}

// returns list of files (path+filename) from specified directory
std::vector<std::string> get_directory_weights(const std::string& images_path)
{
    std::regex allowed_exts("^\\.nnd$",
        std::regex_constants::ECMAScript | std::regex_constants::icase | std::regex_constants::optimize);
    return get_directory_files(images_path, allowed_exts);
}

void nn_data_load_from_image(
    std::string  filename,                       // Load of all data from a image filename
    cldnn::pointer<float>::iterator dst_buffer,
    uint32_t                   std_size,         // size of image both: height and width
    bool                       RGB_order)        // if true - image have RGB order, otherwise BGR
                                                 // supported formats: JPEG, J2K, JP2, PNG, BMP, WEBP, GIF, TIFF
{
    if (FIBITMAP *bitmap_raw = fi::crop_image_to_square_and_resize(fi::load_image_from_file(filename), std_size)) {
        FIBITMAP *bitmap;
        if (FreeImage_GetBPP(bitmap_raw) != 24) {
            bitmap = FreeImage_ConvertTo24Bits(bitmap_raw);
            FreeImage_Unload(bitmap_raw);
        }
        else bitmap = bitmap_raw;

        auto bytes_per_pixel = FreeImage_GetLine(bitmap) / std_size;
        auto data_buffer = dst_buffer;
        if (RGB_order) {
            for (uint32_t y = 0u; y<std_size; ++y) {
                uint8_t *pixel = FreeImage_GetScanLine(bitmap, std_size - y - 1);
                for (uint32_t x = 0u; x<std_size; ++x) {
                    *(data_buffer + 0 + x * 3 + y * 3 * std_size) = pixel[FI_RGBA_RED];
                    *(data_buffer + 1 + x * 3 + y * 3 * std_size) = pixel[FI_RGBA_GREEN];
                    *(data_buffer + 2 + x * 3 + y * 3 * std_size) = pixel[FI_RGBA_BLUE];
                    pixel += bytes_per_pixel;
                }
            }
        }
        else {
            for (uint32_t y = 0u; y<std_size; ++y) {
                uint8_t *pixel = FreeImage_GetScanLine(bitmap, std_size - y - 1);
                for (uint32_t x = 0u; x<std_size; ++x) {
                    *(data_buffer + 0 + x * 3 + y * 3 * std_size) = pixel[FI_RGBA_BLUE];
                    *(data_buffer + 1 + x * 3 + y * 3 * std_size) = pixel[FI_RGBA_GREEN];
                    *(data_buffer + 2 + x * 3 + y * 3 * std_size) = pixel[FI_RGBA_RED];
                    pixel += bytes_per_pixel;
                }
            }
        }
        FreeImage_Unload(bitmap);
    }
}

static half_t convert_pixel_channel_to_half(uint8_t val)
{
#if defined HALF_HALF_HPP
    return val;
#else
    if (!val)
        return half_t(0x0000U);

    if (val >> 4) // 4..7
    {
        if (val >> 6) // 6..7
        {
            return (val & 0x80)
                ? half_t(0x5800U | ((val & 0x7FU) << 3))
                : half_t(0x5400U | ((val & 0x3FU) << 4));
        }
        else //  4..5
        {
            return (val & 0x20)
                ? half_t(0x5000U | ((val & 0x1FU) << 5))
                : half_t(0x4C00U | ((val & 0x0FU) << 6));
        }
    }
    else // 0..3
    {
        if (val >> 2) // 2..3
        {
            return (val & 0x08)
                ? half_t(0x4800U | ((val & 0x07U) << 7))
                : half_t(0x4400U | ((val & 0x03U) << 8));
        }
        else // 0..1
        {
            return (val & 0x02)
                ? half_t(0x4000U | ((val & 0x01U) << 9))
                : half_t(0x3C00U);
        }
    }
#endif
}

void nn_data_load_from_image(
    std::string  filename,                       // Load of all data from a image filename
    cldnn::pointer<half_t>::iterator dst_buffer,
    uint32_t                   std_size,         // size of image both: height and width
    bool                       RGB_order)        // if true - image have RGB order, otherwise BGR
                                                 // supported formats: JPEG, J2K, JP2, PNG, BMP, WEBP, GIF, TIFF
{
    if (FIBITMAP *bitmap_raw = fi::crop_image_to_square_and_resize(fi::load_image_from_file(filename), std_size)) {
        FIBITMAP *bitmap;
        if (FreeImage_GetBPP(bitmap_raw) != 24) {
            bitmap = FreeImage_ConvertTo24Bits(bitmap_raw);
            FreeImage_Unload(bitmap_raw);
        }
        else bitmap = bitmap_raw;

        auto bytes_per_pixel = FreeImage_GetLine(bitmap) / std_size;
        auto data_buffer = dst_buffer;
        if (RGB_order) {
            for (uint32_t y = 0u; y<std_size; ++y) {
                uint8_t *pixel = FreeImage_GetScanLine(bitmap, std_size - y - 1);
                for (uint32_t x = 0u; x<std_size; ++x) {
                    *(data_buffer + 0 + x * 3 + y * 3 * std_size) = convert_pixel_channel_to_half(pixel[FI_RGBA_RED]);
                    *(data_buffer + 1 + x * 3 + y * 3 * std_size) = convert_pixel_channel_to_half(pixel[FI_RGBA_GREEN]);
                    *(data_buffer + 2 + x * 3 + y * 3 * std_size) = convert_pixel_channel_to_half(pixel[FI_RGBA_BLUE]);
                    pixel += bytes_per_pixel;
                }
            }
        }
        else {
            for (uint32_t y = 0u; y<std_size; ++y) {
                uint8_t *pixel = FreeImage_GetScanLine(bitmap, std_size - y - 1);
                for (uint32_t x = 0u; x<std_size; ++x) {
                    *(data_buffer + 0 + x * 3 + y * 3 * std_size) = convert_pixel_channel_to_half(pixel[FI_RGBA_BLUE]);
                    *(data_buffer + 1 + x * 3 + y * 3 * std_size) = convert_pixel_channel_to_half(pixel[FI_RGBA_GREEN]);
                    *(data_buffer + 2 + x * 3 + y * 3 * std_size) = convert_pixel_channel_to_half(pixel[FI_RGBA_RED]);
                    pixel += bytes_per_pixel;
                }
            }
        }
        FreeImage_Unload(bitmap);
    }
}

// i am not sure what is better: pass memory as primitive where layout, ptr and size are included
// or pass as separate parameters to avoid including neural.h in common tools?
template <typename MemElemTy>
void load_images_from_file_list(
    const std::vector<std::string>& images_list,
    cldnn::memory& memory)
{
    auto memory_layout = memory.get_layout();
    auto dst_ptr = memory.pointer<MemElemTy>();
    auto it = dst_ptr.begin();

    auto dim = memory_layout.size.spatial;

    if(memory_layout.size.format != cldnn::format::byxf) throw std::runtime_error("Only bfyx format is supported as input to images from files");

    if(!cldnn::data_type_match<MemElemTy>(memory_layout.data_type))
        throw std::runtime_error("Memory format expects different type of elements than specified");
    auto single_image_size = dim[0] * dim[1] * 3;
    for (auto img : images_list)
    {
        // "false" because we want to load images in BGR format because weights are in BGR format and we don't want any conversions between them.
        nn_data_load_from_image(img, it, dim[0], false);
        it += single_image_size;
    }
}

// Explicit instantiation of all used template function instances used in examples.
template void load_images_from_file_list<float>(const std::vector<std::string>&, cldnn::memory&);
template void load_images_from_file_list<half_t>(const std::vector<std::string>&, cldnn::memory&);

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
cldnn::network build_network(const cldnn::engine& engine, const cldnn::topology& topology, const execution_params &ep)
{
    if (ep.print_type == Verbose)
    {
        std::cout << "GPU Program compilation started" << std::endl;
    }

    cldnn::instrumentation::timer<> timer_compilation;

    cldnn::build_options options;

    //TODO set proper network build options
    options.set_option(cldnn::build_option::optimize_data(true));
    options.set_option(cldnn::build_option::profiling(ep.profiling));
    options.set_option(cldnn::build_option::debug(ep.dump_hidden_layers || ep.profiling));

    std::vector<cldnn::primitive_id> outputs{ "output" };
    if (!ep.dump_layer_name.empty())  outputs.push_back(ep.dump_layer_name);
    if (!ep.run_single_layer.empty()) outputs.push_back(ep.run_single_layer);
    options.set_option(cldnn::build_option::outputs(outputs));
    try 
    {
        cldnn::program program(engine, topology, options);
        auto compile_time = timer_compilation.uptime();

        if (ep.print_type == Verbose)
        {
            std::cout << "GPU Program compilation finished in " << instrumentation::to_string(compile_time) << std::endl;
            std::cout << "Network allocation started" << std::endl;
        }

        cldnn::network network(program);
        auto allocation_time = timer_compilation.uptime() - compile_time;
        
        if (ep.print_type == Verbose)
        {
            std::cout << "Network allocation finished in " << instrumentation::to_string(allocation_time) << std::endl;
        }

        return network;
    }
    catch (const cldnn::error &err)
    {
        std::cout << "ERROR: " << err.what() << std::endl;
        if (err.status() == CLDNN_OUT_OF_RESOURCES)
            std::cout << "HINT: Try to use smaller batch size" << std::endl;
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

std::chrono::nanoseconds execute_topology(cldnn::network network,
                                          const execution_params &ep,
                                          CIntelPowerGadgetLib& energyLib,
                                          cldnn::memory& output)
{
    bool log_energy = ep.perf_per_watt && energyLib.IntelEnergyLibInitialize();

    if (ep.print_type == Verbose)
    {
        std::cout << "Start execution";
        if (ep.loop > 1)
        {
            std::cout << " in a loop " << ep.loop << " times:";
        }
        std::cout << std::endl;
    }

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

    decltype(network.execute()) outputs;

    cldnn::instrumentation::timer<> timer_execution;

    for (decltype(ep.loop) i = 0; i < ep.loop; i++)
    {
        outputs = network.execute();
        if (log_energy)
            energyLib.ReadSample();
    }

    //GPU primitives scheduled in unblocked manner
    auto scheduling_time(timer_execution.uptime());

    //OCL buffers mapping blocks until all primitives are completed
    output = outputs.at("output").get_memory();
    auto execution_time(timer_execution.uptime());

    if (log_energy)
    {
        energyLib.ReadSample();
        energyLib.StopLog();
    }

    if (ep.print_type == Verbose)
    {
        std::cout << ep.topology_name << " scheduling finished in " << instrumentation::to_string(scheduling_time) << std::endl;
        std::cout << ep.topology_name << " execution finished in " << instrumentation::to_string(execution_time) << std::endl;
    }

    if (ep.dump_hidden_layers)
    {
        auto input = outputs.at("input").get_memory();
        instrumentation::logger::log_memory_to_file(input, "input0");
        for (auto& p : outputs)
        {
            instrumentation::logger::log_memory_to_file(p.second.get_memory(), p.first, ep.dump_single_batch, ep.dump_batch_id, ep.dump_single_feature, ep.dump_feature_id);
        }
        // for now its enough. rest will be done when we have equals those values
    }
    else if (!ep.dump_layer_name.empty())
    {
        auto it = outputs.find(ep.dump_layer_name);
        if(it != std::end(outputs))
        {
            instrumentation::logger::log_memory_to_file(it->second.get_memory(), it->first, ep.dump_single_batch, ep.dump_batch_id, ep.dump_single_feature, ep.dump_feature_id);
        }
        else
        {
            std::cout << "WARNING: " << ep.topology_name << " does not contain " << ep.dump_layer_name << " layer!" << std::endl;
        }
    }
    else
    {
        // We do not log results for microbench
        if (ep.topology_name != "microbench")
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

    return std::chrono::duration_cast<std::chrono::nanoseconds>(execution_time);
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

    cldnn::engine_configuration configuration(ep.profiling, ep.meaningful_kernels_names);
    cldnn::engine engine(configuration);

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

    if (ep.print_type == Verbose)
    {
        std::cout << "Building " << ep.topology_name << " started" << std::endl;
    }
    cldnn::instrumentation::timer<> timer_build;
    cldnn::layout input_layout = { ep.use_half ? cldnn::data_types::f16 : cldnn::data_types::f32, {} };
    if (ep.topology_name == "alexnet")
        primitives = build_alexnet(ep.weights_dir, engine, input_layout, gpu_batch_size);
    else if (ep.topology_name == "vgg16" || ep.topology_name == "vgg16_face")
        primitives = build_vgg16(ep.weights_dir, engine, input_layout, gpu_batch_size);
    else if (ep.topology_name == "googlenet")
        primitives = build_googlenetv1(ep.weights_dir, engine, input_layout, gpu_batch_size);
    else if (ep.topology_name == "gender")
        primitives = build_gender(ep.weights_dir, engine, input_layout, gpu_batch_size);
    else if (ep.topology_name == "microbench")
        primitives = build_microbench(ep.weights_dir, engine, input_layout, gpu_batch_size);
    else if(ep.topology_name == "squeezenet")
        primitives = build_squeezenet(ep.weights_dir, engine, input_layout, gpu_batch_size);
    else
        throw std::runtime_error("Topology \"" + ep.topology_name + "\" not implemented!");

    auto build_time = timer_build.uptime();

    if (ep.print_type == Verbose)
    {
        std::cout << "Building " << ep.topology_name << " finished in " << instrumentation::to_string(build_time) << std::endl;
    }

    auto network = build_network(engine, primitives, ep);
    auto input = cldnn::memory::allocate(engine, input_layout);

    //TODO check if we can define the 'empty' memory
    float zero = 0;
    cldnn::layout zero_layout( cldnn::data_types::f32, {cldnn::format::x, {1}} );
    auto output = cldnn::memory::attach(zero_layout, &zero, 1);

    if (ep.topology_name != "microbench")
    {
        auto neurons_list_filename = "names.txt";
        if (ep.topology_name == "vgg16_face")
            neurons_list_filename = "vgg16_face.txt";
        else if (ep.topology_name == "gender")
            neurons_list_filename = "gender.txt";
        auto img_list = get_directory_images(ep.input_dir);
        if (img_list.empty())
            throw std::runtime_error("specified input images directory is empty (does not contain image data)");

        auto number_of_batches = (img_list.size() % batch_size == 0)
            ? img_list.size() / batch_size : img_list.size() / batch_size + 1;
        std::vector<std::string> images_in_batch;
        auto images_list_iterator = img_list.begin();
        auto images_list_end = img_list.end();
        for (decltype(number_of_batches) batch = 0; batch < number_of_batches; batch++)
        {
            images_in_batch.clear();
            for (uint32_t i = 0; i < batch_size && images_list_iterator != images_list_end; ++i, ++images_list_iterator)
            {
                images_in_batch.push_back(*images_list_iterator);
            }

            // load croped and resized images into input
            if (ep.use_half)
            {
                load_images_from_file_list<half_t>(images_in_batch, input);
            }
            else
            {
                load_images_from_file_list(images_in_batch, input);
            }

            network.set_input_data("input", input);
            auto time = execute_topology(network, ep, energyLib, output);

            auto time_in_sec = std::chrono::duration_cast<std::chrono::duration<double, std::chrono::seconds::period>>(time).count();

            output_file.batch(output, join_path(get_executable_info()->dir(), neurons_list_filename), images_in_batch, ep.print_type);

            if (time_in_sec != 0.0)
            {
                if (ep.print_type != ExtendedTesting)
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
    else {
        auto time = execute_topology(network, ep, energyLib, output);
        auto time_in_sec = std::chrono::duration_cast<std::chrono::duration<double, std::chrono::seconds::period>>(time).count();
        if (time_in_sec != 0.0)
        {
            if (ep.print_type != ExtendedTesting)
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
