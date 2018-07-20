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
#include "lstm_utils.h"

#include "topologies.h"

#include <boost/filesystem.hpp>
#include <boost/optional.hpp>

#include <iostream>

#include <regex>
#include <string>
#include <algorithm>
#include <limits>
#include <api/CPP/data.hpp>
#include <api/CPP/network.hpp>
#include "file.h"

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
std::vector<std::string> get_input_list(const std::string& images_path)
{
    std::regex allowed_exts("^\\.(jpe?g|png|bmp|gif|j2k|jp2|tiff|txt|idx3\\-ubyte)$",
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

    if(memory_layout.format != cldnn::format::byxf) throw std::runtime_error("Only byxf format is supported as input to images from files");

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

uint32_t swap_endian(uint32_t val) {
    val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
    return (val << 16) | (val >> 16);
}

// Explicit instantiation of all used template function instances used in examples.
template void load_images_from_file_list<float>(const std::vector<std::string>&, cldnn::memory&);
template void load_images_from_file_list<half_t>(const std::vector<std::string>&, cldnn::memory&);

template <typename MemElemTy>
void load_data_from_file_list_lenet(
    const std::vector<std::string>& images_list,
    cldnn::memory& memory, const uint32_t images_offset, const uint32_t images_number, const bool train, cldnn::memory& memory_labels)
{
    auto dst_ptr = memory.pointer<MemElemTy>();
    auto it = dst_ptr.begin();

    auto memory_layout = memory.get_layout();
    int count = 0;
    if (!cldnn::data_type_match<MemElemTy>(memory_layout.data_type))
        throw std::runtime_error("Memory format expects different type of elements than specified");

    //we use mnist image set for testing and training lenet. The images file from mnist are hardcoded to:
    // - train-images.idx3-ubyte for training
    // - t10k-images.idx3-ubyte for testing
    std::string img_name = "";
    if (!train)
        img_name = "t10k-images.idx3-ubyte";
    else
        img_name = "train-images.idx3-ubyte";

    std::string img = "";
    for (auto img_from_list : images_list)
    {
        if (img_from_list.find(img_name) != std::string::npos)
        {
            img = img_from_list;
            break;
        }
    }

    if(img == "")
        throw std::runtime_error("Image file from Lenet not found.");

    std::ifstream rfile(img, std::ios::binary);

    if (rfile)
    {
        // Read the magic and the meta data
        uint32_t magic;
        uint32_t num_items;
        uint32_t rows;
        uint32_t cols;

        rfile.read(reinterpret_cast<char*>(&magic), 4);
        magic = swap_endian(magic);
        if (magic != 2051)
            throw std::runtime_error("Incorrect image file magic.");
        rfile.read(reinterpret_cast<char*>(&num_items), 4);
        num_items = swap_endian(num_items);
        rfile.read(reinterpret_cast<char*>(&rows), 4);
        rows = swap_endian(rows);
        rfile.read(reinterpret_cast<char*>(&cols), 4);
        cols = swap_endian(cols);
        auto img_size = rows * cols;

        std::vector<unsigned char> tmpBuffer(img_size * images_number);

        rfile.seekg(images_offset * img_size, rfile.cur);
        rfile.read(reinterpret_cast<char *>(&tmpBuffer[0]), img_size * images_number);
        rfile.close();

        for (uint32_t i = 0; i < img_size * images_number; ++i) {
            *it = static_cast<MemElemTy>(tmpBuffer[i]);
            it++;
        }

        //read in labels
        auto labels_ptr = memory_labels.pointer<MemElemTy>();
        auto labels_it = labels_ptr.begin();
                
        std::string img_ext = "-images.idx3-ubyte";
        auto labels_file = img.substr(0, img.length() - img_ext.length()) + "-labels.idx1-ubyte";
        std::ifstream rfile_labels(labels_file, std::ios::binary);

        if (rfile_labels)
        {
            // Read the magic and the meta data
            uint32_t magic;
            uint32_t num_items;

            rfile_labels.read(reinterpret_cast<char*>(&magic), 4);
            magic = swap_endian(magic);
            if (magic != 2049)
                throw std::runtime_error("Incorrect image file magic.");
            rfile_labels.read(reinterpret_cast<char*>(&num_items), 4);
            num_items = swap_endian(num_items);

            std::vector<unsigned char> tmpBuffer(sizeof(char)*images_number);

            rfile_labels.seekg(images_offset, rfile_labels.cur);
            rfile_labels.read(reinterpret_cast<char *>(&tmpBuffer[0]), images_number);
            rfile_labels.close();

            for (uint32_t i = 0; i < images_number; ++i) {
                *labels_it = static_cast<MemElemTy>(tmpBuffer[i]);
                labels_it++;
            }
        }
        else
            throw std::runtime_error("Cannot read labels for lenet topology.");
        count++;
    }
    else
        throw std::runtime_error("Cannot read image for lenet topology.");
}

template void load_data_from_file_list_lenet<float>(const std::vector<std::string>&, cldnn::memory&, const uint32_t, const uint32_t, const bool, cldnn::memory&);
template void load_data_from_file_list_lenet<half_t>(const std::vector<std::string>&, cldnn::memory&, const uint32_t, const uint32_t, const bool, cldnn::memory&);

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
    if (ep.topology_name == "lenet_train")
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

    std::vector<cldnn::primitive_id> outputs(0);

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
            outputs.push_back("output");

        outputs.push_back(ep.dump_layer_name);
    }

    if (ep.topology_name == "lenet_train")
    {
        //set weights outputs for lenet training
        outputs.push_back("conv1_bias.nnd");
        outputs.push_back("conv1_weights.nnd");
        outputs.push_back("conv2_bias.nnd");
        outputs.push_back("conv2_weights.nnd");
        outputs.push_back("ip1_bias.nnd");
        outputs.push_back("ip1_weights.nnd");
        outputs.push_back("ip2_bias.nnd");
        outputs.push_back("ip2_weights.nnd");

        outputs.push_back("conv1_bias_prev.nnd");
        outputs.push_back("conv1_weights_prev.nnd");
        outputs.push_back("conv2_bias_prev.nnd");
        outputs.push_back("conv2_weights_prev.nnd");
        outputs.push_back("ip1_bias_prev.nnd");
        outputs.push_back("ip1_weights_prev.nnd");
        outputs.push_back("ip2_bias_prev.nnd");
        outputs.push_back("ip2_weights_prev.nnd");

        outputs.push_back("softmax");
        outputs.push_back("ip2_grad_weights");
        outputs.push_back("ip1_grad_weights");
        outputs.push_back("conv2_grad_weights");
        outputs.push_back("output");
    }

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

        if (ep.print_type == ExtendedTesting)
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


template<typename MemElemTy = float>
void prepare_data_for_lstm(lstm_utils& lstm_data, const std::vector<std::string>& input_files, const std::string& vocabulary_file)
{
    lstm_data.pre_processing(input_files, vocabulary_file);
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
                                                cldnn::memory& output, cldnn::memory* labels = nullptr)
{
    bool log_energy = do_log_energy(ep, energyLib);

    if (ep.print_type == Verbose)
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

    return get_execution_time(timer_execution, ep, output, outputs, log_energy, energyLib);
}

std::chrono::nanoseconds execute_rnn_topology(cldnn::network network,
    const execution_params &ep,
    CIntelPowerGadgetLib& energyLib,
    cldnn::memory& output,
    cldnn::memory& input,
    lstm_utils& lstm_data)
{
    bool log_energy = do_log_energy(ep, energyLib);

    if (ep.print_type == Verbose)
    {
        std::cout << "Start execution.";
        if (ep.loop > 1)
        {
            std::cout << " Will make " << ep.loop << " predictions.";
        }
        std::cout << std::endl;
    }
    decltype(network.execute()) outputs;
    cldnn::instrumentation::timer<> timer_execution;

    for (decltype(ep.loop) i = 0; i < ep.loop; i++)
    {
        outputs = network.execute();
        lstm_data.post_process_output(outputs.at("output").get_memory());
        if (ep.use_half)
        {
            lstm_data.fill_memory<half_t>(input);  
        }
        else
        {
            lstm_data.fill_memory<float>(input);
        }
        network.set_input_data("input", input);
		
        if (log_energy)
            energyLib.ReadSample();

    }
    for (size_t i = 0; i < ep.batch; i++)
    {
        std::cout << "BATCH: " << i << " results: " << std::endl;
        std::cout << lstm_data.get_predictions(i, ep.print_type == PrintType::ExtendedTesting ? true : false) << std::endl;
    }

    return get_execution_time(timer_execution, ep, output, outputs, log_energy, energyLib);
}


std::chrono::nanoseconds get_execution_time(cldnn::instrumentation::timer<>& timer_execution,
                                          const execution_params &ep,
                                          cldnn::memory& output,
                                          cldnn_output& outputs,
                                          bool log_energy,
                                          CIntelPowerGadgetLib& energyLib)
{

    //GPU primitives scheduled in unblocked manner
    auto scheduling_time(timer_execution.uptime());

    //OCL buffers mapping blocks until all primitives are completed
    if (ep.topology_name != "microbench_conv" && ep.topology_name != "microbench_lstm")
    {
        std::string output_primitve_id = ep.run_until_primitive_name.empty() ? "output" : ep.run_until_primitive_name;
        output = outputs.at(output_primitve_id).get_memory();
    }

    if (ep.topology_name == "lenet_train")
    {
        for (auto& p : outputs)
        {
            file::serialize_train(p.second.get_memory(), join_path(ep.weights_dir, p.first));
        }
        output = outputs.at("softmax").get_memory();
    }

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

void run_topology(const execution_params &ep)
{
    uint32_t batch_size = ep.batch;

    uint32_t gpu_batch_size = get_gpu_batch_size(batch_size);
    if (gpu_batch_size != batch_size && !ep.rnn_type_of_topology)
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

    if (ep.print_type == Verbose)
    {
        std::cout << "Building " << ep.topology_name << " started" << std::endl;
    }
    else if (ep.print_type == ExtendedTesting)
    {
        std::cout << "Extended testing of " << ep.topology_name << std::endl;
    }
    cldnn::instrumentation::timer<> timer_build;
    cldnn::layout input_layout = { ep.use_half ? cldnn::data_types::f16 : cldnn::data_types::f32, cldnn::format::byxf, {} };
    std::map<cldnn::primitive_id, cldnn::layout> microbench_conv_inputs;
    std::map<cldnn::primitive_id, cldnn::layout> microbench_lstm_inputs;
    microbench_conv_inputs.insert({ "input_layout", input_layout }); //add dummy input so we pass data format to mcirobench_conv topology
    if (ep.topology_name == "alexnet")
        primitives = build_alexnet(ep.weights_dir, engine, input_layout, gpu_batch_size);
    else if (ep.topology_name == "vgg16" || ep.topology_name == "vgg16_face")
        primitives = build_vgg16(ep.weights_dir, engine, input_layout, gpu_batch_size);
    else if (ep.topology_name == "googlenet")
        primitives = build_googlenetv1(ep.weights_dir, engine, input_layout, gpu_batch_size);
    else if (ep.topology_name == "gender")
        primitives = build_gender(ep.weights_dir, engine, input_layout, gpu_batch_size);
    else if (ep.topology_name == "microbench_conv")
        primitives = build_microbench_conv(ep.weights_dir, engine, microbench_conv_inputs, gpu_batch_size);
    else if (ep.topology_name == "resnet50")
        primitives = build_resnet50(ep.weights_dir, engine, input_layout, gpu_batch_size);
    else if (ep.topology_name == "resnet50-i8")
        primitives = build_resnet50_i8(ep.weights_dir, engine, input_layout, gpu_batch_size);
    else if (ep.topology_name == "lstm_char")
        primitives = build_char_level(ep.weights_dir, engine, input_layout, gpu_batch_size, ep.sequence_length);   
	else if (ep.topology_name == "squeezenet")
    {
        if (ep.calibration)
        {
            primitives = build_squeezenet_quant(ep.weights_dir, engine, input_layout, gpu_batch_size);
        }
        else
        {
            primitives = build_squeezenet(ep.weights_dir, engine, input_layout, gpu_batch_size);
        }
    }
    else if (ep.topology_name == "lenet")
        primitives = build_lenet(ep.weights_dir, engine, input_layout, gpu_batch_size);
    else if (ep.topology_name == "lenet_train")
        primitives = build_lenet_train(ep.weights_dir, engine, input_layout, gpu_batch_size, ep.use_existing_weights);
    else if (ep.topology_name == "microbench_lstm")
        primitives = build_microbench_lstm(ep.weights_dir, engine, ep.lstm_ep, microbench_lstm_inputs);
    else if (ep.topology_name == "ssd_mobilenet")
        primitives = build_ssd_mobilenet(ep.weights_dir, engine, input_layout, gpu_batch_size);
    else if (ep.topology_name == "ssd_mobilenet-i8")
        primitives = build_ssd_mobilenet_i8(ep.weights_dir, engine, input_layout, gpu_batch_size);
    else
        throw std::runtime_error("Topology \"" + ep.topology_name + "\" not implemented!");
    microbench_conv_inputs.erase("input_layout");

    auto build_time = timer_build.uptime();

    if (ep.print_type == Verbose)
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
    auto network = build_network(engine, primitives, ep);
    //TODO check if we can define the 'empty' memory
    float zero = 0;
    cldnn::layout zero_layout( cldnn::data_types::f32, cldnn::format::bfyx, {1,1,1,1} );
    auto output = cldnn::memory::attach(zero_layout, &zero, 1);

    if (ep.topology_name != "microbench_conv" && ep.topology_name != "microbench_lstm")
    {
        auto input = cldnn::memory::allocate(engine, input_layout);
        auto neurons_list_filename = "names.txt";
        if (ep.topology_name == "vgg16_face")
            neurons_list_filename = "vgg16_face.txt";
        if (ep.topology_name == "lenet" || ep.topology_name == "lenet_train")
            neurons_list_filename = "lenet.txt";
        else if (ep.topology_name == "gender")
            neurons_list_filename = "gender.txt";

        auto input_list = get_input_list(ep.input_dir);
        if (input_list.empty())
            throw std::runtime_error("specified input images directory is empty (does not contain image data)");

        auto number_of_batches = (input_list.size() % batch_size == 0)
            ? input_list.size() / batch_size : input_list.size() / batch_size + 1;
        std::vector<std::string> input_files_in_batch;
        auto input_list_iterator = input_list.begin();
        auto input_list_end = input_list.end();

        if (ep.topology_name == "lenet" || ep.topology_name == "lenet_train")
            number_of_batches = 1;

        for (decltype(number_of_batches) batch = 0; batch < number_of_batches; batch++)
        {
            input_files_in_batch.clear();
            for (uint32_t i = 0; i < batch_size && input_list_iterator != input_list_end; ++i, ++input_list_iterator)
            {
                input_files_in_batch.push_back(*input_list_iterator);
            }
            double time_in_sec = 0.0;
            lstm_utils lstm_data(ep.sequence_length, batch_size, (unsigned int)ep.loop, ep.temperature);
            // load croped and resized images into input
            if (!ep.rnn_type_of_topology && ep.topology_name != "lenet" && ep.topology_name != "lenet_train")
            {
                if (ep.use_half)
                {
                    load_images_from_file_list<half_t>(input_files_in_batch, input);
                }
                else
                {
                    load_images_from_file_list(input_files_in_batch, input);
                }

            }
            else if (ep.topology_name == "lenet")
            {
                float acc = 0;
                uint32_t labels_num = 10;
                auto labels = cldnn::memory::allocate(engine, { input_layout.data_type, cldnn::format::bfyx,{ input_layout.size.batch[0],1,1,1 } });
                for (uint32_t i = ep.image_offset; i < ep.image_number + ep.image_offset; i += batch_size)
                {
                    if (ep.use_half)
                        load_data_from_file_list_lenet<half_t>(input_list, input, i, batch_size, false, labels);
                    else
                        load_data_from_file_list_lenet(input_list, input, i, batch_size, false, labels);
                    network.set_input_data("input", input);
                    auto outputs = network.execute();
                    auto o = outputs.at("output").get_memory().pointer<float>();

                    auto l = labels.pointer<float>();
                    for (uint32_t b = 0; b < batch_size; b++)
                    {
                        auto e = l[b];
                        std::vector< std::pair<float, uint32_t> > output_vec;
                        //TODO: update this lines to detect output layout and read results based on that
                        if (batch_size == 1) {
                            //check if true label is on top of predictions
                            for (uint32_t j = 0; j < labels_num; j++)
                                output_vec.push_back(std::make_pair(o[j], j));

                            std::sort(output_vec.begin(), output_vec.end(), std::greater<std::pair<float, uint32_t> >());

                            if (output_vec[0].second == e)
                                acc++;
                        }
                        else
                        {
                            //check if true label is on top of predictions
                            for (uint32_t j = 0; j < labels_num; j++)
                                output_vec.push_back(std::make_pair(o[b + j * batch_size], j));

                            std::sort(output_vec.begin(), output_vec.end(), std::greater<std::pair<float, uint32_t> >());

                            if (output_vec[0].second == e)
                                acc++;
                        }
                    }
                }
                std::cout << "Images processed = " << ep.image_number << std::endl;
                std::cout << "Accuracy = " << acc/ep.image_number << std::endl;
                continue;
            }
            else if (ep.topology_name == "lenet_train")
            {
                auto labels = cldnn::memory::allocate(engine, { input_layout.data_type, cldnn::format::bfyx,{ input_layout.size.batch[0],1,1,1 } });
                float base_learning_rate = ep.learning_rate;
                float learning_rate = base_learning_rate;
                for (uint32_t learn_it = ep.image_offset; learn_it < ep.image_number + ep.image_offset; learn_it += batch_size)
                {
                    double loss = 0;
                    //update learning rate, policy "inv", gamma=0.0001, power=0.75.
                    //TODO: enable getting learning rate params from command line
                    learning_rate = base_learning_rate * pow(1.f + 0.0001f * learn_it, -0.75f); 
                    loss = 0;
                    network.set_learning_rate(learning_rate);

                    if (ep.use_half)
                        load_data_from_file_list_lenet<half_t>(input_list, input, learn_it, batch_size, true, labels);
                    else
                        load_data_from_file_list_lenet(input_list, input, learn_it, batch_size, true, labels);

                    network.set_input_data("input", input);
                    network.set_input_data("labels", labels);
                    auto time = execute_cnn_topology(network, ep, energyLib, output);
                    time_in_sec = std::chrono::duration_cast<std::chrono::duration<double, std::chrono::seconds::period>>(time).count();
                    auto expected = labels.pointer<float>();
                    auto vals = output.pointer<float>();

                    for (uint32_t b = 0; b < batch_size; b++)
                    {
                        auto e = expected[b];
                        loss -= log(std::max(vals[b + e * batch_size], std::numeric_limits<float>::min()));
                    }
                    loss = loss / batch_size;
                    std::cout << "Iter: " << learn_it - ep.image_offset << std::endl;
                    std::cout << "Loss = " << loss << std::endl;
                    std::cout << "Learning Rate = " << learning_rate << std::endl;
                }
                continue;
            }
            else
            {
                prepare_data_for_lstm(lstm_data, input_files_in_batch, ep.vocabulary_file);
                if (ep.use_half)
                {
                    lstm_data.fill_memory<half_t>(input);
                }
                else
                {
                    lstm_data.fill_memory<float>(input);
                }
            }
            network.set_input_data("input", input);

            std::chrono::nanoseconds time;
            if (!ep.rnn_type_of_topology)
            {
                time = execute_cnn_topology(network, ep, energyLib, output);
            }
            else
            {
                time = execute_rnn_topology(network, ep, energyLib, output, input, lstm_data);
            }

            time_in_sec = std::chrono::duration_cast<std::chrono::duration<double, std::chrono::seconds::period>>(time).count();

            if (ep.run_until_primitive_name.empty() && ep.run_single_kernel_name.empty() && !ep.rnn_type_of_topology && ep.topology_name != "lenet" && ep.topology_name != "lenet_train")
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
    else if (ep.topology_name == "microbench_conv") {
        auto fill_with = memory_filler::filler_type::zero;
        for (const auto& inp_data : microbench_conv_inputs)
        {
            auto mem = cldnn::memory::allocate(engine, inp_data.second);
            if (!ep.use_half)
            {
                memory_filler::fill_memory<float>(mem, fill_with);
            }
            else
            {
                memory_filler::fill_memory<half_t>(mem, fill_with);
            }

            network.set_input_data(inp_data.first, mem);
        }

        auto time = execute_cnn_topology(network, ep, energyLib, output);
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
    else if (ep.topology_name == "microbench_lstm") {
        auto fill_with = memory_filler::filler_type::zero;
        for (const auto& inp_data : microbench_lstm_inputs)
        {
            auto mem = cldnn::memory::allocate(engine, inp_data.second);
            memory_filler::fill_memory<float>(mem, fill_with);
            network.set_input_data(inp_data.first, mem);
        }
		
        auto time = execute_cnn_topology(network, ep, energyLib, output, nullptr);
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
