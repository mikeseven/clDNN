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

#include "api/instrumentation.h"
#include "common_tools.h"
#include "FreeImage_wraps.h"
#include "output_parser.h"

#include <boost/filesystem.hpp>

#include <iostream>

#include <regex>
#include <string>


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
    neural::memory::ptr<float>::iterator dst_buffer,
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
};

static neural::half_t convert_pixel_channel_to_half(uint8_t val)
{
#if defined HALF_HALF_HPP
    return val;
#else
    if (!val)
        return neural::half_t(0x0000U);

    if (val >> 4) // 4..7
    {
        if (val >> 6) // 6..7
        {
            return (val & 0x80)
                ? neural::half_t(0x5800U | ((val & 0x7FU) << 3))
                : neural::half_t(0x5400U | ((val & 0x3FU) << 4));
        }
        else //  4..5
        {
            return (val & 0x20)
                ? neural::half_t(0x5000U | ((val & 0x1FU) << 5))
                : neural::half_t(0x4C00U | ((val & 0x0FU) << 6));
        }
    }
    else // 0..3
    {
        if (val >> 2) // 2..3
        {
            return (val & 0x08)
                ? neural::half_t(0x4800U | ((val & 0x07U) << 7))
                : neural::half_t(0x4400U | ((val & 0x03U) << 8));
        }
        else // 0..1
        {
            return (val & 0x02)
                ? neural::half_t(0x4000U | ((val & 0x01U) << 9))
                : neural::half_t(0x3C00U);
        }
    }
#endif
}

void nn_data_load_from_image(
    std::string  filename,                       // Load of all data from a image filename
    neural::memory::ptr<neural::half_t>::iterator dst_buffer,
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
};

// i am not sure what is better: pass memory as primitive where layout, ptr and size are included
// or pass as separate parameters to avoid including neural.h in common tools?
template <typename MemElemTy>
void load_images_from_file_list(
    const std::vector<std::string>& images_list,
    neural::primitive& memory)
{
    auto memory_primitive = memory.as<const neural::memory&>().argument;
    auto dst_ptr = memory.as<const neural::memory&>().pointer<MemElemTy>();
    auto it = dst_ptr.begin();
    // validate if primitvie is memory type
    if (!memory.is<const neural::memory&>()) throw std::runtime_error("Given primitive is not a memory");

    auto batches = std::min(memory_primitive.size.batch[0], (uint32_t)images_list.size());
    auto dim = memory_primitive.size.spatial;

    if (dim[0] != dim[1]) throw std::runtime_error("w and h aren't equal");
    if (memory_primitive.format != neural::memory::format::byxf_f32 &&
        memory_primitive.format != neural::memory::format::byxf_f16) throw std::runtime_error("Only bfyx format is supported as input to images from files");
    if (neural::memory::traits(memory_primitive.format).type->id != neural::template type_id<MemElemTy>()->id)
        throw std::runtime_error("Memory format expects different type of elements than specified");
    auto single_image_size = dim[0] * dim[0] * 3;
    for (auto img : images_list)
    {
        // "false" because we want to load images in BGR format because weights are in BGR format and we don't want any conversions between them.
        nn_data_load_from_image(img, it, dim[0], false);
        it += single_image_size;
    }
}

template void load_images_from_file_list<float>(const std::vector<std::string>&, neural::primitive&);
template void load_images_from_file_list<neural::half_t>(const std::vector<std::string>&, neural::primitive&);
using namespace neural;

void print_profiling_table(std::ostream& os, const std::vector<instrumentation::profiling_info>& profiling_info) {
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
worker create_worker()
{
    std::cout << "GPU Program compilation started" << std::endl;
    instrumentation::timer<> timer_compilation;
    auto worker = worker_gpu::create();
    auto compile_time = timer_compilation.uptime();
    std::cout << "GPU Program compilation finished in " << instrumentation::to_string(compile_time) << std::endl;
    return worker;
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

std::chrono::nanoseconds execute_topology(const worker& worker,
                                          const std::vector<std::pair<primitive, std::string>>& primitives,
                                          const primitive& output,
                                          bool dump_hl,
                                          const std::string& topology,
                                          size_t primitives_number)
{
    // we need this exact number of primitives(those are created in create_alexnet) 
    assert(primitives.size() == primitives_number);

    std::cout << "Start execution" << std::endl;
    instrumentation::timer<> timer_execution;

    for (auto& p : primitives)
    {
        worker.execute(p.first.work());
    }

    //GPU primitives scheduled in unblocked manner
    auto scheduling_time(timer_execution.uptime());

    //OCL buffers mapping blocks until all primitives are completed
    output.as<const neural::memory&>().pointer<char>();

    auto execution_time(timer_execution.uptime());
    std::cout << topology << " scheduling finished in " << instrumentation::to_string(scheduling_time) << std::endl;
    std::cout << topology << " execution finished in " << instrumentation::to_string(execution_time) << std::endl;
    if (dump_hl)
    {
        instrumentation::logger::log_memory_to_file(primitives[0].first.input[0].primitive(), "input0");
        for (auto& p : primitives)
        {
            instrumentation::logger::log_memory_to_file(p.first, p.second);
        }
        // for now its enough. rest will be done when we have equals those values
    }
    else
    {
        instrumentation::logger::log_memory_to_file(output, "final_result");
    }

    print_profiling_table(std::cout, worker.as<worker_gpu&>().get_profiling_info());

    return std::chrono::duration_cast<std::chrono::nanoseconds>(execution_time);
}

// Optimizing weights
void weight_optimization(weights_optimizer &wo, const worker& worker)
{
    std::cout << "Weights optimization started" << std::endl;
    instrumentation::timer<> timer_execution;
    wo.optimize(worker);
    auto optimizing_time(timer_execution.uptime());
    std::cout << "Weights optimization finished in " << instrumentation::to_string(optimizing_time) << std::endl;
}