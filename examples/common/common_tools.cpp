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

#include "common_tools.h"

#include "FreeImage_wraps.h"

#include <boost/filesystem.hpp>

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

// i am not sure what is better: pass memory as primitive where layout, ptr and size are included
// or pass as separate parameters to avoid including neural.h in common tools?
void load_images_from_file_list(
    const std::vector<std::string>& images_list,
    neural::primitive& memory)
{
    auto memory_primitive = memory.as<const neural::memory&>().argument;
    auto dst_ptr = memory.as<const neural::memory&>().pointer<float>();
    auto it = dst_ptr.begin();
    // validate if primitvie is memory type
    if (!memory.is<const neural::memory&>()) throw std::runtime_error("Given primitive is not a memory");

    auto batches = std::min(memory_primitive.size.batch[0], (uint32_t) images_list.size()) ;
    auto dim = memory_primitive.size.spatial;

    if (dim[0] != dim[1]) throw std::runtime_error("w and h aren't equal");
    if (memory_primitive.format != neural::memory::format::byxf_f32) throw std::runtime_error("Only bfyx format is supported as input to images from files");
    auto single_image_size = dim[0] * dim[0] * 3;
    for (auto img : images_list)
    {
        // "false" because we want to load images in BGR format because weights are in BGR format and we don't want any conversions between them.
        nn_data_load_from_image(img, it,dim[0], false);
        it += single_image_size;
    }
}
