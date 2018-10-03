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


/// Global weak pointer to executable information.
///
/// Used to detect misuses:
///  * Using get_executable_info() before set_executable_info().
///  * Using get_executable_info() after destruction of global info object (during global destruction).
static std::weak_ptr<const executable_info> exec_info_ptr;

/// @brief Sets information about executable based on "main"'s command-line arguments.
///
/// @details It works only once (if successful). Next calls to this function will not modify
///          global executable's information object.
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

/// @brief Gets information about executable.
///
/// @details Information is fetched only if information was set using set_executable_info() and not yet
///          destroyed (during global destruction). Otherwise, exception is thrown.
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

std::string join_path(const std::string& parent, const std::string& child)
{
    return (path(parent) / child).string();
}


/// @brief Returns list of all files from specified directory (recursive) which extensions are matching
///        specified extension pattern.
///
/// @param root_dir  Root directory to scan.
/// @param extension Extension pattern to match.
/// @return          List of file paths (paths are normalized and returned as absolute paths).
static std::vector<std::string> list_dir_files(const std::string& root_dir, const std::regex& extension)
{
    std::vector<std::string> result;
    for (const auto& dir_entry : recursive_directory_iterator(root_dir))
    {
        if (dir_entry.status().type() == file_type::regular_file &&
            std::regex_match(dir_entry.path().extension().string(), extension))
        {
            result.push_back(absolute(dir_entry.path()).string());
        }
    }

    return result;
}

std::vector<std::string> list_input_files(const std::string& root_dir, const bool image_files_only)
{
    const auto allowed_exts_pattern = image_files_only
                                          ? "^\\.(?:jpe?g|png|bmp|gif|j2k|jp2|tiff)$"
                                          : "^\\.(?:jpe?g|png|bmp|gif|j2k|jp2|tiff|txt|idx3\\-ubyte|bin)$";
    const std::regex allowed_exts(
        allowed_exts_pattern,
        std::regex_constants::ECMAScript | std::regex_constants::icase | std::regex_constants::optimize);
    return list_dir_files(root_dir, allowed_exts);
}

// returns list of files (path+filename) from specified directory
std::vector<std::string> get_directory_weights(const std::string& images_path)
{
    const std::regex allowed_exts(
        "^\\.nnd$",
        std::regex_constants::ECMAScript | std::regex_constants::icase | std::regex_constants::optimize);
    return list_dir_files(images_path, allowed_exts);
}

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


/// @brief Converts pixel channel value (8-bit) to corresponding element type.
///
/// @param val Input value.
/// @return    Element representation with the same numerical value.
template <typename ElemTy = float>
static ElemTy convert_pixel_channel(const std::uint8_t val)
{
    return val;
}

/// @brief Converts pixel channel value (8-bit) to corresponding half-precision floating
///        point value (no normalization).
///
/// @param val Input value.
/// @return    Half representation with the same numerical value.
template <>
half_t convert_pixel_channel<half_t>(const std::uint8_t val)
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


/// @brief Loads data from image file into clDNN allocation / memory buffer (store layout YXFB; FP16).
///
/// @details The loaded image is normalized before reading actual data:
///           -# Image is optionally resized (aspect ratio is kept) when the @p min_size is not @c 0.
///           -# Image is optionally normalized to square image (aspect ratio is preserved).
///              The parts of images outside square (square side is set to minimum of image width and height)
///              are cut out.
///           -# If image is not 24-bit RGB, it is converted to 24-bit RGB.
///           .
///          If any of these steps failed, the image is not read and the @c false is returned.
///
/// @param image_file_path  Path (relative or absolute) to image file.
/// @param output_buffer_it Iterator (output iterator, copy-constructible) that points to
///                         buffer where data from image should be stored.
/// @param rgb_order        Indicates that color channels should be stored in features in RGB order.
///                         If @c true then RGB order is used; otherwise, BGR order is applied.
/// @param normalized_size  Size of image square to which image will be normalized.
///                         If @c 0 is specified, the image is not normalized to square.
/// @param min_size         Size of either width or height (which is lower) to which image
///                         should be resized (keeping aspect ratio) before normalization.
///                         If @c 0 is specified, the image is not resized.
/// @return                 @c true if image load was successful; otherwise, @c false.
template <typename OutIterTy>
static auto load_data_from_image(const std::string& image_file_path,
                                 OutIterTy output_buffer_it,
                                 const bool rgb_order           = false,
                                 const unsigned normalized_size = 0,
                                 const unsigned min_size        = 0)
    -> std::enable_if_t<std::is_arithmetic<typename std::iterator_traits<OutIterTy>::value_type>::value ||
                        std::is_same<typename std::iterator_traits<OutIterTy>::value_type, half_t>::value, bool>
{
    using out_elem_type = typename std::iterator_traits<OutIterTy>::value_type;


    std::shared_ptr<fipImage> normalized_image;
    try
    {
        const auto loaded_image = min_size != 0
                                      ? itk::resize_keep_ar(itk::load(image_file_path), min_size)
                                      : itk::load(image_file_path);
        normalized_image = normalized_size != 0
                               ? itk::crop_resize_to_square(loaded_image, normalized_size)
                               : loaded_image;
    }
    catch (const std::runtime_error&)
    {
        return false;
    }

    if (!normalized_image->isValid() || normalized_image->accessPixels() == nullptr)
        return false;
    if (normalized_image->getBitsPerPixel() != 24)
    {
        const auto convert_success = normalized_image->convertTo24Bits();
        if (!convert_success)
            return false;
    }

    const auto width = normalized_image->getWidth();
    const auto height = normalized_image->getHeight();

    const auto bytes_per_pixel = normalized_image->getLine() / width;
    auto buffer_it = output_buffer_it;

    if (rgb_order)
    {
        for (unsigned y = 0; y < height; ++y)
        {
            auto image_pixel = reinterpret_cast<std::uint8_t*>(normalized_image->getScanLine(height - y - 1));
            if (image_pixel == nullptr)
                return false;

            for (unsigned x = 0; x < width; ++x)
            {
                *buffer_it = convert_pixel_channel<out_elem_type>(image_pixel[FI_RGBA_RED]);   ++buffer_it;
                *buffer_it = convert_pixel_channel<out_elem_type>(image_pixel[FI_RGBA_GREEN]); ++buffer_it;
                *buffer_it = convert_pixel_channel<out_elem_type>(image_pixel[FI_RGBA_BLUE]);  ++buffer_it;
                image_pixel += bytes_per_pixel;
            }
        }
    }
    else
    {
        for (unsigned y = 0; y < height; ++y)
        {
            auto image_pixel = reinterpret_cast<std::uint8_t*>(normalized_image->getScanLine(height - y - 1));
            if (image_pixel == nullptr)
                return false;

            for (unsigned x = 0; x < width; ++x)
            {
                *buffer_it = convert_pixel_channel<out_elem_type>(image_pixel[FI_RGBA_BLUE]);  ++buffer_it;
                *buffer_it = convert_pixel_channel<out_elem_type>(image_pixel[FI_RGBA_GREEN]); ++buffer_it;
                *buffer_it = convert_pixel_channel<out_elem_type>(image_pixel[FI_RGBA_RED]);   ++buffer_it;
                image_pixel += bytes_per_pixel;
            }
        }
    }

    return true;
}

std::string get_image_file(const std::string& img_name, const std::vector<std::string>& images_list)
{
    std::string img = "";
    for (auto img_from_list : images_list)
    {
        if (img_from_list.find(img_name) != std::string::npos)
        {
            img = img_from_list;
            break;
        }
    }
    if (img == "")
        throw std::runtime_error("Image file was not found.");

    return img;
}

// i am not sure what is better: pass memory as primitive where layout, ptr and size are included
// or pass as separate parameters to avoid including neural.h in common tools?
template <typename MemElemTy>
void load_images_from_file_list(
    const std::vector<std::string>& images_list,
    cldnn::memory& memory,
    const uint32_t min_size)
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
        load_data_from_image(img, it, false, dim[0], min_size);
        it += single_image_size;
    }
}

uint32_t swap_endian(uint32_t val) {
    val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
    return (val << 16) | (val >> 16);
}

// Explicit instantiation of all used template function instances used in examples.
template void load_images_from_file_list<float>(const std::vector<std::string>&, cldnn::memory&, const uint32_t);
template void load_images_from_file_list<half_t>(const std::vector<std::string>&, cldnn::memory&, const uint32_t);

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

    std::string img = get_image_file(img_name, images_list);
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

template <typename MemElemTy>
void load_data_from_file_list_imagenet(
    const std::vector<std::string>& images_list, const std::string& input_dir,
    cldnn::memory& memory, const uint32_t images_offset, const uint32_t images_number, const bool, cldnn::memory& memory_labels)
{
    auto dst_ptr = memory.pointer<MemElemTy>();

    auto memory_layout = memory.get_layout();
    if (!cldnn::data_type_match<MemElemTy>(memory_layout.data_type))
        throw std::runtime_error("Memory format expects different type of elements than specified");

    if ((images_offset + images_number) > images_list.size())
        throw std::runtime_error("images_offset + images_number is bigger than number of images in imagenet directory");

    auto class_num = 0;
    auto images_per_class = 0;

    for (directory_iterator it(input_dir); it != directory_iterator(); ++it)
        class_num++;

    for (recursive_directory_iterator it(input_dir); it != recursive_directory_iterator(); ++it)
        images_per_class++;
    images_per_class = (images_per_class - class_num) / class_num;

    std::vector<std::string> requested_images;
    for (uint32_t i = images_offset; i < images_offset + images_number; i++)
    {
        auto img_idx = i * images_per_class % (class_num * images_per_class) + i * images_per_class / (class_num * images_per_class);
        requested_images.push_back(images_list[img_idx]);
    }

    //read in images
    load_images_from_file_list<MemElemTy>(requested_images, memory, 256);

    //read in labels
    auto labels_ptr = memory_labels.pointer<MemElemTy>();

    auto labels_file = join_path(get_executable_info()->dir(), "synset_words.txt");
    std::ifstream rfile_labels(labels_file, std::ios::binary);

    if (rfile_labels)
    {
        std::string line;
        std::vector<std::string> line_mappings;
        while (std::getline(rfile_labels, line))
            line_mappings.push_back(line.substr(0, 9));

        std::vector<std::uint32_t> requested_labels;
        for (uint32_t j = 0; j < requested_images.size(); j++)
        {
            auto img_label = requested_images[j].substr(requested_images[j].find_last_of("/\\") - 9, 9);
            auto pos = std::find(line_mappings.begin(), line_mappings.end(), img_label);

            if (pos != line_mappings.end())
            {
                auto vec_idx = (uint32_t)(pos - line_mappings.begin());
                labels_ptr[j] = (MemElemTy)vec_idx;
            }
            else
                throw std::runtime_error("Cannot find requested label in synset_words.txt file.");
        }
    }
    else
        throw std::runtime_error("Cannot read labels file for imagenet.");
}

template void load_data_from_file_list_imagenet<float>(const std::vector<std::string>&, const std::string&, cldnn::memory&, const uint32_t, const uint32_t, const bool, cldnn::memory&);
template void load_data_from_file_list_imagenet<half_t>(const std::vector<std::string>&, const std::string&, cldnn::memory&, const uint32_t, const uint32_t, const bool, cldnn::memory&);

template <typename MemElemTy>
void load_data_from_file_list_cifar10(
    const std::vector<std::string>& images_list, const std::string& input_dir,
    cldnn::memory& memory, const uint32_t images_offset, const uint32_t images_number, const bool train, cldnn::memory& memory_labels)
{
    auto dst_ptr = memory.pointer<MemElemTy>();
    auto labels_ptr = memory_labels.pointer<MemElemTy>();
    auto labels_it = labels_ptr.begin();

    auto memory_layout = memory.get_layout();
    int count = 0;
    if (!cldnn::data_type_match<MemElemTy>(memory_layout.data_type))
        throw std::runtime_error("Memory format expects different type of elements than specified");

    //The images file from cifar10 are hardcoded to:
    // - training: data_batch.bin
    // - testing: test_batch.bin
    std::string img_name = "";
    if (!train)
        img_name = "test_batch.bin";
    else
        img_name = "data_batch.bin";

    std::ifstream rfile(get_image_file(img_name, images_list), std::ios::binary);

    const uint32_t img_spatial = 32;
    const uint32_t img_size = 1 + img_spatial * img_spatial * 3; //1-byte for label, 32*32*3 bytes for image data;

    if (rfile)
    {
        std::vector<unsigned char> tmpBuffer(img_size * images_number);

        rfile.seekg(images_offset * img_size, rfile.cur);
        rfile.read(reinterpret_cast<char *>(&tmpBuffer[0]), img_size * images_number);
        rfile.close();

        //read in image data
        for (uint32_t j = 0; j < images_number; ++j)
        {
            for (uint32_t y = 0u; y < img_spatial; ++y)
            {
                for (uint32_t x = 0u; x < img_spatial; ++x)
                {
                    dst_ptr[j * (img_size - 1) + y * 3 * img_spatial + x * 3 + 2] = static_cast<MemElemTy>(tmpBuffer[j * img_size + 1 + y * img_spatial + x + 0]);
                    dst_ptr[j * (img_size - 1) + y * 3 * img_spatial + x * 3 + 1] = static_cast<MemElemTy>(tmpBuffer[j * img_size + 1 + y * img_spatial + x + 1024]);
                    dst_ptr[j * (img_size - 1) + y * 3 * img_spatial + x * 3 + 0] = static_cast<MemElemTy>(tmpBuffer[j * img_size + 1 + y * img_spatial + x + 2048]);
                }
            }
        }

        //read in labels
        for (uint32_t i = 0; i < images_number; i++) {
            *labels_it = static_cast<MemElemTy>(tmpBuffer[i * img_size]);
            labels_it++;
        }
    }
    else
        throw std::runtime_error("Cannot read image cifar10 image file.");
}

template void load_data_from_file_list_cifar10<float>(const std::vector<std::string>&, const std::string&, cldnn::memory&, const uint32_t, const uint32_t, const bool, cldnn::memory&);
template void load_data_from_file_list_cifar10<half_t>(const std::vector<std::string>&, const std::string&, cldnn::memory&, const uint32_t, const uint32_t, const bool, cldnn::memory&);


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
            load_data_from_image(img, img_tmp_it, false, size_x, 256);

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

template<>
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

std::chrono::nanoseconds execute_rnn_topology(cldnn::network network,
    const execution_params &ep,
    CIntelPowerGadgetLib& energyLib,
    cldnn::memory& output,
    cldnn::memory& input,
    lstm_utils& lstm_data)
{
    bool log_energy = do_log_energy(ep, energyLib);

    if (ep.print_type == print_type::verbose)
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
        std::cout << lstm_data.get_predictions(i, ep.print_type == print_type::extended_testing ? true : false) << std::endl;
    }

    return get_execution_time(timer_execution, ep, output, outputs, log_energy, energyLib);
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
