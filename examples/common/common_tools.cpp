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

#include "os_windows.h"
#include "FreeImage_wraps.h"
#include "api/neural.h"
#include <string>

// returns list of files (path+filename) from specified directory
std::vector<std::string> get_directory_images(std::string images_path) {
    std::vector<std::string> result;
    if (DIR *folder = opendir(images_path.c_str())) {
        dirent *folder_entry = readdir(folder);
        const auto image_file = std::regex(".*\\.(jpe?g|png|bmp|gif|j2k|jp2|tiff|JPEG|JPG|PNG)$");
        while (folder_entry != nullptr) {
            if (std::regex_match(folder_entry->d_name, image_file) && is_regular_file(folder_entry))
                result.push_back(images_path + "/" + folder_entry->d_name);
            folder_entry = readdir(folder);
        }
        closedir(folder);
    }
    return result;
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
        nn_data_load_from_image(img, it,dim[0], false);
        it += single_image_size;
    }
}