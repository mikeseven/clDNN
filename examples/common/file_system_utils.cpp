// Copyright (c) 2016-2018 Intel Corporation
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

#include "file_system_utils.h"

#include "image_toolkit.h"

#include <boost/filesystem.hpp>

#include <limits>
#include <regex>
#include <unordered_map>

using namespace boost::filesystem;


namespace cldnn
{
namespace utils
{
namespace examples
{

/// @brief Returns list of all files from specified directory (recursive) which extensions are matching
///        specified extension pattern.
///
/// @param root_dir     Root directory to scan.
/// @param extension_re Extension regular expression to match (filtering expression).
/// @return             List of file paths (paths are normalized and returned as absolute paths).
static std::vector<std::string> list_dir_files(const std::string& root_dir, const std::regex& extension_re)
{
    std::vector<std::string> result;
    for (const auto& dir_entry : recursive_directory_iterator(root_dir))
    {
        if (dir_entry.status().type() == file_type::regular_file &&
            std::regex_match(dir_entry.path().extension().string(), extension_re))
        {
            result.push_back(absolute(dir_entry.path()).string());
        }
    }

    return result;
}


std::string join_path(const std::string& parent, const std::string& child)
{
    return (path(parent) / child).string();
}

std::vector<std::string> rebase_file_paths(const std::vector<std::string>& file_paths, const std::string& root_dir,
                                           const std::string& new_root_dir)
{
    std::vector<std::string> rebased_file_paths;
    for (const auto& file_path : file_paths)
    {
        auto rel_file_path = relative(file_path, root_dir);
        if (new_root_dir.empty())
            rebased_file_paths.emplace_back(rel_file_path.string());
        else
            rebased_file_paths.emplace_back((new_root_dir / rel_file_path).string());
    }
    return rebased_file_paths;
}

std::vector<std::string> list_input_files(const std::string& root_dir, const input_file_type file_type)
{
    if (file_type == input_file_type::none)
        return {};

    // Allowed patterns of extensions (excluding initial dot and anchors).
    static const std::unordered_map<input_file_type, std::string> allowed_exts_patterns {
        {input_file_type::image,                                                    "jpe?g|jif|jpe|jxr|[hw]dp|png|bmp|"
                                                                                    "gif|j2[ck]|jp2|tiff?|exr|hdr|ico|"
                                                                                    "jng|tga|targa|wbmp|webp|xpm|ppm"},
        {input_file_type::text,                                                     "te?xt"},
        {input_file_type::image_set_mnist_db,                                       "idx3-ubyte"},
        {input_file_type::image_set_lmdb,                                           "mdb"},
        {input_file_type::image_set_cifar10 | input_file_type::image_set_cifar100,  "bin"},
        {input_file_type::label_set_mnist_db,                                       "idx1-ubyte"},
    };

    // Construct extensions regular expression pattern for input files.
    std::string selected_exts_pattern = "^\\.(?:";
    selected_exts_pattern.reserve(256);
    auto ext_sep = "";
    for (const auto& allowed_exts_pattern : allowed_exts_patterns)
    {
        if ((file_type & allowed_exts_pattern.first) != input_file_type::none)
        {
            selected_exts_pattern += ext_sep;
            ext_sep = "|";
            selected_exts_pattern += allowed_exts_pattern.second;
        }
    }
    selected_exts_pattern += ")$";

    // List files.
    const std::regex allowed_exts_re(
        selected_exts_pattern,
        std::regex_constants::ECMAScript | std::regex_constants::icase | std::regex_constants::optimize);
    return list_dir_files(root_dir, allowed_exts_re);
}

std::vector<std::string> list_weight_files(const std::string& root_dir)
{
    const std::regex allowed_exts_re(
        "^\\.nnd$",
        std::regex_constants::ECMAScript | std::regex_constants::icase | std::regex_constants::optimize);
    return list_dir_files(root_dir, allowed_exts_re);
}

auto get_image_dims(const std::vector<std::string>::const_iterator image_paths_first,
                    const std::vector<std::string>::const_iterator image_paths_last,
                    const boost::optional<std::size_t>& limit)
    -> std::vector<std::tuple<std::string, unsigned, unsigned>>
{
    const auto image_limit = limit.value_or(std::numeric_limits<std::size_t>::max());

    std::vector<std::tuple<std::string, unsigned, unsigned>> image_dims;

    auto image_paths_it = image_paths_first;
    std::size_t image_idx = 0;
    while (image_paths_it != image_paths_last && (!limit.is_initialized() || image_idx < image_limit))
    {
        const auto& image_path = *image_paths_it++;

        if (const auto image = itk::load(image_path))
        {
            if (image->getWidth() > 0 && image->getHeight() > 0)
            {
                image_dims.emplace_back(image_path, image->getWidth(), image->getHeight());
                ++image_idx;
            }
        }
    }
    return image_dims; // NRVO
}


template <typename MemElemTy>
auto load_image_files(const std::vector<std::string>::const_iterator image_paths_first,
                      const std::vector<std::string>::const_iterator image_paths_last,
                      memory& images_memory, const unsigned min_size)
    -> std::pair<std::vector<std::string>::const_iterator, std::vector<std::string>>
{
    const auto& imgs_mem_layout = images_memory.get_layout();
    auto buffer = images_memory.pointer<MemElemTy>();
    auto buffer_it = buffer.begin();

    if(imgs_mem_layout.format != cldnn::format::byxf)
        throw std::runtime_error("Only memory object with BYXF layout are supported as placeholder for images data");
    if(!cldnn::data_type_match<MemElemTy>(imgs_mem_layout.data_type))
        throw std::logic_error("Declared data element used by memory object is different than "
                               "type used to access its buffer");

    const auto xy_sizes    = imgs_mem_layout.size.spatial;
    const auto batch_num   = static_cast<unsigned>(imgs_mem_layout.size.batch[0]);
    const auto feature_num = imgs_mem_layout.size.feature[0];
    assert(xy_sizes.size() == 2 && "Unexpected number of spatial dimensions was used (not 2).");
    assert(feature_num == 3 && "Unexpected number of features / channels was used (not 3).");
    static_cast<const void>(feature_num); // Used only in debug code (simulate usage).
    if (xy_sizes[0] <= 0 || xy_sizes[1] <= 0)
        throw std::runtime_error("Spatial sizes specified in layout of memory object must be positive");

    auto expected_image_size = xy_sizes[0] * xy_sizes[1] * 3; // 24 bits per pixel.
    std::vector<std::string> loaded_file_paths;
    loaded_file_paths.reserve(batch_num + 1);

    auto image_paths_it = image_paths_first;
    auto batch_idx = 0U;
    while (image_paths_it != image_paths_last && batch_idx < batch_num)
    {
        const auto& image_path = *image_paths_it++;

        if (itk::load_image_data(image_path, buffer_it, false,
                                 static_cast<unsigned>(xy_sizes[0]), static_cast<unsigned>(xy_sizes[1]),
                                 min_size))
        {
            loaded_file_paths.emplace_back(image_path);
            buffer_it += expected_image_size;
            ++batch_idx;
        }
    }
    return {image_paths_it, loaded_file_paths}; //NRVO
}

// Explicit instantiation of all used template function instances used in examples.
template auto load_image_files<float>(std::vector<std::string>::const_iterator,
                                      std::vector<std::string>::const_iterator,
                                      memory&, unsigned)
    -> std::pair<std::vector<std::string>::const_iterator, std::vector<std::string>>;
template auto load_image_files<half_t>(std::vector<std::string>::const_iterator,
                                       std::vector<std::string>::const_iterator,
                                       memory&, unsigned)
    -> std::pair<std::vector<std::string>::const_iterator, std::vector<std::string>>;


template <typename MemElemTy>
auto save_image_files(const std::vector<std::string>::const_iterator image_paths_first,
                      const std::vector<std::string>::const_iterator image_paths_last,
                      const memory& images_memory, const unsigned min_size)
    -> std::pair<std::vector<std::string>::const_iterator, std::vector<std::shared_ptr<fipImage>>>
{
    const auto& imgs_mem_layout = images_memory.get_layout();
    auto buffer = images_memory.pointer<MemElemTy>();
    auto buffer_it = buffer.begin();

    if(imgs_mem_layout.format != cldnn::format::byxf)
        throw std::runtime_error("Only memory object with BYXF layout are supported as source for images data");
    if(!cldnn::data_type_match<MemElemTy>(imgs_mem_layout.data_type))
        throw std::logic_error("Declared data element used by memory object is different than "
                               "type used to access its buffer");

    const auto xy_sizes  = imgs_mem_layout.size.spatial;
    const auto batch_num = static_cast<unsigned>(imgs_mem_layout.size.batch[0]);
    const auto feature_num = imgs_mem_layout.size.feature[0];
    assert(xy_sizes.size() == 2 && "Unexpected number of spatial dimensions was used (not 2).");
    assert(feature_num == 3 && "Unexpected number of features / channels was used (not 3).");
    static_cast<const void>(feature_num); // Used only in debug code (simulate usage).
    if (xy_sizes[0] <= 0 || xy_sizes[1] <= 0)
        throw std::runtime_error("Spatial sizes specified in layout of memory object must be positive");

    auto expected_image_size = xy_sizes[0] * xy_sizes[1] * 3; // 24 bits per pixel.
    std::vector<std::shared_ptr<fipImage>> saved_images;
    saved_images.reserve(batch_num + 1);

    auto image_paths_it = image_paths_first;
    auto batch_idx = 0U;
    while (image_paths_it != image_paths_last && batch_idx < batch_num)
    {
        const auto& image_path = *image_paths_it++;

        saved_images.emplace_back(
            itk::save_image_data(image_path, buffer_it, false,
                                 static_cast<unsigned>(xy_sizes[0]), static_cast<unsigned>(xy_sizes[1]),
                                 min_size));
        buffer_it += expected_image_size;
        ++batch_idx;
    }
    return {image_paths_it, saved_images}; // NRVO
}

// Explicit instantiation of all used template function instances used in examples.
template auto save_image_files<float>(std::vector<std::string>::const_iterator,
                                      std::vector<std::string>::const_iterator,
                                      const memory&, unsigned)
    -> std::pair<std::vector<std::string>::const_iterator, std::vector<std::shared_ptr<fipImage>>>;
template auto save_image_files<half_t>(std::vector<std::string>::const_iterator,
                                       std::vector<std::string>::const_iterator,
                                       const memory&, unsigned)
    -> std::pair<std::vector<std::string>::const_iterator, std::vector<std::shared_ptr<fipImage>>>;

} // namespace examples
} // namespace utils
} // namespace cldnn

// --------------------------------------------------------------------------------------------------------------------
// TODO: Refactor rest of load functions.
// --------------------------------------------------------------------------------------------------------------------

#include "executable_utils.h"

using namespace cldnn::utils::examples;


std::string get_image_file(const std::string& img_name, const std::vector<std::string>& images_list)
{
    std::string img;
    for (const auto& img_from_list : images_list)
    {
        if (img_from_list.find(img_name) != std::string::npos)
        {
            img = img_from_list;
            break;
        }
    }
    if (img.empty())
        throw std::runtime_error("Image file was not found.");

    return img;
}

static uint32_t swap_endian(uint32_t val) {
    val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
    return (val << 16) | (val >> 16);
}

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
    std::string img_name;
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

        rfile.seekg(images_offset * img_size, std::ios::cur);
        rfile.read(reinterpret_cast<char *>(&tmpBuffer[0]), img_size * images_number);
        rfile.close();

        for (uint32_t i = 0; i < img_size * images_number; ++i) {
            *it = static_cast<MemElemTy>(tmpBuffer[i]);
            ++it;
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
                ++labels_it;
            }
        }
        else
            throw std::runtime_error("Cannot read labels for lenet topology.");
        count++;
    }
    else
        throw std::runtime_error("Cannot read image for lenet topology.");
}

template void load_data_from_file_list_lenet<float>(const std::vector<std::string>&, cldnn::memory&, uint32_t, uint32_t, bool, cldnn::memory&);
template void load_data_from_file_list_lenet<half_t>(const std::vector<std::string>&, cldnn::memory&, uint32_t, uint32_t, bool, cldnn::memory&);

template <typename MemElemTy>
void load_data_from_file_list_imagenet(
    const std::vector<std::string>& images_list, const std::string& input_dir,
    cldnn::memory& memory, const uint32_t images_offset, const uint32_t images_number, const bool, cldnn::memory& memory_labels)
{
    auto dst_ptr = memory.pointer<MemElemTy>();

    const auto memory_layout = memory.get_layout();
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
    load_image_files<MemElemTy>(requested_images, memory, 256);

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

template void load_data_from_file_list_imagenet<float>(const std::vector<std::string>&, const std::string&, cldnn::memory&, uint32_t, uint32_t, bool, cldnn::memory&);
template void load_data_from_file_list_imagenet<half_t>(const std::vector<std::string>&, const std::string&, cldnn::memory&, uint32_t, uint32_t, bool, cldnn::memory&);

template <typename MemElemTy>
void load_data_from_file_list_cifar10(
    const std::vector<std::string>& images_list, const std::string&,
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
    std::string img_name;
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
           ++labels_it;
        }
    }
    else
        throw std::runtime_error("Cannot read image cifar10 image file.");
}

template void load_data_from_file_list_cifar10<float>(const std::vector<std::string>&, const std::string&, cldnn::memory&, uint32_t, uint32_t, bool, cldnn::memory&);
template void load_data_from_file_list_cifar10<half_t>(const std::vector<std::string>&, const std::string&, cldnn::memory&, uint32_t, uint32_t, bool, cldnn::memory&);
