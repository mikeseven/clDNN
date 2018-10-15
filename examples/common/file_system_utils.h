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

#pragma once

#include <api/CPP/memory.hpp>

#include <boost/optional.hpp>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>


// --------------------------------------------------------------------------------------------------------------------
// File system and path manipulation utilities.
// --------------------------------------------------------------------------------------------------------------------

// ReSharper disable once CppInconsistentNaming
class fipImage;


namespace cldnn
{
namespace utils
{
namespace examples
{

/// @brief  Type of input file / input item (flags enum).
///
/// @details Enum is flags enum - the values can be combined using bitwise operations.
enum class input_file_type
{
    image = 0x0001, ///< Input is an image (in standard image format).
    text  = 0x0002, ///< Input is a plain text file.

    // Image sets / archives.
    image_set_mnist_db = 0x0100, ///< Input is an image set organized in MNIST binary data set (idx3-ubyte format).
    image_set_lmdb     = 0x0200, ///< Input is an image set organized into LMDB (Lightning Memory-Mapped Database)
                                 ///< key/value binary storage (ImageNet/Caffe-compatible).
    image_set_cifar10  = 0x0400, ///< Input is an image set organized into CIFAR-10 binary data set
                                 ///< (label & image archive).
    image_set_cifar100 = 0x0800, ///< Input is an image set organized into CIFAR-10 binary data set
                                 ///< (label & image archive).

    label_set_mnist_db = 0x010000, ///< Input is a label set organized in MNIST binary data set (idx1-ubyte format).

    // Combinations.
    none              = 0,                                       ///< No input allowed (used for filtering).
    data_set_mnist_db = image_set_mnist_db | label_set_mnist_db, ///< Input is a MNIST binary data set.
    any               = ~none,                                   ///< Any recognizable input is allowed (used for
                                                                 ///< filtering).
};

constexpr input_file_type operator |(input_file_type lhs, input_file_type rhs)
{
    using int_type = std::underlying_type_t<input_file_type>;
    return static_cast<input_file_type>(static_cast<int_type>(lhs) | static_cast<int_type>(rhs));
}

constexpr input_file_type operator &(input_file_type lhs, input_file_type rhs)
{
    using int_type = std::underlying_type_t<input_file_type>;
    return static_cast<input_file_type>(static_cast<int_type>(lhs) & static_cast<int_type>(rhs));
}

constexpr input_file_type operator ^(input_file_type lhs, input_file_type rhs)
{
    using int_type = std::underlying_type_t<input_file_type>;
    return static_cast<input_file_type>(static_cast<int_type>(lhs) ^ static_cast<int_type>(rhs));
}

constexpr input_file_type operator ~(input_file_type rhs)
{
    using int_type = std::underlying_type_t<input_file_type>;
    return static_cast<input_file_type>(~static_cast<int_type>(rhs));
}

// --------------------------------------------------------------------------------------------------------------------

/// Joins path using native path/directory separator.
///
/// @param parent Parent path.
/// @param child  Child part of path.
///
/// @return Joined path.
std::string join_path(const std::string& parent, const std::string& child);

/// @brief Re-roots specified file paths from old root directory to new one.
///
/// @details Function does only pure path operations. It does not check for existence of files / directories, etc.
///
/// @param file_paths   File paths to rebase.
/// @param root_dir     Old root directory path (base directory path).
/// @param new_root_dir New root directory path.
/// @n                  If not specified (empty), the function will return relative paths
///                     to old base.
/// @return             List (the same order) of rebased file paths.
std::vector<std::string> rebase_file_paths(const std::vector<std::string>& file_paths, const std::string& root_dir,
                                           const std::string& new_root_dir = "");

/// @brief Returns list of all input files from specified directory (recursive).
///
/// @param root_dir         Root directory to scan for input files.
/// @param file_type        Type (or combination of types) of input files to be returned. See input_file_type
///                         enumeration for details.
/// @return                 List of file paths (paths are normalized and returned as absolute paths).
std::vector<std::string> list_input_files(const std::string& root_dir,
                                          input_file_type file_type = input_file_type::any);

/// @brief Returns list of all weight files from specified directory (recursive).
///
/// @param root_dir Root directory to scan for weight files.
/// @return         List of file paths (paths are normalized and returned as absolute paths).
std::vector<std::string> list_weight_files(const std::string& root_dir);

/// @brief Gets dimensions of group of images.
///
/// @param image_paths_first       Range of paths (relative or absolute) to image files (inclusive start).
/// @param image_paths_last        Range of paths (relative or absolute) to image files (exclusive end).
/// @param limit                   Optional limit for number of output results.
/// @return                        List of file names which dimensions have been read
///                                successfully.
/// @n                             Each element is a tuple with file path, image width and image height.
auto get_image_dims(std::vector<std::string>::const_iterator image_paths_first,
                    std::vector<std::string>::const_iterator image_paths_last,
                    const boost::optional<std::size_t>& limit = boost::none)
    -> std::vector<std::tuple<std::string, unsigned, unsigned>>;

/// @brief Gets dimensions of group of images.
///
/// @param image_paths             List of paths (relative or absolute) to image files.
/// @param limit                   Optional limit for number of output results.
/// @return                        List of file names which dimensions have been read
///                                successfully.
/// @n                             Each element is a tuple with file path, image width and image height.
inline auto get_image_dims(const std::vector<std::string>& image_paths,
                           const boost::optional<std::size_t>& limit = boost::none)
    -> std::vector<std::tuple<std::string, unsigned, unsigned>>
{
    return get_image_dims(image_paths.begin(), image_paths.end(), limit);
}

/// @brief Loads images from specified file paths into memory object.
///
/// @details Loads images from files pointed by paths from [@p image_paths_first; @p image_paths_last) range.
/// @n
/// @n       Each loaded image is optionally resized (aspect ratio is preserved, so the width or height,
///          whichever is lower, is set to @p min_size) and next normalized (crop-resized) to dimensions compatible
///          with layout sizes of @p images_memory object.
/// @n
/// @n       The function tries to load images from the range until batch size is reached (also stored in layout)
///          or all image files are loaded.
///
/// @tparam MemElemTy Type of memory object elements to which convert pixels from image.
///
/// @param image_paths_first       Range of paths (relative or absolute) to image files (inclusive start).
/// @param image_paths_last        Range of paths (relative or absolute) to image files (exclusive end).
/// @param [in, out] images_memory Memory object containing metadata and buffers to store image.
/// @param min_size                Size of either width or height (which is lower) to which image
///                                should be resized (keeping aspect ratio) before normalization.
///                                If @c 0 is specified, the image is not resized.
/// @return                        Paths of successfully loaded image files.
///
/// @exception std::logic_error   Specified memory element type is different than declared in layout
///                               of memory object.
/// @exception std::runtime_error Passed memory object has unsupported layout format (non-BYXF).
/// @exception std::runtime_error One of spatial sizes defined in layout of memory object is non-positive.
template <typename MemElemTy = float>
auto load_image_files(std::vector<std::string>::const_iterator image_paths_first,
                      std::vector<std::string>::const_iterator image_paths_last,
                      memory& images_memory, unsigned min_size = 0)
    -> std::pair<std::vector<std::string>::const_iterator, std::vector<std::string>>;

/// @brief Loads images from specified file paths into memory object.
///
/// @details Loads images from files pointed by paths from @p image_paths.
/// @n
/// @n       Each loaded image is optionally resized (aspect ratio is preserved, so the width or height,
///          whichever is lower, is set to @p min_size) and next normalized (crop-resized) to dimensions compatible
///          with layout sizes of @p images_memory object.
/// @n
/// @n       The function tries to load images from the list until batch size is reached (also stored in layout)
///          or all image files are loaded.
///
/// @tparam MemElemTy Type of memory object elements to which convert pixels from image.
///
/// @param image_paths             List of paths (relative or absolute) to image files.
/// @param [in, out] images_memory Memory object containing metadata and buffers to store image.
/// @param min_size                Size of either width or height (which is lower) to which image
///                                should be resized (keeping aspect ratio) before normalization.
///                                If @c 0 is specified, the image is not resized.
/// @return                        Paths of successfully loaded image files.
///
/// @exception std::logic_error   Specified memory element type is different than declared in layout
///                               of memory object.
/// @exception std::runtime_error Passed memory object has unsupported layout format (non-BYXF).
/// @exception std::runtime_error One of spatial sizes defined in layout of memory object is non-positive.
template <typename MemElemTy = float>
auto load_image_files(const std::vector<std::string>& image_paths, memory& images_memory, const unsigned min_size = 0)
    -> std::vector<std::string>
{
    return load_image_files<MemElemTy>(image_paths.cbegin(), image_paths.cend(), images_memory, min_size).second;
}

/// @brief Saves images from memory object into image files specified by file paths.
///
/// @details Saves images to files pointed by paths from from [@p image_paths_first; @p image_paths_last) range.
/// @n
/// @n       Each saved image is optionally resized (aspect ratio is preserved, so the width or height,
///          whichever is lower, is set to @p min_size) before saving.
/// @n
/// @n       The function tries to save images to paths from the range until batch size is reached
///          (also stored in layout) or all image files are saved.
///
/// @tparam MemElemTy Type of memory object elements which will be converted back to pixels from image.
///
/// @param image_paths_first Range of paths (relative or absolute) for image files (inclusive start).
/// @param image_paths_last  Range of paths (relative or absolute) for image files (exclusive end).
/// @param images_memory     Memory object containing metadata and buffers with stored images.
/// @param min_size          Size of either width or height (which is lower) to which image
///                          should be resized (keeping aspect ratio) before saving.
///                          If @c 0 is specified, the image is not resized.
/// @return                  Saved images (please note that you have to introduce fipImage
///                          to utilize return fully, e.g. via including image_toolkit.h.
///
/// @exception std::logic_error   Specified memory element type is different than declared in layout
///                               of memory object.
/// @exception std::runtime_error Passed memory object has unsupported layout format (non-BYXF).
/// @exception std::runtime_error One of spatial sizes defined in layout of memory object is non-positive.
template <typename MemElemTy = float>
auto save_image_files(std::vector<std::string>::const_iterator image_paths_first,
                      std::vector<std::string>::const_iterator image_paths_last,
                      const memory& images_memory, unsigned min_size = 0)
    -> std::pair<std::vector<std::string>::const_iterator, std::vector<std::shared_ptr<fipImage>>>;

/// @brief Saves images from memory object into image files specified by file paths.
///
/// @details Saves images to files pointed by paths from @p image_paths.
/// @n
/// @n       Each saved image is optionally resized (aspect ratio is preserved, so the width or height,
///          whichever is lower, is set to @p min_size) before saving.
/// @n
/// @n       The function tries to save images to paths from the list until batch size is reached
///          (also stored in layout) or all image files are saved.
///
/// @tparam MemElemTy Type of memory object elements which will be converted back to pixels from image.
///
/// @param image_paths   List of paths (relative or absolute) for image files.
/// @param images_memory Memory object containing metadata and buffers with stored images.
/// @param min_size      Size of either width or height (which is lower) to which image
///                      should be resized (keeping aspect ratio) before saving.
///                      If @c 0 is specified, the image is not resized.
/// @return              Saved images (please note that you have to introduce fipImage
///                      to utilize return fully, e.g. via including image_toolkit.h.
///
/// @exception std::logic_error   Specified memory element type is different than declared in layout
///                               of memory object.
/// @exception std::runtime_error Passed memory object has unsupported layout format (non-BYXF).
/// @exception std::runtime_error One of spatial sizes defined in layout of memory object is non-positive.
template <typename MemElemTy = float>
auto save_image_files(const std::vector<std::string>& image_paths, const memory& images_memory,
                      const unsigned min_size = 0)
    -> std::vector<std::shared_ptr<fipImage>>
{
    return save_image_files<MemElemTy>(image_paths.cbegin(), image_paths.cend(), images_memory, min_size).second;
}

} // namespace examples
} // namespace utils
} // namespace cldnn

// --------------------------------------------------------------------------------------------------------------------
// TODO: Refactor rest of load functions.
// --------------------------------------------------------------------------------------------------------------------

std::string get_image_file(const std::string& img_name, const std::vector<std::string>& images_list);

template <typename MemElemTy = float>
void load_data_from_file_list_lenet(const std::vector<std::string>& images_list, cldnn::memory& memory, uint32_t images_offset, uint32_t images_number, bool train, cldnn::memory& memory_labels);

template <typename MemElemTy = float>
void load_data_from_file_list_imagenet(const std::vector<std::string>& images_list, const std::string& input_dir, cldnn::memory& memory, uint32_t images_offset, uint32_t images_number, bool train, cldnn::memory& memory_labels);

template <typename MemElemTy = float>
void load_data_from_file_list_cifar10(const std::vector<std::string>& images_list, const std::string& input_dir, cldnn::memory& memory, uint32_t images_offset, uint32_t images_number, bool train, cldnn::memory& memory_labels);
