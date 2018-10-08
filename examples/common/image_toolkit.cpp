// Copyright (c) 2016, 2018 Intel Corporation
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

#include "image_toolkit.h"

#include <boost/filesystem.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <deque>
#include <mutex>
#include <sstream>
#include <utility>
#include <numeric>


namespace cldnn
{
namespace utils
{
namespace examples
{
namespace image_toolkit
{
// Aliases.
namespace bfs = boost::filesystem;

namespace
{
/// @brief Helper: Crop rectangle from left-top point to right-bottom point (right and bottom edge exclusive).
struct crop_rect
{
    unsigned left, top, right, bottom;
};

/// @brief Helper: Image spatial size.
struct spatial_size
{
    unsigned width;
    unsigned height;
};


/// @brief Storage for error messages.
struct error_messages_storage
{
    /// @brief Synchronization root for this storage.
    std::mutex sync_root;
    /// @brief Collection of error messages.
    std::deque<std::string> messages;
};

/// @brief Provides access to static storage containing errors from FreeImage library.
///
/// @return Storage for error messages (with synchronization capability).
error_messages_storage& get_fi_error_messages_storage()
{
    static error_messages_storage error_messages; // call-once, sync
    return error_messages;
}

/// @brief Gets error messages from FreeImage library.
///
/// @param drop      Indicates that after returning error message, the message should be removed from error
///                  messages storage.
/// @param only_last Indicates that only last error message should be collected.
/// @n               If @c false, all messages are joined (new line as separator) and returned.
/// @return          Error message(s).
std::string get_fi_error_message(const bool drop = true, const bool only_last = false)
{
    auto& storage = get_fi_error_messages_storage();
    std::lock_guard<std::mutex> storage_lock(storage.sync_root);

    if (storage.messages.empty())
        return {};
    if (only_last)
    {
        const auto msg = storage.messages.back();
        if (drop)
            storage.messages.pop_back();
        return msg;
    }

    std::ostringstream out;
    auto sep = "";
    for (const auto& message : storage.messages)
    {
        out << sep << message;
        sep = "\n";
    }
    if (drop)
        storage.messages.clear();
    return out.str(); // RVO
}

/// @brief Error handler for FreeImage library.
///
/// @param format  Format that caused issues.
/// @param message Error message.
void free_image_error_collection_handler(const FREE_IMAGE_FORMAT format, const char* message)
{
    // How many messages should collection of messages have at peak.
    constexpr auto max_messages = 100U;

    std::stringstream out;
    if (format != FIF_UNKNOWN)
        out << "[" << FreeImage_GetFormatFromFIF(format) << "] ";
    out << message;

    auto& storage = get_fi_error_messages_storage();
    std::lock_guard<std::mutex> storage_lock(storage.sync_root);

    storage.messages.emplace_back(out.str());
    if (storage.messages.size() > max_messages)
        storage.messages.pop_front();
}

/// @brief Installs FreeImage error handler (if it not installed already).
///
/// @param force Forces handler re-installation.
void install_fi_error_handler(const bool force = false)
{
    static std::mutex handler_sync_root;
    static auto handler_installed = false;

    std::lock_guard<std::mutex> handler_lock(handler_sync_root);

    if (!handler_installed || force)
    {
        FreeImage_SetOutputMessage(free_image_error_collection_handler);
        handler_installed = true;
    }
}


/// @brief Missing function from fipImage: Rescales rectangular part of current image.
///
/// @param [in, out] image Current image.
/// @param dst_width       Target width of image.
/// @param dst_height      Target height of image.
/// @param left            X coordinate of top-left point of crop rectangle.
/// @param top             Y coordinate of top-left point of crop rectangle.
/// @param right           X coordinate of bottom-right point of crop rectangle.
/// @param bottom          Y coordinate of bottom-right point of crop rectangle.
/// @param filter          Rescaling filter.
/// @return                Indicates that operation succeeded.
// ReSharper disable once CppInconsistentNaming
bool fipImage_rescaleRect(fipImage& image, const unsigned dst_width, const unsigned dst_height,
                                 const int left, const int top, const int right, const int bottom,
                                 const FREE_IMAGE_FILTER filter = FILTER_CATMULLROM)
{
    if (image.isValid())
    {
        switch (image.getImageType())
        {
        case FIT_BITMAP:
        case FIT_UINT16:
        case FIT_RGB16:
        case FIT_RGBA16:
        case FIT_FLOAT:
        case FIT_RGBF:
        case FIT_RGBAF:
            break;
        default:
            return false;
        }

        const auto dst = FreeImage_RescaleRect(image, static_cast<int>(dst_width), static_cast<int>(dst_height),
                                               left, top, right, bottom, filter);

        if (dst == nullptr)
            return false;
        image = dst;
        return image.isValid() != FALSE;
    }
    return false;
}

} // unnamed namespace


std::shared_ptr<fipImage> load(const std::string& image_file_path)
{
    install_fi_error_handler();

    auto image = std::make_shared<fipImage>();

    const auto load_success_flag = image->load(image_file_path.c_str());
    if (!load_success_flag)
        throw std::runtime_error("Failed to load image (image library reported failure): \"" +
                                 image_file_path + "\".\n    Details: " + get_fi_error_message());
    return image;
}

std::shared_ptr<fipImage> save(std::shared_ptr<fipImage> image, const std::string& image_file_path)
{
    install_fi_error_handler();

    if (image == nullptr)
        throw std::invalid_argument("Parameter must be set (not nullptr): \"image\"");

    auto image_dir = bfs::path(image_file_path).parent_path();
    if (!image_dir.empty())
        create_directories(image_dir);

    const auto store_success_flag = image->save(image_file_path.c_str());
    if (!store_success_flag)
        throw std::runtime_error("Failed to save image (image library reported failure): \"" +
                                 image_file_path + "\".\n    Details: " + get_fi_error_message());
    return image;
}

std::shared_ptr<fipImage> crop_resize(std::shared_ptr<fipImage> image,
                                      const unsigned width, const unsigned height)
{
    install_fi_error_handler();

    // Indicates number of pixels in calculated width or height below which
    // image is pre-resized before cropping (so it will contain more useful data).
    constexpr auto pre_resize_threshold = 10U;

    if (image == nullptr)
        throw std::invalid_argument("Parameter must be set (not nullptr): \"image\"");
    if (width <= 0)
        throw std::invalid_argument("Parameter must be positive: \"width\"");
    if (height <= 0)
        throw std::invalid_argument("Parameter must be positive: \"height\"");

    // Use simple crop-resize method if cropping to square is detected.
    if (width == height)
        return crop_resize(std::move(image), width); // RVO

    spatial_size image_dims           = {image->getWidth(), image->getHeight()};
    const spatial_size out_image_dims = {width, height};

    const auto image_aspect_ratio     = static_cast<const float>(image_dims.width) / image_dims.height;
    const auto out_image_aspect_ratio = static_cast<const float>(out_image_dims.width) / out_image_dims.height;

    if (image_aspect_ratio == out_image_aspect_ratio)
        return resize(std::move(image), out_image_dims.width, out_image_dims.height); // RVO

    crop_rect square_rect;  // NOLINT(cppcoreguidelines-pro-type-member-init, hicpp-member-init)
    if (image_aspect_ratio > out_image_aspect_ratio)
    {
        spatial_size crop_image_dims = {static_cast<unsigned>(std::round(image_dims.height * out_image_aspect_ratio)), image_dims.height};
        if (crop_image_dims.width < std::min(image_dims.width, pre_resize_threshold))
        {
            resize_keep_ar(image, out_image_dims.height);
            image_dims      = {image->getWidth(), image->getHeight()};
            crop_image_dims = out_image_dims;
        }

        const auto cut_size      = image_dims.width - crop_image_dims.width;
        const auto cut_margin    = cut_size / 2;
        const auto cut_remainder = cut_size - 2 * cut_margin;

        square_rect = {cut_margin, 0, image_dims.width - cut_margin - cut_remainder, image_dims.height};
    }
    else
    {
        spatial_size crop_image_dims = {image_dims.width, static_cast<unsigned>(std::round(image_dims.width / out_image_aspect_ratio))};
        if (crop_image_dims.height < std::min(image_dims.height, pre_resize_threshold))
        {
            resize_keep_ar(image, out_image_dims.width);
            image_dims      = {image->getWidth(), image->getHeight()};
            crop_image_dims = out_image_dims;
        }

        const auto cut_size      = image_dims.height - crop_image_dims.height;
        const auto cut_margin    = cut_size / 2;
        const auto cut_remainder = cut_size - 2 * cut_margin;

        square_rect = {0, cut_margin, image_dims.width, image_dims.height - cut_margin - cut_remainder};
    }

    const auto rescale_success = fipImage_rescaleRect(*image,
                                                      out_image_dims.width, out_image_dims.height,
                                                      square_rect.left,
                                                      square_rect.top,
                                                      square_rect.right,
                                                      square_rect.bottom,
                                                      FILTER_CATMULLROM);
    if (!rescale_success)
        throw std::runtime_error("Failed to rescale square part of image (image library reported failure)."
                                 "\n    Details: " + get_fi_error_message());

    return image; // NRVO
}

std::shared_ptr<fipImage> crop_resize(std::shared_ptr<fipImage> image,
                                      const unsigned square_size)
{
    install_fi_error_handler();

    if (image == nullptr)
        throw std::invalid_argument("Parameter must be set (not nullptr): \"image\"");

    const spatial_size image_dims = {image->getWidth(), image->getHeight()};

    if (image_dims.width == image_dims.height)
        return resize(std::move(image), square_size);

    crop_rect square_rect;  // NOLINT(cppcoreguidelines-pro-type-member-init, hicpp-member-init)
    if (image_dims.width > image_dims.height)
    {
        const auto cut_size      = image_dims.width - image_dims.height;
        const auto cut_margin    = cut_size / 2;
        const auto cut_remainder = cut_size - 2 * cut_margin;

        square_rect = {cut_margin, 0, image_dims.width - cut_margin - cut_remainder, image_dims.height};
    }
    else
    {
        const auto cut_size      = image_dims.height - image_dims.width;
        const auto cut_margin    = cut_size / 2;
        const auto cut_remainder = cut_size - 2 * cut_margin;

        square_rect = {0, cut_margin, image_dims.width, image_dims.height - cut_margin - cut_remainder};
    }

    const auto rescale_success = fipImage_rescaleRect(*image,
                                                      square_size, square_size,
                                                      square_rect.left,
                                                      square_rect.top,
                                                      square_rect.right,
                                                      square_rect.bottom,
                                                      FILTER_CATMULLROM);
    if (!rescale_success)
        throw std::runtime_error("Failed to rescale square part of image (image library reported failure)."
                                 "\n    Details: " + get_fi_error_message());

    return image; // NRVO
}

std::shared_ptr<fipImage> resize(std::shared_ptr<fipImage> image, const unsigned width, const unsigned height)
{
    install_fi_error_handler();

    if (image == nullptr)
        throw std::invalid_argument("Parameter must be set (not nullptr): \"image\"");
    if (width <= 0)
        throw std::invalid_argument("Parameter must be positive: \"width\"");
    if (height <= 0)
        throw std::invalid_argument("Parameter must be positive: \"height\"");

    const spatial_size image_dims     = {image->getWidth(), image->getHeight()};
    const spatial_size out_image_dims = {width, height};

    if (image_dims.width == out_image_dims.width && image_dims.height == out_image_dims.height)
        return image; // NRVO

    const auto rescale_success = image->rescale(out_image_dims.width, out_image_dims.height, FILTER_CATMULLROM);
    if (!rescale_success)
        throw std::runtime_error("Failed to rescale input image (image library reported failure."
                                 "\n    Details: " + get_fi_error_message());

    return image; // NRVO
}

std::shared_ptr<fipImage> resize(std::shared_ptr<fipImage> image, const unsigned square_size)
{
    install_fi_error_handler();

    if (image == nullptr)
        throw std::invalid_argument("Parameter must be set (not nullptr): \"image\"");

    const spatial_size image_dims = {image->getWidth(), image->getHeight()};

    const auto out_square_size = square_size > 0 ? square_size : std::max(image_dims.width, image_dims.height);

    return resize(std::move(image), out_square_size, out_square_size); // RVO
}

std::shared_ptr<fipImage> resize_keep_ar(std::shared_ptr<fipImage> image, const unsigned min_size)
{
    install_fi_error_handler();

    if (image == nullptr)
        throw std::invalid_argument("Parameter must be set (not nullptr): \"image\"");
    if (min_size <= 0)
        throw std::invalid_argument("Parameter must be positive: \"min_size\"");

    const spatial_size image_dims = {image->getWidth(), image->getHeight()};

    const auto aspect_ratio = static_cast<const float>(image_dims.width) / image_dims.height;

    spatial_size out_image_dims;  // NOLINT(cppcoreguidelines-pro-type-member-init, hicpp-member-init)
    if (image_dims.width <= image_dims.height)
        out_image_dims = {min_size, static_cast<unsigned>(std::round(min_size / aspect_ratio))};
    else
        out_image_dims = {static_cast<unsigned>(std::round(min_size * aspect_ratio)), min_size};

    return resize(std::move(image), out_image_dims.width, out_image_dims.height); // RVO
}


diff_results diff(const std::shared_ptr<const fipImage>& actual, const std::shared_ptr<const fipImage>& reference,
                  const bool gen_channel_images, const bool gen_histograms, const bool use_crop_resize)
{
    diff_results results;

    auto img_actual    = actual;
    auto img_reference = reference;

    if (actual == nullptr)
        throw std::invalid_argument("Parameter must be set (not nullptr): \"actual\"");
    if (reference == nullptr)
        throw std::invalid_argument("Parameter must be set (not nullptr): \"reference\"");
    if (!img_actual->isValid() || img_actual->accessPixels() == nullptr)
        throw std::runtime_error("Failed to compare images (actual image is invalid or contains no pixel data)");
    if (!img_reference->isValid() || img_reference->accessPixels() == nullptr)
        throw std::runtime_error("Failed to compare images (reference image is invalid or contains no pixel data)");

    // Unify BPP of images to 24 bits.
    if (img_actual->getBitsPerPixel() != 24)
    {
        auto converted_image       = std::make_shared<fipImage>(*img_actual);
        const auto convert_success = converted_image->convertTo24Bits();
        if (!convert_success || !converted_image->isValid())
            throw std::runtime_error("Failed to convert actual image to 24-bit pixel format "
                                     "(image library reported failure)."
                                     "\n    Details: " + get_fi_error_message());
        img_actual = converted_image;
    }

    // Unify sizes of images (reference is crop-resized or resized).
    if (img_actual->getWidth() != img_reference->getWidth() || img_actual->getHeight() != img_reference->getHeight() ||
        img_reference->getBitsPerPixel() != 24)
    {
        auto normalized_image = std::make_shared<fipImage>(*img_reference);

        if (img_actual->getWidth() != img_reference->getWidth() ||
            img_actual->getHeight() != img_reference->getHeight())
        {
            if (use_crop_resize)
                crop_resize(normalized_image, img_actual->getWidth(), img_actual->getHeight());
            else
                resize(normalized_image, img_actual->getWidth(), img_actual->getHeight());
        }

        if (img_reference->getBitsPerPixel() != 24)
        {
            const auto convert_success = normalized_image->convertTo24Bits();
            if (!convert_success || !normalized_image->isValid())
                throw std::runtime_error("Failed to convert actual image to 24-bit pixel format "
                                         "(image library reported failure)."
                                         "\n    Details: " + get_fi_error_message());
        }

        img_reference = normalized_image;
    }

    assert(img_actual->getHeight() == img_reference->getHeight() &&
           img_actual->getLine() == img_reference->getLine() &&
           "Sizes of images are still different after unification.");

    const auto width  = img_actual->getWidth();
    const auto height = img_actual->getHeight();
    const auto bpp    = img_actual->getBitsPerPixel();

    results.image = std::make_shared<fipImage>(FIT_BITMAP, width, height, bpp);

    const auto bytes_per_pixel = img_actual->getLine() / width;

    for (unsigned y = 0; y < height; ++y)
    {
        auto img_actual_pixel    = reinterpret_cast<const std::uint8_t*>(img_actual->getScanLine(y));
        auto img_reference_pixel = reinterpret_cast<const std::uint8_t*>(img_reference->getScanLine(y));
        auto img_diff_pixel      = reinterpret_cast<std::uint8_t*>(results.image->getScanLine(y));
        assert(img_actual_pixel != nullptr && "Inaccessible scanline in actual image.");
        assert(img_reference_pixel != nullptr && "Inaccessible scanline in reference image.");
        assert(img_diff_pixel != nullptr && "Inaccessible scanline in diff image.");

        for (unsigned x = 0; x < width; ++x)
        {
            auto diff_blue  = std::abs(img_actual_pixel[FI_RGBA_BLUE]  - img_reference_pixel[FI_RGBA_BLUE]);
            auto diff_green = std::abs(img_actual_pixel[FI_RGBA_GREEN] - img_reference_pixel[FI_RGBA_GREEN]);
            auto diff_red   = std::abs(img_actual_pixel[FI_RGBA_RED]   - img_reference_pixel[FI_RGBA_RED]);

            using channel_diff_type = decltype(diff_blue);
            constexpr channel_diff_type min_val = std::numeric_limits<std::uint8_t>::min();
            constexpr channel_diff_type max_val = std::numeric_limits<std::uint8_t>::max();
            constexpr double channel_max_diff = max_val - min_val;
            constexpr auto pixel_max_diff     = 3 * channel_max_diff;
            constexpr auto pixel_1pc_diff     = 0.01 * pixel_max_diff;
            constexpr auto pixel_5pc_diff     = 0.05 * pixel_max_diff;

            const double fp_diff_blue  = diff_blue;
            const double fp_diff_green = diff_green;
            const double fp_diff_red   = diff_red;

            const auto fp_diff = fp_diff_blue + fp_diff_green + fp_diff_red;

            results.sad += fp_diff;
            results.ssd += fp_diff_blue * fp_diff_blue + fp_diff_green * fp_diff_green + fp_diff_red * fp_diff_red;
            if (fp_diff > 0) { ++results.diff_pixel_freq; }
            if (fp_diff > pixel_1pc_diff) { ++results.diff_1pc_pixel_freq; }
            if (fp_diff > pixel_5pc_diff) { ++results.diff_5pc_pixel_freq; }

            img_diff_pixel[FI_RGBA_BLUE]  = static_cast<std::uint8_t>(std::min(std::max(diff_blue,  min_val), max_val));
            img_diff_pixel[FI_RGBA_GREEN] = static_cast<std::uint8_t>(std::min(std::max(diff_green, min_val), max_val));
            img_diff_pixel[FI_RGBA_RED]   = static_cast<std::uint8_t>(std::min(std::max(diff_red,   min_val), max_val));

            img_actual_pixel    += bytes_per_pixel;
            img_reference_pixel += bytes_per_pixel;
            img_diff_pixel      += bytes_per_pixel;
        }
    }

    const auto pixels_count = static_cast<double>(width) * static_cast<double>(height);
    if (pixels_count > 0)
    {
        results.sad_mean = results.sad / pixels_count;
        results.ssd_mean = results.ssd / pixels_count;

        results.diff_pixel_freq /= pixels_count;
        results.diff_1pc_pixel_freq /= pixels_count;
        results.diff_5pc_pixel_freq /= pixels_count;
    }

    if (gen_channel_images)
    {
        results.image_red   = std::make_shared<fipImage>();
        results.image_green = std::make_shared<fipImage>();
        results.image_blue  = std::make_shared<fipImage>();
        const auto convert_success =
            results.image->splitChannels(*results.image_red, *results.image_green, *results.image_blue);
        if (!convert_success ||
            !results.image_red->isValid() || !results.image_green->isValid() || !results.image_blue->isValid())
        {
            throw std::runtime_error("Failed to split diff image into 8-bit channel images "
                                     "(image library reported failure)."
                                     "\n    Details: " + get_fi_error_message());
        }
    }

    if (gen_histograms)
    {
        results.histogram = std::make_shared<std::array<std::uint32_t, 256>>();
        const auto gen_success = results.image->getHistogram(reinterpret_cast<DWORD*>(results.histogram->data()));
        if (!gen_success)
        {
            throw std::runtime_error("Failed to generate histogram from diff image "
                                     "(image library reported failure)."
                                     "\n    Details: " + get_fi_error_message());
        }

        if (gen_channel_images)
        {
            results.histogram_red   = std::make_shared<std::array<std::uint32_t, 256>>();
            results.histogram_green = std::make_shared<std::array<std::uint32_t, 256>>();
            results.histogram_blue  = std::make_shared<std::array<std::uint32_t, 256>>();
            const auto gen_red_success =
                results.image->getHistogram(reinterpret_cast<DWORD*>(results.histogram_red->data()), FICC_RED);
            const auto gen_green_success =
                results.image->getHistogram(reinterpret_cast<DWORD*>(results.histogram_green->data()), FICC_GREEN);
            const auto gen_blue_success =
                results.image->getHistogram(reinterpret_cast<DWORD*>(results.histogram_blue->data()), FICC_BLUE);
            if (!gen_red_success || !gen_green_success || !gen_blue_success)
            {
                throw std::runtime_error("Failed to generate channel histograms from diff image "
                                         "(image library reported failure)."
                                         "\n    Details: " + get_fi_error_message());
            }
        }
    }

    return results; // NRVO
}

std::shared_ptr<fipImage> create_histogram(const std::shared_ptr<std::array<std::uint32_t, 256>>& data)
{
    constexpr unsigned columns = std::tuple_size<std::decay_t<decltype(*data)>>::value;

    constexpr auto bpp    = 8U;
    constexpr auto scale  = 2U;
    constexpr auto width  = scale * columns;
    constexpr auto height = scale * 100U;

    if (data == nullptr)
        throw std::invalid_argument("Parameter must be set (not nullptr): \"data\"");

    // Normalize data.
    const auto max_data_val = *std::max_element(data->cbegin(), data->cend());
    const auto data_val_scale = max_data_val != 0 ? 1.0 * height / max_data_val : 0.0;
    std::array<double, columns> norm_data;  // NOLINT(cppcoreguidelines-pro-type-member-init, hicpp-member-init)
    std::transform(data->cbegin(), data->cend(), norm_data.begin(),
                   [=](const std::uint32_t& val) { return data_val_scale * val; });

    auto image = std::make_shared<fipImage>(FIT_BITMAP, width, height, bpp);
    const auto bytes_per_pixel = image->getLine() / width;

    for (auto y = 0U; y < height; ++y)
    {
        auto image_pixel = reinterpret_cast<std::uint8_t*>(image->getScanLine(y));
        assert(image_pixel != nullptr && "Inaccessible pixels of freshly created image.");

        constexpr auto min_val = std::numeric_limits<std::uint8_t>::min();
        constexpr auto max_val = std::numeric_limits<std::uint8_t>::max();

        for (auto c = 0U; c < columns; ++c) // column
        {
            const auto norm_data_val = norm_data[c];
            if (norm_data_val >= y + 1)
            {
                for (auto w = 0U; w < scale; ++w)
                {
                    *image_pixel = min_val;
                    image_pixel += bytes_per_pixel;
                }
            }
            else if (norm_data_val <= y)
            {
                for (auto w = 0U; w < scale; ++w)
                {
                    *image_pixel = max_val;
                    image_pixel += bytes_per_pixel;
                }
            }
            else
            {
                const auto alpha = norm_data_val - y;
                const auto blend_val = static_cast<std::uint8_t>(std::round(min_val * alpha + max_val * (1 - alpha)));

                for (auto w = 0U; w < scale; ++w)
                {
                    *image_pixel = blend_val;
                    image_pixel += bytes_per_pixel;
                }
            }
        }
    }
    return image; // NRVO
}

} // image_toolkit
} // namespace examples
} // namespace utils
} // namespace cldnn
