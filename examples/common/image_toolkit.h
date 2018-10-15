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

#pragma once

// Handle corrosive windows.h macros.
#ifndef NOMINMAX
    #define NOMINMAX
    #define FIP_NOMINMAX_NEEDS_CLEANUP
#endif

#include <FreeImagePlus.h>

#ifdef FIP_NOMINMAX_NEEDS_CLEANUP
    #undef FIP_NOMINMAX_NEEDS_CLEANUP
    #undef NOMINMAX
#endif

#include <api/CPP/cldnn_defs.h>

#include <array>
#include <cassert>
#include <cstdint>
#include <iterator>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>

// --------------------------------------------------------------------------------------------------------------------
// Toolkit providing basic image support (based on FreeImagePlus).
// --------------------------------------------------------------------------------------------------------------------

namespace cldnn
{
namespace utils
{
namespace examples
{
namespace image_toolkit
{
/// @brief Loads image from file.
///
/// @param image_file_path Path (absolute or relative) to image file which will be loaded.
/// @return                Newly created image file (as shared pointer).
///
/// @exception std::runtime_error Loading of image failed (missing / inaccessible file or invalid / unsupported
///                               file format).
std::shared_ptr<fipImage> load(const std::string& image_file_path);

/// @brief Stores / Saves image from file.
///
/// @param image           Input image (it will not be modified).
/// @param image_file_path Path (absolute or relative) of image file to which image will be written.
/// @return                Input image (pass-thru).
///
/// @exception std::invalid_argument Image is not set (@c nullptr).
/// @exception std::runtime_error    Saving of image failed (inaccessible file path or write is unsupported
///                                  to specified image type).
std::shared_ptr<fipImage> save(std::shared_ptr<fipImage> image, const std::string& image_file_path);

/// @brief Crops image to rectangle and resizes it (Catmull-Rom filter; aspect ratio preserved via
///        cropping image to square).
///
/// @details Image is resized to rectangle of specified dimensions.
/// @n
/// @n       Aspect ratio is preserved by cropping initial image to square and resizing it.
///
/// @param image       Input image (it will be modified in-place).
/// @param width       Width (in pixels) of target rectangle (to which image will be resized).
/// @param height      Height (in pixels) of target rectangle (to which image will be resized).
/// @return            Input image (pass-thru after modifications).
///
/// @exception std::invalid_argument Image is not set (@c nullptr).
/// @exception std::invalid_argument Either @p width or @p height is non-positive.
/// @exception std::runtime_error Crop or resize of image failed.
std::shared_ptr<fipImage> crop_resize(std::shared_ptr<fipImage> image, unsigned width, unsigned height);

/// @brief Crops image to square and resizes it (Catmull-Rom filter; aspect ratio preserved via
///        cropping image to square).
///
/// @details Image is resized to square of specified size (if @p square_size is greater than @c 0) or
///          of size of width or height, whichever is shorter (size after cropping).
/// @n
/// @n       Aspect ratio is preserved by cropping initial image to square and resizing it.
///
/// @param image       Input image (it will be modified in-place).
/// @param square_size Size (in pixels) of target square side (to which image will be resized).
/// @return            Input image (pass-thru after modifications).
///
/// @exception std::invalid_argument Image is not set (@c nullptr).
/// @exception std::runtime_error    Crop or resize of image failed.
std::shared_ptr<fipImage> crop_resize(std::shared_ptr<fipImage> image, unsigned square_size = 0);

/// @brief Resizes image (Catmull-Rom filter; aspect ratio NOT preserved).
///
/// @details Image is resized to rectangle of specified @p width and @p height.
/// @n
/// @n       Does not preserve aspect ratio.
///
/// @param image       Input image (it will be modified in-place).
/// @param width       Width (in pixels) of target rectangle (to which image will be resized).
/// @param height      Height (in pixels) of target rectangle (to which image will be resized).
/// @return            Input image (pass-thru after modifications).
///
/// @exception std::invalid_argument Image is not set (@c nullptr).
/// @exception std::invalid_argument Either @p width or @p height is non-positive.
/// @exception std::runtime_error    Resize of image failed.
std::shared_ptr<fipImage> resize(std::shared_ptr<fipImage> image, unsigned width, unsigned height);

/// @brief Resizes image to square (Catmull-Rom filter; aspect ratio NOT preserved).
///
/// @details Image is resized to square of specified size (if @p square_size is greater than @c 0) or
///          of size of width or height, whichever is longer.
/// @n
/// @n       Does not preserve aspect ratio.
///
/// @param image       Input image (it will be modified in-place).
/// @param square_size Size (in pixels) of target square side (to which image will be resized).
/// @return            Input image (pass-thru after modifications).
///
/// @exception std::invalid_argument Image is not set (@c nullptr).
/// @exception std::runtime_error    Resize of image failed.
std::shared_ptr<fipImage> resize(std::shared_ptr<fipImage> image, unsigned square_size = 0);

/// @brief Resizes image in the way that aspect ratio is preserved (Catmull-Rom filter; aspect ratio preserved).
///
/// @details Image is resized in a way that aspect ratio is preserved, but lower of dimensions (width or height)
///          is changed to specified value (@p min_size).
/// @n
/// @n       Preserves aspect ratio.
///
/// @param image    Input image (it will be modified in-place).
/// @param min_size Size (in pixels) of target minimum dimension of resized image (either width or height will
///                 of destination image will be set to this value - depending which is lower).
/// @return         Input image (pass-thru after modifications).
///
/// @exception std::invalid_argument Image is not set (@c nullptr).
/// @exception std::invalid_argument @p min_size is non-positive.
/// @exception std::runtime_error    Resize of image failed.
std::shared_ptr<fipImage> resize_keep_ar(std::shared_ptr<fipImage> image, unsigned min_size);

// --------------------------------------------------------------------------------------------------------------------

/// @brief Results from image comparison.
struct diff_results
{
    /// @brief Sum of absolute difference of all pixels between images.
    double sad      = 0.0;
    /// @brief Per pixel mean value of sum of absolute difference.
    double sad_mean = 0.0;

    /// @brief Sum of squared difference of all pixels between images.
    double ssd      = 0.0;
    /// @brief Per pixel mean value of sum of squared difference.
    double ssd_mean = 0.0;

    /// @brief Ratio of number of different pixels to the number of all pixels in compared images.
    double diff_pixel_freq     = 0.0;
    /// @brief Ratio of number of moderately different pixels (more than 1 percentile of possible difference)
    ///        to the number of all pixels in compared images.
    double diff_1pc_pixel_freq = 0.0;
    /// @brief Ratio of number of highly different pixels (more than 5 percentile of possible difference)
    ///        to the number of all pixels in compared images.
    double diff_5pc_pixel_freq = 0.0;

    /// @brief Difference image. 24-bit image where each pixel is calculated as absolute difference
    ///        of compared images.
    /// @n     Each channel value is calculated separately and values outside valid range are clamped
    ///        to valid pixel range.
    std::shared_ptr<fipImage> image;

    /// @brief Difference image of red channel. 8-bit image where red channel of each pixel is calculated
    ///        as absolute difference of red channels of compared images.
    /// @n     Values outside valid range are clamped to valid pixel range.
    std::shared_ptr<fipImage> image_red;
    /// @brief Difference image of green channel. 8-bit image where green channel of each pixel is calculated
    ///        as absolute difference of green channels of compared images.
    /// @n     Values outside valid range are clamped to valid pixel range.
    std::shared_ptr<fipImage> image_green;
    /// @brief Difference image of blue channel. 8-bit image where blue channel of each pixel is calculated
    ///        as absolute difference of blue channels of compared images.
    /// @n     Values outside valid range are clamped to valid pixel range.
    std::shared_ptr<fipImage> image_blue;

    /// @brief Histogram of diff_results::image.
    std::shared_ptr<std::array<std::uint32_t, 256>> histogram;

    /// @brief Histogram of diff_results::image_red.
    std::shared_ptr<std::array<std::uint32_t, 256>> histogram_red;
    /// @brief Histogram of diff_results::image_green.
    std::shared_ptr<std::array<std::uint32_t, 256>> histogram_green;
    /// @brief Histogram of diff_results::image_blue.
    std::shared_ptr<std::array<std::uint32_t, 256>> histogram_blue;
};

/// @brief Compares two images and returns similarity information.
///
/// @details Compares two images and returns diff image containing (clamped) absolute difference of each pixel
///          and multiple statistics (SAD, SSD, mean SAD, mean SSD, diff pixel frequencies).
/// @n       If actual and reference images have different dimensions, the reference image is resized to
///          dimensions of actual image before comparing them.
///
/// @param actual             Actual image to compare.
/// @param reference          Reference image (golden image).
/// @param gen_channel_images Indicates that single-channel diff images should be also outputted in results.
/// @param gen_histograms     Indicates that histograms of diff images should be also included in results.
/// @param use_crop_resize    Indicates that, when @p actual and @p reference images have different
///                           dimensions, the reference image should be resized using crop_resize (keeps aspect
///                           ratio, but crops part of image).
///                           If @c false, the reference image is resized (if necessary) using resize (does not keep
///                           aspect ratio).
///
/// @return                   Comparison results. Please note that depending on parameters some fields
///                           can be @c nullptr / empty.
///
/// @exception std::invalid_argument One of images is not set (@c nullptr).
/// @exception std::runtime_error    Input images are invalid.
/// @exception std::runtime_error    Manipulation of image failed (FreeImagePlus reported errors).
diff_results diff(const std::shared_ptr<const fipImage>& actual, const std::shared_ptr<const fipImage>& reference,
                  bool gen_channel_images = true, bool gen_histograms = true, bool use_crop_resize = true);

/// @brief Creates histogram image from histogram data (8-bit, greyscale).
///
/// @param data Histogram data.
/// @return     Image with histogram.
///
/// @exception std::invalid_argument Histogram data is not set (@c nullptr).
std::shared_ptr<fipImage> create_histogram(const std::shared_ptr<std::array<std::uint32_t, 256>>& data);

// --------------------------------------------------------------------------------------------------------------------

namespace detail
{
/// @brief Converts pixel channel value (8-bit) to corresponding element type.
///
/// @tparam ElemTy Type to which pixel value should be converted.
/// @n             Type must be arithmetic (otherwise the function is not considered as candidate
///                overload).
///
/// @param val Input value.
/// @return    Element representation with the same numerical value.
template <typename ElemTy = float>
auto convert_pixel_channel(const std::uint8_t val)
    -> std::enable_if_t<std::is_arithmetic<ElemTy>::value, ElemTy>
{
    return static_cast<ElemTy>(val);
}

/// @brief Converts pixel channel value (8-bit) to corresponding half-precision floating
///        point value (no normalization).
///
/// @tparam ElemTy Type to which pixel value should be converted (same as half_t).
///
/// @param val Input value.
/// @return    Half representation with the same numerical value.
template <typename ElemTy = half_t>
auto convert_pixel_channel(const std::uint8_t val)
    -> std::enable_if_t<std::is_same<std::remove_cv_t<ElemTy>, half_t>::value, half_t>
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


/// @brief Converts element value back to corresponding pixel channel value (8-bit).
///
/// @tparam ElemTy Type which value should be converted back to pixel value.
/// @n             Type must be arithmetic (otherwise the function is not considered as candidate
///                overload).
///
/// @param val Input value.
/// @return    Pixel value with the same/nearest numerical value.
template <typename ElemTy = float>
auto convert_back_pixel_channel(const ElemTy val)
    -> std::enable_if_t<std::is_arithmetic<ElemTy>::value, std::uint8_t>
{
    using clamp_type = std::common_type_t<ElemTy, std::uint8_t>;

    constexpr auto min_val = static_cast<clamp_type>(std::numeric_limits<std::uint8_t>::min());
    constexpr auto max_val = static_cast<clamp_type>(std::numeric_limits<std::uint8_t>::max());

    const auto ext_val = static_cast<clamp_type>(val);
    return static_cast<std::uint8_t>(ext_val >= min_val ? (ext_val <= max_val ? ext_val : max_val) : min_val);
}

/// @brief Converts half-precision floating point value back to corresponding pixel channel
///        value (8-bit; no normalization).
///
/// @param val Input value.
/// @return    Pixel value with the same/nearest numerical value.
inline std::uint8_t convert_back_pixel_channel(const half_t val)
{
#if defined HALF_HALF_HPP
    return static_cast<std::uint8_t>(val);
#else
    return convert_back_pixel_channel<float>(static_cast<float>(val));
#endif
}

} // namespace detail

/// @brief Loads data from image file into clDNN allocation / memory buffer (store layout (B)YXF; FP16).
///
/// @details The loaded image is normalized before reading actual data:
///           -# Image is optionally resized (aspect ratio is kept) when the @p min_size is not @c 0.
///           -# Image is optionally normalized to rectangle image (aspect ratio is preserved).
///              The parts of images outside rectangle (scaled to fully contain either width or height)
///              are cut out.
///           -# If image is not 24-bit RGB, it is converted to 24-bit RGB.
///           .
///          If any of these steps failed, the image is not read and the @c false is returned.
///
/// @param image_file_path   Path (relative or absolute) to image file.
/// @param output_buffer_it  Iterator (output iterator, CopyConstructible) that points to
///                          buffer where data from image should be stored.
/// @param rgb_order         Indicates that color channels should be stored in features in RGB order.
///                          If @c true then RGB order is used; otherwise, BGR order is applied.
/// @param normalized_width  Width of image rectangle to which image will be normalized.
/// @n                       If @c 0 is specified, the image is not normalized to rectangle.
/// @param normalized_height Height of image rectangle to which image will be normalized.
/// @n                       If @c 0 is specified, the image is not normalized to rectangle.
/// @param min_size          Size of either width or height (which is lower) to which image
///                          should be resized (keeping aspect ratio) before normalization.
///                          If @c 0 is specified, the image is not resized.
/// @return                  @c true if image load was successful; otherwise, @c false.
template <typename OutIterTy>
auto load_image_data(const std::string& image_file_path, OutIterTy output_buffer_it,
                     const bool rgb_order,
                     const unsigned normalized_width, const unsigned normalized_height,
                     const unsigned min_size)
    -> std::enable_if_t<std::is_arithmetic<typename std::iterator_traits<OutIterTy>::value_type>::value ||
                        std::is_same<std::remove_cv_t<typename std::iterator_traits<OutIterTy>::value_type>,
                                     half_t>::value, bool>
{
    // ADL-friendly search for convert_pixel_channel.
    using detail::convert_pixel_channel;

    using out_elem_type = typename std::iterator_traits<OutIterTy>::value_type;


    std::shared_ptr<fipImage> normalized_image;
    try
    {
        const auto loaded_image = min_size != 0
                                      ? resize_keep_ar(load(image_file_path), min_size)
                                      : load(image_file_path);
        normalized_image = normalized_width != 0 && normalized_height != 0
                               ? crop_resize(loaded_image, normalized_width, normalized_height)
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

    const auto width  = normalized_image->getWidth();
    const auto height = normalized_image->getHeight();

    const auto bytes_per_pixel = normalized_image->getLine() / width;
    auto buffer_it = output_buffer_it;

    if (rgb_order)
    {
        for (unsigned y = 0; y < height; ++y)
        {
            auto image_pixel = reinterpret_cast<const std::uint8_t*>(normalized_image->getScanLine(height - y - 1));
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
            auto image_pixel = reinterpret_cast<const std::uint8_t*>(normalized_image->getScanLine(height - y - 1));
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

/// @brief Loads data from image file into clDNN allocation / memory buffer (store layout (B)YXF; FP16).
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
auto load_image_data(const std::string& image_file_path,
                     OutIterTy output_buffer_it,
                     const bool rgb_order             = false,
                     const unsigned normalized_size   = 0,
                     const unsigned min_size          = 0)
    -> std::enable_if_t<std::is_arithmetic<typename std::iterator_traits<OutIterTy>::value_type>::value ||
                        std::is_same<std::remove_cv_t<typename std::iterator_traits<OutIterTy>::value_type>,
                                     half_t>::value, bool>
{
    return load_image_data<OutIterTy>(image_file_path, output_buffer_it, rgb_order,
                                      normalized_size, normalized_size, min_size);
}

/// @brief Save data from clDNN allocation / memory buffer (store layout (B)YXF; FP16) into image file.
///
/// @param image_file_path Path (relative or absolute) to where image file should be saved.
/// @param input_buffer_it Iterator (input iterator, CopyConstructible) that points to
///                        buffer where image data should be read.
/// @param rgb_order       Indicates that color channels are stored in features in RGB order.
///                        If @c true then RGB order is used; otherwise, BGR order is assumed.
/// @param image_width     Width of image.
/// @param image_height    Height of image.
/// @param min_size        Size of either width or height (which is lower) to which image
///                        should be resized (keeping aspect ratio) before saving.
///                        If @c 0 is specified, the image is not resized.
/// @return                Saved image.
///
/// @exception std::invalid_argument Either @p width or @p height is non-positive.
/// @exception std::runtime_error    Resize of image failed.
/// @exception std::runtime_error    Save of image failed.
template <typename IterTy>
auto save_image_data(const std::string& image_file_path, IterTy input_buffer_it,
                     const bool rgb_order,
                     const unsigned image_width, const unsigned image_height,
                     const unsigned min_size = 0)
    -> std::enable_if_t<std::is_arithmetic<typename std::iterator_traits<IterTy>::value_type>::value ||
                        std::is_same<std::remove_cv_t<typename std::iterator_traits<IterTy>::value_type>,
                                     half_t>::value, std::shared_ptr<fipImage>>
{
    // ADL-friendly search for convert_pixel_channel.
    using detail::convert_back_pixel_channel;

    using in_elem_type = typename std::iterator_traits<IterTy>::value_type;


    if (image_width <= 0)
        throw std::invalid_argument("Parameter must be positive: \"image_width\"");
    if (image_height <= 0)
        throw std::invalid_argument("Parameter must be positive: \"image_height\"");

    const auto image = std::make_shared<fipImage>(FIT_BITMAP, image_width, image_height, 24);


    if (!image->isValid() || image->accessPixels() == nullptr)
        return nullptr;
    assert(image->getBitsPerPixel() == 24 && "Requested image BPP is different than expected (24).");

    const auto width  = image->getWidth();
    const auto height = image->getHeight();

    const auto bytes_per_pixel = image->getLine() / width;
    auto buffer_it = input_buffer_it;

    if (rgb_order)
    {
        for (unsigned y = 0; y < height; ++y)
        {
            auto image_pixel = reinterpret_cast<std::uint8_t*>(image->getScanLine(height - y - 1));
            assert(image_pixel != nullptr && "Inaccessible scanline in image.");

            for (unsigned x = 0; x < width; ++x)
            {
                image_pixel[FI_RGBA_RED]   = convert_back_pixel_channel(*buffer_it); ++buffer_it;
                image_pixel[FI_RGBA_GREEN] = convert_back_pixel_channel(*buffer_it); ++buffer_it;
                image_pixel[FI_RGBA_BLUE]  = convert_back_pixel_channel(*buffer_it); ++buffer_it;
                image_pixel += bytes_per_pixel;
            }
        }
    }
    else
    {
        for (unsigned y = 0; y < height; ++y)
        {
            auto image_pixel = reinterpret_cast<std::uint8_t*>(image->getScanLine(height - y - 1));
            assert(image_pixel != nullptr && "Inaccessible scanline in image.");

            for (unsigned x = 0; x < width; ++x)
            {
                image_pixel[FI_RGBA_BLUE]  = convert_back_pixel_channel(*buffer_it); ++buffer_it;
                image_pixel[FI_RGBA_GREEN] = convert_back_pixel_channel(*buffer_it); ++buffer_it;
                image_pixel[FI_RGBA_RED]   = convert_back_pixel_channel(*buffer_it); ++buffer_it;
                image_pixel += bytes_per_pixel;
            }
        }
    }

    // Final post-process and image saving.
    if (min_size != 0)
        resize_keep_ar(image, min_size);
    return save(image, image_file_path);
}


} // namespace image_toolkit

namespace itk = image_toolkit; // Short alias (to ease usage).

} // namespace examples
} // namespace utils
} // namespace cldnn
