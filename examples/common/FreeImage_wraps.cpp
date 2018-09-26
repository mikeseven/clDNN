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

#include "FreeImage_wraps.h"

#include <algorithm>
#include <cmath>


/// @brief Missing function from fipImage: Rescales rectangular part of current image.
///
/// @param image      [in, out] Current image.
/// @param dst_width  Target width of image.
/// @param dst_height Target height of image.
/// @param left       X coordinate of top-left point of crop rectangle.
/// @param top        Y coordinate of top-left point of crop rectangle.
/// @param right      X coordinate of top-left point of crop rectangle.
/// @param bottom     Y coordinate of top-left point of crop rectangle.
/// @param filter     Rescaling filter.
/// @return           Indicates that operation succeeded.
// ReSharper disable once CppInconsistentNaming
static bool fipImage_rescaleRect(fipImage& image, const unsigned dst_width, const unsigned dst_height,
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


std::shared_ptr<fipImage> image_toolkit::load(const std::string& image_file_path)
{
    auto image = std::make_shared<fipImage>();

    const auto load_success_flag = image->load(image_file_path.c_str());
    if (!load_success_flag)
        throw std::runtime_error("Failed to load image (image library reported failure): \"" + image_file_path + "\"");
    return image;
}

std::shared_ptr<fipImage> image_toolkit::crop_resize_to_square(std::shared_ptr<fipImage> image,
                                                               const unsigned square_size)
{
    struct crop_rect
    {
        unsigned left, top, right, bottom;
    };


    const auto width  = image->getWidth();
    const auto height = image->getHeight();

    if (width == height)
        return resize_to_square(image, square_size);

    crop_rect square_rect;  // NOLINT(cppcoreguidelines-pro-type-member-init, hicpp-member-init)
    if (width > height)
    {
        const auto cut_size      = width - height;
        const auto cut_margin    = cut_size / 2;
        const auto cut_remainder = cut_size - 2 * cut_margin;

        square_rect = {cut_margin, 0, width - cut_margin - cut_remainder, height};
    }
    else
    {
        const auto cut_size      = height - width;
        const auto cut_margin    = cut_size / 2;
        const auto cut_remainder = cut_size - 2 * cut_margin;

        square_rect = {0, cut_margin, width, height - cut_margin - cut_remainder};
    }

    const auto rescale_success = fipImage_rescaleRect(*image,
                                                      square_size, square_size,
                                                      square_rect.left,
                                                      square_rect.top,
                                                      square_rect.right,
                                                      square_rect.bottom,
                                                      FILTER_CATMULLROM);
    if (!rescale_success)
        throw std::runtime_error("Failed to rescale square part of image (image library reported failure)");

    return image; // NRVO
}

std::shared_ptr<fipImage> image_toolkit::resize_to_square(std::shared_ptr<fipImage> image, const unsigned square_size)
{
    const auto width  = image->getWidth();
    const auto height = image->getHeight();

    const auto out_square_size = square_size > 0 ? square_size : std::max(width, height);
    if (out_square_size == width && out_square_size == height)
        return image; // NRVO

    const auto rescale_success = image->rescale(out_square_size, out_square_size, FILTER_CATMULLROM);
    if (!rescale_success)
        throw std::runtime_error("Failed to rescale input image (image library reported failure)");

    return image; // NRVO
}

std::shared_ptr<fipImage> image_toolkit::resize_keep_ar(std::shared_ptr<fipImage> image, const unsigned min_size)
{
    struct spatial_point
    {
        unsigned width;
        unsigned height;
    };


    const spatial_point image_dims = {image->getWidth(), image->getHeight()};

    const auto aspect_ratio = static_cast<const float>(image_dims.width) / image_dims.height;

    spatial_point dst_image_dims;  // NOLINT(cppcoreguidelines-pro-type-member-init, hicpp-member-init)
    if (image_dims.width <= image_dims.height)
        dst_image_dims = {min_size, static_cast<unsigned>(std::round(min_size / aspect_ratio))};
    else
        dst_image_dims = {static_cast<unsigned>(std::round(min_size * aspect_ratio)), min_size};

    const auto rescale_success = image->rescale(dst_image_dims.width, dst_image_dims.height, FILTER_CATMULLROM);
    if (!rescale_success)
        throw std::runtime_error("Failed to rescale input image (image library reported failure)");

    return image; // NRVO
}
