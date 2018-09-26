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

#include <memory>
#include <stdexcept>
#include <string>


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
    /// @exception std::runtime_error Crop or resize of image failed.
    std::shared_ptr<fipImage> crop_resize_to_square(std::shared_ptr<fipImage> image, unsigned square_size = 0);

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
    /// @exception std::runtime_error Resize of image failed.
    std::shared_ptr<fipImage> resize_to_square(std::shared_ptr<fipImage> image, unsigned square_size = 0);

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
    /// @exception std::runtime_error Resize of image failed.
    std::shared_ptr<fipImage> resize_keep_ar(std::shared_ptr<fipImage> image, unsigned min_size);
} // namespace image_toolkit

namespace itk = image_toolkit; // Short alias (to ease usage).
