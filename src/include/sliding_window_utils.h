// Copyright (c) 2017 Intel Corporation
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

#include <api/CPP/tensor.hpp>

#include <cassert>
#include <stdexcept>

#include "meta_utils.h"


namespace cldnn
{

/// @brief Sliding window output range computation mode.
enum class swor_mode
{
    all,         ///< Range is computed in the way that each sliding window in range is fully contained inside
                 ///< (non-padded) input data.
    exceed_once, ///< Range is computed in the way that each except at most one sliding window in range is fully
                 ///< contained inside (non-padded) input data. The last window may partially exceed (non-padded)
                 ///< input data range.
    any,         ///< Range is computed in the way that each sliding window in range is fully or at least partially
                 ///< contained inside (non-padded) input data.
};

/// @brief Calculates output range (size) for sliding window moving on input data range specified by @p input_size.
///
/// @param input_size Range/Size of input data (non-padded or treated as valid). Only spatial coordinates are
///                   considered.
/// @param size       Size of sliding window. Only spatial coordinates are considered.
/// @param offset     Offset/Padding of sliding window in input. Only spatial coordinates are considered. Padding/Offset
///                   is applied from both sides of input data: negative value extends/pads data, positive - crops it.
/// @param stride     Horizontal/Vertical stride of sliding in input data.
/// @param dilation   Horizontal/Vertical dilation of sliding window on input data.
/// @param sym_offset Treat offset as applied on input symmetrically (from both sides). If @c false, the @p offset
///                   is applied only from left/upper side.
/// @param degen_val  If values from calculation are in allowed range, but calculated output size is invalid,
///                   the @p degen_val is returned. Any non-positive value is considered degenerated and will be
///                   switched to value passed in this parameter.
/// @return Output range (size) of sliding window.
template <swor_mode RangeMode = swor_mode::all>
tensor calc_sliding_window_output_range(const tensor& input_size,
                                        const tensor& size,
                                        const tensor& offset,
                                        const tensor& stride,
                                        const tensor& dilation = {1, 1, 1, 1},
                                        bool sym_offset = true,
                                        const tensor::value_type& degen_val = 0);

/// @brief Fall-back implementation.
template <swor_mode RangeMode>
tensor calc_sliding_window_output_range(
    const tensor&, const tensor&, const tensor&, const tensor&, const tensor&, bool, const tensor::value_type&)
{
    static_assert(meta::always_false<meta::val_tuple<swor_mode, RangeMode>>::value,
                  "Sliding window output range calculation mode is not supported. Please implement specialization "
                  "for new swor_mode.");

    return tensor();
}

template <>
inline tensor calc_sliding_window_output_range<swor_mode::all>(
    const tensor& input_size, const tensor& size, const tensor& offset, const tensor& stride, const tensor& dilation,
    bool sym_offset, const tensor::value_type& degen_val)
{
    if(input_size.spatial[0] <= 0 || input_size.spatial[1] <= 0)
        throw std::invalid_argument("Input data spatial sizes must be positive (>= 1).");
    if(size.spatial[0] <= 0 || size.spatial[1] <= 0)
        throw std::invalid_argument("Sliding window spatial sizes must be positive (>= 1).");
    if(stride.spatial[0] <= 0 || stride.spatial[1] <= 0)
        throw std::invalid_argument("Sliding window h/v strides must be positive (>= 1).");
    if(dilation.spatial[0] <= 0 || dilation.spatial[1] <= 0)
        throw std::invalid_argument("Sliding window h/v input dialations must be positive (>= 1).");

    auto off_factor = sym_offset ? 2 : 1;
    tensor wnd_ext_size{1, 1, (size.spatial[0] - 1) * dilation.spatial[0] + 1,
                              (size.spatial[1] - 1) * dilation.spatial[1] + 1};

    auto output_range_x = static_cast<cldnn::tensor::value_type>(
        off_factor * offset.spatial[0] + wnd_ext_size.spatial[0] <= input_size.spatial[0]
            ? (input_size.spatial[0] - off_factor * offset.spatial[0] - wnd_ext_size.spatial[0]) / stride.spatial[0] + 1
            : degen_val);
    auto output_range_y = static_cast<cldnn::tensor::value_type>(
        off_factor * offset.spatial[1] + wnd_ext_size.spatial[1] <= input_size.spatial[1]
            ? (input_size.spatial[1] - off_factor * offset.spatial[1] - wnd_ext_size.spatial[1]) / stride.spatial[1] + 1
            : degen_val);

    return {1, 1, output_range_x, output_range_y};
}

template <>
inline tensor calc_sliding_window_output_range<swor_mode::exceed_once>(
    const tensor& input_size, const tensor& size, const tensor& offset, const tensor& stride, const tensor& dilation,
    bool sym_offset, const tensor::value_type& degen_val)
{
    if(input_size.spatial[0] <= 0 || input_size.spatial[1] <= 0)
        throw std::invalid_argument("Input data spatial sizes must be positive (>= 1).");
    if(size.spatial[0] <= 0 || size.spatial[1] <= 0)
        throw std::invalid_argument("Sliding window spatial sizes must be positive (>= 1).");
    if(stride.spatial[0] <= 0 || stride.spatial[1] <= 0)
        throw std::invalid_argument("Sliding window h/v strides must be positive (>= 1).");
    if(dilation.spatial[0] <= 0 || dilation.spatial[1] <= 0)
        throw std::invalid_argument("Sliding window h/v input dialations must be positive (>= 1).");

    auto off_factor = sym_offset ? 2 : 1;
    tensor wnd_ext_size{1, 1, (size.spatial[0] - 1) * dilation.spatial[0] + 1,
                              (size.spatial[1] - 1) * dilation.spatial[1] + 1};

    auto output_range_x = static_cast<cldnn::tensor::value_type>(
        off_factor * offset.spatial[0] + wnd_ext_size.spatial[0] <= input_size.spatial[0] + stride.spatial[0] - 1
            ? (input_size.spatial[0] - off_factor * offset.spatial[0] - wnd_ext_size.spatial[0] + stride.spatial[0] - 1) / stride.spatial[0] + 1
            : degen_val);
    auto output_range_y = static_cast<cldnn::tensor::value_type>(
        off_factor * offset.spatial[1] + wnd_ext_size.spatial[1] <= input_size.spatial[1] + stride.spatial[1] - 1
            ? (input_size.spatial[1] - off_factor * offset.spatial[1] - wnd_ext_size.spatial[1] + stride.spatial[1] - 1) / stride.spatial[1] + 1
            : degen_val);

    return {1, 1, output_range_x, output_range_y};
}

template <>
inline tensor calc_sliding_window_output_range<swor_mode::any>(
    const tensor& input_size, const tensor& size, const tensor& offset, const tensor& stride, const tensor& dilation,
    bool sym_offset, const tensor::value_type& degen_val)
{
    if(input_size.spatial[0] <= 0 || input_size.spatial[1] <= 0)
        throw std::invalid_argument("Input data spatial sizes must be positive (>= 1).");
    if(size.spatial[0] <= 0 || size.spatial[1] <= 0)
        throw std::invalid_argument("Sliding window spatial sizes must be positive (>= 1).");
    if(stride.spatial[0] <= 0 || stride.spatial[1] <= 0)
        throw std::invalid_argument("Sliding window h/v strides must be positive (>= 1).");
    if(dilation.spatial[0] <= 0 || dilation.spatial[1] <= 0)
        throw std::invalid_argument("Sliding window h/v input dialations must be positive (>= 1).");

    auto off_factor = sym_offset ? 2 : 1;

    auto output_range_x = static_cast<cldnn::tensor::value_type>(
        off_factor * offset.spatial[0] <= input_size.spatial[0] - 1
            ? (input_size.spatial[0] - off_factor * offset.spatial[0] - 1) / stride.spatial[0] + 1
            : degen_val);
    auto output_range_y = static_cast<cldnn::tensor::value_type>(
        off_factor * offset.spatial[1] <= input_size.spatial[1] - 1
            ? (input_size.spatial[1] - off_factor * offset.spatial[1] - 1) / stride.spatial[1] + 1
            : degen_val);

    return {1, 1, output_range_x, output_range_y};
}


/// @brief Calculates minumum needed input range (size) for sliding window to get at least specified @p output_size.
///
/// Currently @see calc_sliding_window_output_range for @see swor_mode::exceed_once calculates smallest @p output_size,
/// so input size will be computed assuming this model.
///
/// @param output_size Range/Size of output data (non-padded or treated as valid). Only spatial coordinates are
///                    considered.
/// @param size        Size of sliding window. Only spatial coordinates are considered.
/// @param offset      Offset/Padding of sliding window in input. Only spatial coordinates are considered. Padding/Offset
///                    is applied from both sides of input data: negative value extends/pads data, positive - crops it.
/// @param stride      Horizontal/Vertical stride of sliding in input data.
/// @param dilation    Horizontal/Vertical dilation of sliding window on input data.
/// @param sym_offset  Treat offset as applied on input symmetrically (from both sides). If @c false, the @p offset
///                    is applied only from left/upper side.
/// @param degen_val   If values from calculation are in allowed range, but calculated output size is invalid,
///                    the @p degen_val is returned. Any non-positive value is considered degenerated and will be
///                    switched to value passed in this parameter.
/// @return Input range (size) for sliding window to get equal or greater @p output_size.
inline tensor calc_sliding_window_needed_input_range(const tensor& output_size,
                                                     const tensor& size,
                                                     const tensor& offset,
                                                     const tensor& stride,
                                                     const tensor& dilation = {1, 1, 1, 1},
                                                     bool sym_offset = true,
                                                     const tensor::value_type& degen_val = 0)
{
    if(output_size.spatial[0] <= 0 || output_size.spatial[1] <= 0)
        throw std::invalid_argument("Output data spatial sizes must be positive (>= 1).");
    if(size.spatial[0] <= 0 || size.spatial[1] <= 0)
        throw std::invalid_argument("Sliding window spatial sizes must be positive (>= 1).");
    if(stride.spatial[0] <= 0 || stride.spatial[1] <= 0)
        throw std::invalid_argument("Sliding window h/v strides must be positive (>= 1).");
    if(dilation.spatial[0] <= 0 || dilation.spatial[1] <= 0)
        throw std::invalid_argument("Sliding window h/v input dialations must be positive (>= 1).");

    auto off_factor = sym_offset ? 2 : 1;
    tensor wnd_ext_size{1, 1, (size.spatial[0] - 1) * dilation.spatial[0] + 1,
                              (size.spatial[1] - 1) * dilation.spatial[1] + 1};

    auto output_range_x = off_factor * offset.spatial[0] + (output_size.spatial[0] - 1) * stride.spatial[0] + wnd_ext_size.spatial[0];
    auto output_range_y = off_factor * offset.spatial[1] + (output_size.spatial[1] - 1) * stride.spatial[1] + wnd_ext_size.spatial[1];

    if (output_range_x <= 0)
        output_range_x = degen_val;
    if (output_range_y <= 0)
        output_range_y = degen_val;

    return {1, 1, output_range_x, output_range_y};
}

} //namespace cldnn
