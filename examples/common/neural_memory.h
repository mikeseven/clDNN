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


#include "api/CPP/memory.hpp"

#include <memory>
#include <stdexcept>


namespace cldnn
{
namespace backward_comp
{
// TODO: Refactor the code to incorporate it in file.h/file.cpp.
struct neural_memory
{
    /// @brief Type-code for layout of .nnd file (enum values wrapper). See format::type for details.
    struct nnd_layout_format
    {
        /// @brief Type-code for layout of .nnd file (use for old and new layout handling mode).
        ///
        /// @details In old mode of handling layout, the different layout code is used for different data type.
        ///          New layout handling mode in .nnd file uses only subset of @c _fp32-suffixed enum
        ///          values to encode / decode layout; data type is provided separately.
        enum type : uint8_t
        {
            // FP32 (single precision float)
            bias_x_f32,
            fc_xb_f32,             // 1D+batch, float32
            fc_bx_f32,             // 1D+batch, float32
            yxfb_f32 = 4,          // 3D+batch, float32
            byxf_f32,              // for convolution_cpu_jit_batch1
            bfyx_f32,              // used in Caffe
            fyxb_f32,              // used in Caffe
            weights_bfyx_f32,      // format used only for weights: b - output feature maps, f - input feature maps
            weights_yxfb_f32 = 11, // format used only for weights: b - output feature maps, f - input feature maps
            os_iyx_osv16_f32,      // format used only for weights: os - output feature maps slice, i - input feature maps, yx - spatials, sv16 - 16 values of single slice
            bs_xs_xsv8_bsv8_f32,   // format used only for Fully connected: bs - batch slice, xs - x slice, bsv8 - 8 values of single slice, xsv - 8 values of single slice 
            bs_x_bsv16_f32,        // format used only for fully connected: bs - batch slice (responses slice), bsv16 - 16 values of single batch slice, x - flattened plane of (fyx)
            bf8_xy16_f32,          // format used only for convolution 1x1 input, xy aligned to 16, f aligned to 8

            // FP16 (half precision float)
            bias_x_f16=19,
            fc_xb_f16,              // 1D+batch, FP16 (half precision float)
            fc_bx_f16,              // 1D+batch, FP16 (half precision float)
            yxfb_f16 = 23,          // 3D+batch, FP16 (half precision float)
            byxf_f16,               // for convolution_cpu_jit_batch1
            bfyx_f16,               // used in Caffe
            fyxb_f16,               // used in Caffe
            weights_bfyx_f16,       // format used only for weights: b - output feature maps, f - input feature maps
            weights_yxfb_f16 = 30,  // format used only for weights: b - output feature maps, f - input feature maps
            os_iyx_osv16_f16,       // format used only for weights: os - output feature maps slice, i - input feature maps, yx - spatials, sv16 - 16 values of single slice
            bs_xs_xsv8_bsv8_f16,    // format used only for Fully connected: bs - batch slice, xs - x slice, bsv8 - 8 values of single slice, xsv - 8 values of single slice
            bs_x_bsv16_f16,         // format used only for fully connected: bs - batch slice (responses slice), bsv16 - 16 values of single batch slice, x - flattened plane of (fyx)
            bf8_xy16_f16,           // format used only for convolution 1x1 input, xy aligned to 16, f aligned to 8

            any = static_cast<uint8_t>(-1),
            half_base = bias_x_f16  ///< Base offset in enum values for half-precision layouts (used mainly in old
                                    ///< layout handling mode).
        };


        /// @brief Indicates that specified (old) layout format type is supported / valid.
        ///
        /// @param format_type Format type to test.
        /// @return            @c true if format is valid / supported; otherwise, @c false.
        static bool is_supported(type format_type)
        {
            switch (static_cast<type>(format_type & half_base))
            {
            case bias_x_f32:
            case fc_xb_f32:
            case fc_bx_f32:
            case yxfb_f32:
            case byxf_f32:
            case bfyx_f32:
            case fyxb_f32:
            case weights_bfyx_f32:
            case weights_yxfb_f32:
            case os_iyx_osv16_f32:
            case bs_xs_xsv8_bsv8_f32:
            case bs_x_bsv16_f32:
            case bf8_xy16_f32:
                return true;

            default:
                return false;
            }
        }
    };


    /// @brief Basic type properties of clDNN data type.
    struct type_traits
    {
        /// @brief Creates properties information about specified clDNN data type.
        ///
        /// @param data_type Data type for which traits are created.
        type_traits(data_types data_type)
            : size(data_type_traits::size_of(data_type)),
              is_floating_point(data_type_traits::is_floating_point(data_type)) {}

        const size_t size;              ///< Size of data type in bytes.
        const bool   is_floating_point; ///< Indicates that data type is floating-point type.
    };

    /// @brief Basic properties of clDNN layout format.
    struct layout_traits
    {
        /// @brief Creates basic properties for specified clDNN layout format.
        ///
        /// @param layout Layout format for which properties will be generated.
        explicit layout_traits(const cldnn::layout& layout)
            : layout_traits(layout.format.order().length(), layout.data_type) {}

    protected:
        layout_traits(size_t dimension, cldnn::data_types data_type)
            : dimension(dimension), type(std::make_unique<const type_traits>(data_type)) {}

    public:
        const size_t                       dimension; ///< Number of dimensions in layout.
        std::unique_ptr<const type_traits> type;      ///< Data type used in layout.
    };


    /// @brief Get type-code of new .nnd file layout format based on clDNN layout format.
    ///
    /// @details      The format returned is non-shifted - as for new layout handling mechanism
    ///               where data type is encoded in .nnd separately.
    /// @param format clDNN layout format.
    /// @return       Type-code of new .nnd layout format.
    static uint8_t get_nnd_format_base(cldnn::format format)
    {
        switch (format.value)
        {
        case cldnn::format::yxfb:            return nnd_layout_format::yxfb_f32;
        case cldnn::format::byxf:            return nnd_layout_format::byxf_f32;
        case cldnn::format::bfyx:            return nnd_layout_format::bfyx_f32;
        case cldnn::format::fyxb:            return nnd_layout_format::fyxb_f32;
        case cldnn::format::os_iyx_osv16:    return nnd_layout_format::os_iyx_osv16_f32;
        case cldnn::format::bs_xs_xsv8_bsv8: return nnd_layout_format::bs_xs_xsv8_bsv8_f32;
        case cldnn::format::bs_x_bsv16:      return nnd_layout_format::bs_x_bsv16_f32;
        case cldnn::format::bf8_xy16:        return nnd_layout_format::bf8_xy16_f32;

        default: throw std::invalid_argument("clDNN layout format cannot be converted to .nnd "
                                             "layout format (unsuported in .nnd file).");
        }
    }


    /// @brief Converts clDNN layout to .nnd file layout format.
    ///
    /// @param layout   clDNN layout (format + data type).
    /// @param old_mode Indicates that old mode of handling .nnd layout format should be used.
    ///                 If @c true, it will emit old layout format (shifted properly based on data type
    ///                 used in .nnd).
    /// @return         Layout format for .nnd file.
    static nnd_layout_format::type to_nnd_format(const cldnn::layout& layout, bool old_mode = false)
    {
        switch (layout.format.value)
        {
        case cldnn::format::any: return nnd_layout_format::any;

        default: break;
        }

        if (!old_mode)
            return static_cast<nnd_layout_format::type>(get_nnd_format_base(layout.format));

        uint8_t format_shift = nnd_layout_format::half_base;
        switch (layout.data_type)
        {
        case cldnn::data_types::f32: format_shift *= 0; break;
        case cldnn::data_types::f16: format_shift *= 1; break;
        case cldnn::data_types::i8:  format_shift *= 2; break;
        case cldnn::data_types::u8:  format_shift *= 3; break;

        default: throw std::logic_error("Encountered unhandled clDNN data type.");
        }

        return static_cast<nnd_layout_format::type>(get_nnd_format_base(layout.format) + format_shift);
    }

    /// @brief Converts .nnd file layout format to clDNN layout format.
    ///
    /// @details Supports old and new .nnd layout modes.
    ///
    /// @param nnd_format Layout format of .nnd file.
    /// @return           Layout format for clDNN.
    static cldnn::format to_cldnn_format(nnd_layout_format::type nnd_format)
    {
        switch (nnd_format % nnd_layout_format::half_base)
        {
        case nnd_layout_format::bias_x_f32:          return cldnn::format::bfyx;
        case nnd_layout_format::fc_xb_f32:           return cldnn::format::yxfb;
        case nnd_layout_format::fc_bx_f32:           return cldnn::format::bfyx;
        case nnd_layout_format::yxfb_f32:            return cldnn::format::yxfb;
        case nnd_layout_format::byxf_f32:            return cldnn::format::byxf;
        case nnd_layout_format::bfyx_f32:            return cldnn::format::bfyx;
        case nnd_layout_format::fyxb_f32:            return cldnn::format::fyxb;
        case nnd_layout_format::weights_bfyx_f32:    return cldnn::format::bfyx;
        case nnd_layout_format::weights_yxfb_f32:    return cldnn::format::yxfb;
        case nnd_layout_format::os_iyx_osv16_f32:    return cldnn::format::os_iyx_osv16;
        case nnd_layout_format::bs_xs_xsv8_bsv8_f32: return cldnn::format::bs_xs_xsv8_bsv8;
        case nnd_layout_format::bs_x_bsv16_f32:      return cldnn::format::bs_x_bsv16;
        case nnd_layout_format::bf8_xy16_f32:        return cldnn::format::bf8_xy16;

        default: throw std::runtime_error("Layout format of .nnd file is not currently supported.");
        }
    }

    /// @brief Converts .nnd file layout format to clDNN data type (uses old layout mode).
    ///
    /// @details Supports only old .nnd layout mode.
    ///
    /// @param old_nnd_format Layout format of .nnd file (old form with encoded data type).
    /// @return               Data type for clDNN.
    static cldnn::data_types to_cldnn_data_type_old(nnd_layout_format::type old_nnd_format)
    {
        auto format_shift_m = old_nnd_format / nnd_layout_format::type::half_base;
        switch (format_shift_m)
        {
        case 0: return cldnn::data_types::f32;
        case 1: return cldnn::data_types::f16;
        case 2: return cldnn::data_types::i8;
        case 3: return cldnn::data_types::u8;

        default: throw std::logic_error("Data type calculated from .nnd file layout format is not supported.");
        }
    }


    /// @brief Basic properties of clDNN memory object.
    struct memory_traits
    {
        /// @brief Creates properties information about specified clDNN memory object.
        ///
        /// @param mem      Memory object for which traits are created.
        /// @param old_mode Indicates that old mode of handling .nnd layout format should be used.
        ///                 If @c true, it will return old layout format (shifted properly based on data type
        ///                 used in .nnd).
        explicit memory_traits(memory const& mem, bool old_mode = false)
            : memory_traits(mem.get_layout(), old_mode) {}

    protected:
        memory_traits(const layout& layout, bool old_mode = false)
            : format(to_nnd_format(layout, old_mode)), size(layout.size) {}

    public:
        nnd_layout_format::type format; ///< Expected layout format of .nnd file.
        const tensor&           size;   ///< Sizes of each dimension in memory object.
    };
};

} // namespace backward_comp
} // namespace cldnn
