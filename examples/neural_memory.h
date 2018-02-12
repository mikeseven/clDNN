
#pragma once

#include "api/CPP/memory.hpp"

namespace cldnn
{
namespace backward_comp
{

// TODO remove this backward compatibility class
// there are no longer dependencies inside clDNN core on this class
// file.cpp remains
struct neural_memory
{
    struct format
    {
        enum type : uint8_t
        {
            // FP32 (single precision float)
            bias_x_f32,
            fc_xb_f32,     // 1D+batch, float32
            fc_bx_f32,     // 1D+batch, float32
            yxfb_f32 = 4,   // 3D+batch, float32
            byxf_f32,   // for convolution_cpu_jit_batch1
            bfyx_f32,   // used in Caffe
            fyxb_f32,   // used in Caffe
            weights_bfyx_f32,   // format used only for weights: b - output feature maps, f - input feature maps
            weights_yxfb_f32=11,   // format used only for weights: b - output feature maps, f - input feature maps
            os_iyx_osv16_f32, // format used only for weights: os - output feature maps slice, i - input feature maps, yx - spatials, sv16 - 16 values of single slice
            bs_xs_xsv8_bsv8_f32, // format used only for Fully connected: bs - batch slice, xs - x slice, bsv8 - 8 values of single slice, xsv - 8 values of single slice 
            bs_x_bsv16_f32,      // format used only for fully connected: bs - batch slice (responses slice), bsv16 - 16 values of single batch slice, x - flattened plane of (fyx)
            bf8_xy16_f32,        // format used only for convolution 1x1 input, xy aligned to 16, f aligned to 8

            // FP16 (half precision float)
            bias_x_f16=19,
            fc_xb_f16,            // 1D+batch, FP16 (half precision float)
            fc_bx_f16,            // 1D+batch, FP16 (half precision float)
            yxfb_f16=23,          // 3D+batch, FP16 (half precision float)
            byxf_f16,          // for convolution_cpu_jit_batch1
            bfyx_f16,          // used in Caffe
            fyxb_f16,          // used in Caffe
            weights_bfyx_f16,          // format used only for weights: b - output feature maps, f - input feature maps
            weights_yxfb_f16=30,          // format used only for weights: b - output feature maps, f - input feature maps
            os_iyx_osv16_f16,  // format used only for weights: os - output feature maps slice, i - input feature maps, yx - spatials, sv16 - 16 values of single slice
            bs_xs_xsv8_bsv8_f16, // format used only for Fully connected: bs - batch slice, xs - x slice, bsv8 - 8 values of single slice, xsv - 8 values of single slice
            bs_x_bsv16_f16,    // format used only for fully connected: bs - batch slice (responses slice), bsv16 - 16 values of single batch slice, x - flattened plane of (fyx)
            bf8_xy16_f16,      // format used only for convolution 1x1 input, xy aligned to 16, f aligned to 8

            format_num,
            any = static_cast<uint8_t>(-1),
            half_base = bias_x_f16
        };
    };

    struct type_traits {
        type_traits(cldnn::data_types data_type)
            : size(cldnn::data_type_traits::size_of(data_type))
            , is_floating_point(cldnn::data_type_traits::is_floating_point(data_type))
        {}
        const size_t          size;
        const bool            is_floating_point;
    };

    struct format_traits
    {
        format_traits(size_t dimension, cldnn::data_types data_type)
            : dimension(dimension)
            , type(new type_traits(data_type))
        {
        }
        const size_t       dimension;
        std::unique_ptr<type_traits>  type;
    };

    static format_traits traits(const cldnn::layout& layout)
    {
        return format_traits(layout.format.order().length(), layout.data_type);
    }

    static uint8_t get_format_base(cldnn::format format)
    {
        switch (format.value)
        {
        case cldnn::format::yxfb: return format::type::yxfb_f32;
        case cldnn::format::byxf: return format::type::byxf_f32;
        case cldnn::format::bfyx: return format::type::bfyx_f32;
        case cldnn::format::fyxb: return format::type::fyxb_f32;
        case cldnn::format::os_iyx_osv16: return format::type::os_iyx_osv16_f32;
        case cldnn::format::bs_xs_xsv8_bsv8: return format::type::bs_xs_xsv8_bsv8_f32;
        case cldnn::format::bs_x_bsv16: return format::type::bs_x_bsv16_f32;
        case cldnn::format::bf8_xy16: return format::type::bf8_xy16_f32;
        default: throw std::invalid_argument("unsupported format");
        }
    }

    static neural_memory::format::type convert_format(const cldnn::layout& layout)
    {
        switch (layout.format.value)
        {
        case cldnn::format::format_num: return neural_memory::format::type::format_num;
        case cldnn::format::any: return neural_memory::format::type::any;
        default: break;
        }

        uint8_t format_shift;
        switch (layout.data_type)
        {
        case cldnn::data_types::f32:
            format_shift = 0;
            break;
        case cldnn::data_types::f16:
            format_shift = neural_memory::format::type::half_base;
            break;
        default: throw std::invalid_argument("unsupported data type");
        }
        return static_cast<neural_memory::format::type>(get_format_base(layout.format) + format_shift);
    }

    static cldnn::format to_tensor_format(format::type value)
    {
        switch (value % format::type::half_base)
        {
        case format::type::bias_x_f32: return cldnn::format::bfyx;
        case format::type::fc_xb_f32: return cldnn::format::yxfb;
        case format::type::fc_bx_f32: return cldnn::format::bfyx;
        case format::type::yxfb_f32: return cldnn::format::yxfb;
        case format::type::byxf_f32: return cldnn::format::byxf;
        case format::type::bfyx_f32: return cldnn::format::bfyx;
        case format::type::fyxb_f32: return cldnn::format::fyxb;
        case format::type::weights_bfyx_f32: return cldnn::format::bfyx;
        case format::type::weights_yxfb_f32: return cldnn::format::yxfb;
        case format::type::os_iyx_osv16_f32: return cldnn::format::os_iyx_osv16;
        case format::type::bs_xs_xsv8_bsv8_f32: return cldnn::format::bs_xs_xsv8_bsv8;
        case format::type::bs_x_bsv16_f32: return cldnn::format::bs_x_bsv16;
        case format::type::bf8_xy16_f32: return cldnn::format::bf8_xy16;
        default: throw std::invalid_argument("unsupported format");
        }
    }

    static cldnn::data_types to_data_type(format::type value)
    {
        return value < format::type::half_base ? cldnn::data_types::f32 : cldnn::data_types::f16;
    }

    struct arguments {
        neural_memory::format::type    format;
        const cldnn::tensor&   size;
        arguments(const cldnn::layout& layout) : format(convert_format(layout)), size(layout.size) {}
    };
};

inline neural_memory::arguments argument(cldnn::memory const& mem) { return neural_memory::arguments(mem.get_layout()); }

} //end namespace backward_comp
} //end namespace cldnn