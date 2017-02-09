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

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include <cstdint>
#include <numeric>
#include "cldnn_defs.h"
#include "compounds.h"
#include <map>
#include <algorithm>

namespace cldnn
{
struct format_traits
{
    size_t batch_num;
    size_t feature_num;
    size_t spatial_num;
    std::string order;
    std::string internal_order;
    static const char* batch_chars() { return "bn"; }
    static const char* feature_chars() { return "fioc"; }
    static const char* spatial_chars() { return "xyzhsw"; }
    static bool is_batch_char(char c) { return std::string(batch_chars()).find_first_of(c) != std::string::npos; }
    static bool is_feature_char(char c) { return std::string(feature_chars()).find_first_of(c) != std::string::npos; }
    static bool is_spatial_char(char c) { return std::string(spatial_chars()).find_first_of(c) != std::string::npos; }

    static size_t has_fixed_position_within_group(char c)
    {
        return (c == 'x' || c == 'y' || c == 'i' || c == 'o');
    }

    static size_t get_position_within_group(char c)
    {
        //spatials
        if (c == 'x' || c == 'o')
            return 0;
        if (c == 'y' || c == 'i')
            return 1;

        return (size_t)-1;
    }
};

struct format
{
    enum type : int32_t
    {
           x = cldnn_format_x,
          yx = cldnn_format_yx,
          xy = cldnn_format_xy,
          xb = cldnn_format_xb,   // 1D+batch, float32
          bx = cldnn_format_bx,   // 1D+batch, float32
        yxfn = cldnn_format_yxfn, // 3D + number of neurons - used in fully connected weights
        yxfb = cldnn_format_yxfb, // 3D+batch, float32
        byxf = cldnn_format_byxf, // for convolution_cpu_jit_batch1
        bfyx = cldnn_format_bfyx, // used in Caffe
        fyxb = cldnn_format_fyxb, // used in Caffe
        oiyx = cldnn_format_oiyx, // format used only for weights: o - output feature maps, i - input feature maps
        yxoi = cldnn_format_yxoi, // format used only for weights: o - output feature maps, i - input feature maps
        oyxi = cldnn_format_oyxi, // format used only for weights: o - output feature maps, i - input feature maps
        yxio = cldnn_format_yxio, // format used only for weights: o - output feature maps, i - input feature maps
        os_iyx_osv16 = cldnn_format_os_iyx_osv16, // format used only for weights: os - output feature maps slice, i - input feature maps, yx - spatials, sv16 - 16 values of single slice
        bs_xs_xsv8_bsv8 = cldnn_format_bs_xs_xsv8_bsv8, // format used only for Fully connected: bs - batch slice, xs - x slice, xsv8 - 8 values of single slice, bsv8 - 8 values of single slice 
        bs_x_bsv16 = cldnn_format_bs_x_bsv16, // format used only for fully connected: bs - batch slice (responses slice), bsv16 - 16 values of single batch slice, x - flattened plane of (fyx)

        format_num = cldnn_format_format_num,
        any = cldnn_format_any,
    };

    static const format_traits& traits(type fmt)
    {
        static const std::map<type, format_traits> traits
        {
            { x,   { 1, 1, 1, "x", "??x" } },
            { yx,  { 1, 1, 2, "yx", "??xy" } },
            { xy,  { 1, 1, 2, "xy", "??xy" } },
            { xb,  { 1, 1, 1, "xb", "b?x" } },
            { bx,  { 1, 1, 1, "bx", "b?x" } },
            { yxfn,{ 1, 1, 2, "yxfn", "nfxy" } },
            { yxfb,{ 1, 1, 2, "yxfb", "bfxy" } },
            { byxf,{ 1, 1, 2, "byxf", "bfxy" } },
            { bfyx,{ 1, 1, 2, "bfyx", "bfxy" } },
            { fyxb,{ 1, 1, 2, "fyxb", "bfxy" } },
            { oiyx,{ 1, 2, 2, "oiyx", "?oixy" } },
            { yxoi,{ 1, 2, 2, "yxoi", "?oixy" } },
            { oyxi,{ 1, 2, 2, "oyxi", "?oixy" } },
            { yxio,{ 1, 2, 2, "yxio", "?oixy" } },
            { os_iyx_osv16, { 1, 2, 2, "oiyx", "?oixy" }},
            { bs_xs_xsv8_bsv8, { 1, 1, 1, "bx", "b?x" }},
            { bs_x_bsv16, { 1, 1, 1, "bx", "b?x" }}
        };
        return traits.at(fmt);
    }

    static size_t batch_num(type fmt) { return traits(fmt).batch_num; }
    static size_t feature_num(type fmt) { return traits(fmt).feature_num; }
    static size_t spatial_num(type fmt) { return traits(fmt).spatial_num; }
    static const std::string& order(type fmt) { return traits(fmt).order; }
    static const std::string& internal_order(type fmt) { return traits(fmt).internal_order; }

    size_t batch_num() const { return traits(value).batch_num; }
    size_t feature_num() const { return traits(value).feature_num; }
    size_t spatial_num() const { return traits(value).spatial_num; }
    const std::string& order() const { return traits(value).order; }
    const std::string& internal_order() const { return traits(value).internal_order; }

    type value;
    constexpr format(type t) :value(t) {}
    constexpr operator type() const { return value; }
    constexpr explicit format(cldnn_format_type t) : value(static_cast<type>(t)) {}
    constexpr explicit operator cldnn_format_type() const { return static_cast<cldnn_format_type>(value); }
};

/**
 * \brief 
 */
struct tensor
{
    typedef int32_t value_type;
    //TODO find the way to prevent direct change of following fields.
    cldnn::format format;
    array_ref<value_type> raw;
    array_ref<value_type> batch;
    array_ref<value_type> feature;
    array_ref<value_type> spatial;

    /**
     * \brief Internal storage for tensor's data.
     * had to keep it public to support "Standard Layout"
     * please do not access this field directly
     */
    value_type _sizes[CLDNN_TENSOR_DIM_MAX];

    tensor(cldnn::format fmt, value_type default_size, const std::vector<value_type>& sizes)
        : format(fmt)
        , raw(_sizes, fmt.batch_num() + fmt.feature_num() + fmt.spatial_num())
        , batch  (_sizes, fmt.batch_num())
        , feature(_sizes+ fmt.batch_num(), fmt.feature_num())
        , spatial(_sizes+ fmt.batch_num() + fmt.feature_num(), fmt.spatial_num())
    {
        auto input_order = fmt.order();
        auto internal_order = fmt.internal_order();
        std::fill_n(_sizes, CLDNN_TENSOR_DIM_MAX, default_size);

        if (sizes.size() != input_order.length())
            throw std::invalid_argument("number of sizes does not match format");

        for (size_t i = 0; i < input_order.size(); ++i)
        {
            auto c = input_order[i];
            auto pos = internal_order.find(c);
            if (pos == internal_order.npos)
                throw std::domain_error(std::string("Unknown coord type: ") + c);

            _sizes[pos] = sizes[i];
        }
    }

    tensor(cldnn::format fmt, const std::vector<value_type>& sizes)
        :tensor(fmt, 1, sizes)
    {}

    tensor() :tensor(format::x, 0, { 0 }) {}

    tensor(const cldnn_tensor& other)
        : format(static_cast<cldnn::format::type>(other.format))
        , raw(_sizes, format.batch_num() + format.feature_num() + format.spatial_num())
        , batch(_sizes, format.batch_num())
        , feature(_sizes + format.batch_num(), format.feature_num())
        , spatial(_sizes + format.batch_num() + format.feature_num(), format.spatial_num())
    {
        std::copy_n(other.sizes, CLDNN_TENSOR_DIM_MAX, _sizes);
    }

    operator cldnn_tensor() const
    {
        cldnn_tensor result;
        result.format = static_cast<cldnn_format_type>(format);
        result.batch_num = batch.size();
        result.feature_num = feature.size();
        result.spatial_num = spatial.size();
        std::copy_n(_sizes, CLDNN_TENSOR_DIM_MAX, result.sizes);
        return result;
    }

    tensor(const tensor& other)
        : format(other.format)
        ,     raw(_sizes, format.batch_num() + format.feature_num() + format.spatial_num())
        ,   batch(_sizes, format.batch_num())
        , feature(_sizes + format.batch_num(), format.feature_num())
        , spatial(_sizes + format.batch_num() + format.feature_num(), format.spatial_num())
    {
        std::copy_n(other._sizes, CLDNN_TENSOR_DIM_MAX, _sizes);
    }

    tensor& operator=(const tensor& other)
    {
        if (this == &other)
            return *this;
        format = other.format;
        raw     = { _sizes, format.batch_num() + format.feature_num() + format.spatial_num() };
        batch   = { _sizes, format.batch_num() };
        feature = { _sizes + format.batch_num(), format.feature_num() };
        spatial = { _sizes + format.batch_num() + format.feature_num(), format.spatial_num() };
        std::copy_n(other._sizes, CLDNN_TENSOR_DIM_MAX, _sizes);
        return *this;
    }

    friend bool operator==(const tensor& lhs, const tensor& rhs)
    {
        return lhs.format == rhs.format
            && lhs.raw.size() == rhs.raw.size()
            && std::equal(lhs.raw.begin(), lhs.raw.end(), rhs.raw.begin());
    }

    friend bool operator!=(const tensor& lhs, const tensor& rhs)
    {
        return !(lhs == rhs);
    }

    friend bool operator<(const tensor& lhs, const tensor& rhs)
    {
        if (lhs.format != rhs.format)
            return lhs.format < rhs.format;
        if (lhs.raw.size() != rhs.raw.size())
            return lhs.raw.size() < rhs.raw.size();
        for (size_t i = 0; i < lhs.raw.size(); ++i)
            if (lhs.raw[i] < rhs.raw[i])
                return true;

        return false;
    }

    tensor negate() const
    {
        auto result = *this;
        for (size_t i = 0; i < CLDNN_TENSOR_DIM_MAX; i++)
        {
            result._sizes[i] = -_sizes[i];
        }
        return result;
    }

    tensor mul(value_type multiplier) const
    {
        auto result = *this;
        for(size_t i = 0; i < result.raw.size(); i++ )
        {
            result._sizes[i] *= multiplier;
        }
        return result;
    }

    tensor div(value_type divider) const
    {
        auto result = *this;
        for (size_t i = 0; i < result.raw.size(); i++)
        {
            result._sizes[i] /= divider;
        }
        return result;
    }

    tensor add(const tensor& rhs) const
    {
        auto transformed_rhs = rhs.transform(format, 0);
        auto result = *this;
        for(size_t i = 0; i < result.raw.size(); i++)
        {
            result._sizes[i] += transformed_rhs._sizes[i];
        }
        return result;
    }

    tensor sub(const tensor& rhs) const
    {
        return add(rhs.negate());
    }

    std::vector<value_type> sizes() const {
        auto output_order = format.order();
        auto internal_order = format.internal_order();
        std::vector<value_type> sizes(output_order.size(), 0);

        for (size_t i = 0; i < sizes.size(); ++i)
        {
            auto c = output_order[i];
            auto pos = internal_order.find(c);
            if (pos == internal_order.npos)
                throw std::domain_error(std::string("Unknown coord type") + c);
            
            sizes[i] = _sizes[pos];
        }

        return sizes;
    }

    size_t get_linear_size() const
    {
        auto sizes = this->sizes();
        if(this->format == cldnn::format::os_iyx_osv16 && !is_aligned_to(sizes[0], 16))
        {
            sizes[0] = align_to(sizes[0], 16);
        }
        else if (this->format == cldnn::format::bs_xs_xsv8_bsv8 && !(is_aligned_to(sizes[0], 8) && is_aligned_to(sizes[1], 8)))
        {
            sizes[0] = align_to(sizes[0], 8);
            sizes[1] = align_to(sizes[1], 8);
        }
        else if(this->format == cldnn::format::bs_x_bsv16 && !is_aligned_to(sizes[0], 16))
        {
            sizes[0] = align_to(sizes[0], 16);
        }
        return std::accumulate(
            sizes.begin(),
            sizes.end(),
            static_cast<size_t>(1),
            std::multiplies<size_t>()
        );
    }

    tensor transform(cldnn::format new_fmt, value_type default_size) const
    {
        if (format == new_fmt) return *this;
        auto val_order = format.order();
        auto new_order = new_fmt.order();
        std::vector<value_type> old_sizes = sizes();
        std::vector<value_type> new_sizes(new_order.size(), default_size);
        for(size_t i = 0; i < old_sizes.size(); i++)
        {
            auto c = val_order[i];
            auto new_pos = new_order.find(c);
            if (new_pos == std::string::npos)
                throw std::invalid_argument("cannot convert to new format");
            new_sizes[new_pos] = old_sizes[i];
        }
        return{ new_fmt, default_size, new_sizes };
    }

    size_t get_linear_offset(const tensor& coord) const
    {
        auto my_sizes = sizes();
        auto adjusted_coords = coord.transform(format, 0).sizes();
        if (this->format == cldnn::format::os_iyx_osv16 && !is_aligned_to(my_sizes[0], 16))
        {
            my_sizes[0] = align_to(my_sizes[0], 16);
            adjusted_coords[0] = align_to(adjusted_coords[0], 16);
        }
        else if (this->format == cldnn::format::bs_xs_xsv8_bsv8 && !(is_aligned_to(my_sizes[0], 8) && is_aligned_to(my_sizes[1], 8)))
        {
            my_sizes[0] = align_to(my_sizes[0], 8);
            my_sizes[1] = align_to(my_sizes[1], 8);
            adjusted_coords[0] = align_to(adjusted_coords[0], 8);
            adjusted_coords[1] = align_to(adjusted_coords[1], 8);
        }
        else if (this->format == cldnn::format::bs_x_bsv16 && !is_aligned_to(my_sizes[0], 16))
        {
            my_sizes[0] = align_to(my_sizes[0], 16);
            adjusted_coords[0] = align_to(adjusted_coords[0], 16);
        }

        assert(my_sizes.size() == adjusted_coords.size());

        assert(adjusted_coords.size() > 0);
        size_t offset = adjusted_coords[0];
        for(size_t i = 1; i < adjusted_coords.size(); i++ )
        {
            offset = offset * my_sizes[i - 1] + adjusted_coords[i];
        }
        return offset;
    }
};

CLDNN_API_CLASS(tensor)

inline tensor operator+(const tensor& lhs, const tensor& rhs) { return lhs.add(rhs); }
inline tensor operator-(const tensor& lhs, const tensor& rhs) { return lhs.sub(rhs); }
inline tensor operator*(const tensor& lhs, tensor::value_type rhs) { return lhs.mul(rhs); }
inline tensor operator/(const tensor& lhs, tensor::value_type rhs) { return lhs.div(rhs); }
}