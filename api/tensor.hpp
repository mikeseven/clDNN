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
#include "cldnn_defs.h"
#include "compounds.h"

#include <map>
#include <list>
#include <cstdint>
#include <numeric>
#include <algorithm>

namespace cldnn
{
/// @addtogroup cpp_api C++ API
/// @{

/// @addtogroup cpp_memory
/// @{

/// @brief Format information helper class.
struct format_traits
{
    /// @brief Number of batch dimensions in a format.
    size_t batch_num;
    /// @brief Number of feature map/channel dimensions in a format.
    size_t feature_num;
    /// @brief Number of spatial (x,y) dimensions in a format.
    size_t spatial_num;
    /// @brief Dimensions changing order from rare to often.
    std::string order;
    /// @brief Dimensions order for internal storage.
    std::string internal_order;
    /// @brief Characters representing batch dimensions in an order.
    static const char* batch_chars() { return "bn"; }
    /// @brief Characters representing feature map/channel dimensions in an order.
    static const char* feature_chars() { return "fioc"; }
    /// @brief Characters representing spatial dimensions in an order.
    static const char* spatial_chars() { return "xyzhsw"; }
    /// @brief Checks if @p c represents batch dimension.
    static bool is_batch_char(char c) { return std::string(batch_chars()).find_first_of(c) != std::string::npos; }
    /// @brief Checks if @p c represents feature map/channel dimension.
    static bool is_feature_char(char c) { return std::string(feature_chars()).find_first_of(c) != std::string::npos; }
    /// @brief Checks if @p c represents spatial dimension.
    static bool is_spatial_char(char c) { return std::string(spatial_chars()).find_first_of(c) != std::string::npos; }
};

/// @brief Represents memory formats (orders).
/// @n In CNN most of data is describe as 4 dimensional blocks. In Intel(R) clDNN library we describe memory with 4 letters
/// - b - number of blocks in batch
/// - f - number of feature maps, features or channels
/// - x - spatial, width
/// - y - spatial, height
/// /n
/// For explanation how each format type is implemented in memory we will use naming shown bellow:
/// \image html layout_memory_representation.jpg
struct format
{
    enum type : int32_t
    {
           x = cldnn_format_x,    ///< 1D.
          yx = cldnn_format_yx,   ///< 2D, X-axis then Y-axis: { x0y0, x1y0, x0y1, x1y1}.
          xy = cldnn_format_xy,   ///< 2D, Y-axis then X-axis: { x0y0, x0y1, x1y0, x1y1}.
          xb = cldnn_format_xb,   ///< 1D+batch.
          bx = cldnn_format_bx,   ///< 1D+batch.
        yxfn = cldnn_format_yxfn, ///< 3D + number of neurons. TO REMOVE
        yxfb = cldnn_format_yxfb, ///< batch first, feature and than spatials \n \image html yxfb.jpg
        byxf = cldnn_format_byxf, ///< used in bitmaps, input from user i.e b images of RGB format \n \image html byxf.jpg
        bfyx = cldnn_format_bfyx, ///< the most common format for activations in clDNN. \n \image html bfyx.jpg
        fyxb = cldnn_format_fyxb, ///< format not used inside clDNN, but supported in reorder as extension for user provided formats.
        oiyx = cldnn_format_oiyx, ///< format used only for weights: o - output feature maps, i - input feature maps. TO REMOVE
        yxoi = cldnn_format_yxoi, ///< format used only for weights: o - output feature maps, i - input feature maps. TO REMOVE
        oyxi = cldnn_format_oyxi, ///< format used only for weights: o - output feature maps, i - input feature maps. TO REMOVE
        yxio = cldnn_format_yxio, ///< format used only for weights: o - output feature maps, i - input feature maps. TO REMOVE
        os_iyx_osv16 = cldnn_format_os_iyx_osv16, ///< format used only for convolution weights: os - output feature maps slice, i - input feature maps, yx - spatials, sv16 - 16 values of single slice.
                                                  ///< \n \image html os_iyx_osv16.jpg
        bs_xs_xsv8_bsv8 = cldnn_format_bs_xs_xsv8_bsv8, ///< format used only for fully connected weights: bs - batch slice, xs - x slice, bsv8 - 8 values of single slice.
                                                        ///< \n \image html bs_xs_xsv8_bsv8.jpg
        bs_x_bsv16 = cldnn_format_bs_x_bsv16, ///< format used only for fully connected weights fp16 batch=1 : bs - batch slice (responses slice), bsv16 - 16 values of single batch slice, x - flattened plane of (fyx).
                                              ///< \n \image html bs_x_bsv16.jpg
        format_num = cldnn_format_format_num,
        any = cldnn_format_any,
    };

    /// @brief Get format traits for particular @p format::type
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

    /// @brief Returns number of batch dimensions for a @p format.
    static size_t batch_num(type fmt) { return traits(fmt).batch_num; }
    /// @brief Returns number of feature dimensions for a @p format.
    static size_t feature_num(type fmt) { return traits(fmt).feature_num; }
    /// @brief Returns number of spatial dimensions for a @p format.
    static size_t spatial_num(type fmt) { return traits(fmt).spatial_num; }
    /// @brief Returns an order of dimensions for a @ format.
    static const std::string& order(type fmt) { return traits(fmt).order; }
    /// @brief Returns an internal orders of dimensions for a @p format.
    static const std::string& internal_order(type fmt) { return traits(fmt).internal_order; }
    /// @brief Returns number of dimensions contained within a @p format
    static size_t dimension(type fmt) { return order(fmt).size(); }

    /// @brief Find common format.
    static type common_format(type t1, type t2)
    {
        auto merged_channels = order(t1);
        for (auto c : order(t2))
            if (merged_channels.find(c) == merged_channels.npos)
                merged_channels.push_back(c);

        std::list<type> formats;
        for (int fmt = x; fmt < format_num; ++fmt)
            if (order(static_cast<type>(fmt)).size() == merged_channels.size())
                formats.push_back(static_cast<type>(fmt));

        for (auto c : merged_channels)
        {
            auto itr = formats.begin();
            while (itr != formats.end())
            {
                if (order(*itr).find(c) == std::string::npos)
                    itr = formats.erase(itr);
                else
                    ++itr;
            }
        }

        if (formats.empty())
            throw std::domain_error("Could not find common format for formats: " + order(t1) + " and " + order(t2));

        return formats.front();
    }

    /// @brief Returns number of batch dimensions.
    size_t batch_num() const { return traits(value).batch_num; }
    /// @brief Returns number of feature dimensions.
    size_t feature_num() const { return traits(value).feature_num; }
    /// @brief Returns number of spatial dimensions.
    size_t spatial_num() const { return traits(value).spatial_num; }
    /// @brief Returns an order of dimensions in form of string.
    const std::string& order() const { return traits(value).order; }
    /// @brief Returns an internal orders of dimensions form of string.
    const std::string& internal_order() const { return traits(value).internal_order; }
    /// @brief Returns number of dimensions contained within this format
    size_t dimension() const { return order(value).size(); }

    type value;
    /// @brief Implicit conversion from format::type.
    constexpr format(type t) :value(t) {}
    /// @brief Implicit conversion to format::type.
    constexpr operator type() const { return value; }
    /// @brief Conversion from C API @ref ::cldnn_format_type.
    constexpr explicit format(cldnn_format_type t) : value(static_cast<type>(t)) {}
    /// @brief Conversion to C API @ref ::cldnn_format_type.
    constexpr explicit operator cldnn_format_type() const { return static_cast<cldnn_format_type>(value); }
};

/// @brief N-dimensional vector. Mostly used to represent memory size.
struct tensor
{
    typedef int32_t value_type;     ///< Values type stored in tensor.
    //TODO find the way to prevent direct change of following fields.
    cldnn::format format;
    array_ref<value_type> raw;      ///< Raw representation of all dimensions.
    array_ref<value_type> batch;    ///< Batch dimensions.
    array_ref<value_type> feature;  ///< Feature maps.
    array_ref<value_type> spatial;  ///< Spatial dimensions.

    /**
     * \brief Internal storage for tensor's data.
     * had to keep it public to support "Standard Layout"
     * please do not access this field directly
     */
    value_type _sizes[CLDNN_TENSOR_DIM_MAX];

    /// @brief Constructs @p tensor.
    /// @param[in] fmt Format (order).
    /// @param[in] default_size Default value for coordinates not reperesented in format.
    /// @param[in] sizes Dimensions in order defined in @p fmt.
    /// @details Example:
    /*! @code
     * 
       tensor my_tensor(format::yx, 10, { 2, 3 });   // y=2, x=3, b,f - not set
       cout << my_tensor.batch[0] << endl;           // 10 - default_size
       cout << my_tensor.feature[0] << endl;         // 10 - default_size
       cout << "x=" << my_tensor.spatial[0] << endl; // x=3
       cout << "y=" << my_tensor.spatial[1] << endl; // y=2
     *
     * @endcode
     */ 
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

    /// @brief Constructs @p tensor with default value 1.
    /// @param[in] fmt Format (order).
    /// @param[in] sizes Dimensions in order defined in @p fmt.
    /// @details Useful for @ref memory allocation.
    /// Example:
    /*! @code
    *
    tensor my_tensor(format::yx, { 2, 3 });
    cout << my_tensor.batch[0] << endl;           // 1
    cout << my_tensor.feature[0] << endl;         // 1
    cout << "x=" << my_tensor.spatial[0] << endl; // x=3
    cout << "y=" << my_tensor.spatial[1] << endl; // y=2
    *
    * @endcode
    */
    tensor(cldnn::format fmt, const std::vector<value_type>& sizes)
        :tensor(fmt, 1, sizes)
    {}

    /// @brief Constructs tensor with size 0.
    tensor() :tensor(format::x, 0, { 0 }) {}

    /// @brief Implicit conversion form C API :: cldnn_tensor.
    tensor(const cldnn_tensor& other)
        : format(static_cast<cldnn::format::type>(other.format))
        , raw(_sizes, format.batch_num() + format.feature_num() + format.spatial_num())
        , batch(_sizes, format.batch_num())
        , feature(_sizes + format.batch_num(), format.feature_num())
        , spatial(_sizes + format.batch_num() + format.feature_num(), format.spatial_num())
    {
        std::copy_n(other.sizes, CLDNN_TENSOR_DIM_MAX, _sizes);
    }

    /// @brief Implicit conversion to C API ::cldnn_tensor.
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

    /// @brief Copy construction.
    tensor(const tensor& other)
        : format(other.format)
        ,     raw(_sizes, format.batch_num() + format.feature_num() + format.spatial_num())
        ,   batch(_sizes, format.batch_num())
        , feature(_sizes + format.batch_num(), format.feature_num())
        , spatial(_sizes + format.batch_num() + format.feature_num(), format.spatial_num())
    {
        std::copy_n(other._sizes, CLDNN_TENSOR_DIM_MAX, _sizes);
    }

    /// @brief Copy assignment.
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

    /// @brief Returns a tensor with all negated elements.
    tensor negate() const
    {
        auto result = *this;
        for (size_t i = 0; i < CLDNN_TENSOR_DIM_MAX; i++)
        {
            result._sizes[i] = -_sizes[i];
        }
        return result;
    }

    /// @brief Returns a tensor with all elements multilied to @p multiplier.
    tensor mul(value_type multiplier) const
    {
        auto result = *this;
        for(size_t i = 0; i < result.raw.size(); i++ )
        {
            result._sizes[i] *= multiplier;
        }
        return result;
    }

    /// @brief Returns a tensor with all elements divided by @p divider.
    tensor div(value_type divider) const
    {
        auto result = *this;
        for (size_t i = 0; i < result.raw.size(); i++)
        {
            result._sizes[i] /= divider;
        }
        return result;
    }

    /// @brief Returns a tensor with all elements added by appropriate elements of @p rhs
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

    /// @brief Returns a tensor with all elements subtracted by appropriate elements of @p rhs
    tensor sub(const tensor& rhs) const
    {
        return add(rhs.negate());
    }

    /// @brief Returns a vector of tensors values, ordered regarding to @p format.
    std::vector<value_type> sizes() const {
        auto output_order = format.order();
        auto internal_order = format.internal_order();
        std::vector<value_type> sizes(output_order.size(), 0);

        for (size_t i = 0; i < sizes.size(); ++i)
        {
            auto c = output_order[i];
            auto pos = internal_order.find(c);
            if (pos == internal_order.npos)
                throw std::domain_error(std::string("Unknown coord type: ") + c);
            
            sizes[i] = _sizes[pos];
        }

        return sizes;
    }

    /// @brief Get aligned linear tensor size calculated as multiplication of all elements. 
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

    /// @brief Returns tensor elements count calculated as multiplication of all elements.
    size_t count() const { 
        return std::accumulate(
            raw.begin(),
            raw.end(), 
            static_cast<size_t>(1),
            std::multiplies<size_t>()
        );
    }

    /// @brief Returns new tensor based on current but transformed to new @p format.
    /// @param[in] new_fmt Format of new tensor.
    /// @param[in] default_size Default element values for positions not defined by current format.
    /// @details Example:
    /*!
     * @code
       tensor my_tensor(format::yx, { 2, 3 });
       auto my_sizes = my_tensor.sizes();
       cout << "dims_num=" << my_sizes.size() << endl; // dims_num=2
       cout << "x=" << my_sizes[1] << endl;            // x=3
       cout << "y=" << my_sizes[0] << endl;            // y=2
       auto new_tensor = my_tensor.transform(format::fyxb, 10);
       auto new_sizes = new_tensor.sizes();
       cout << "new_num=" << new_sizes.size() << endl;   // new_num=4
       for(auto dim : new_sizes) cout << " " << dim;     //  10 2 3 10
       cout << endl;
       * @endcode
     */
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

    /// @brief Calculates linear offset for given @p coord within current tensor.
    /// @param coord The coordinate within current tensor.
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

    /// @brief Returns a tensor containing values maximum from @p lhs and @p rhs.
    static tensor max(tensor const& lhs, tensor const& rhs)
    {
        auto comm_format = format::common_format(lhs.format, rhs.format);
        auto trans_lhs = lhs.transform(comm_format, std::numeric_limits<value_type>::min());
        auto trans_rhs = rhs.transform(comm_format, std::numeric_limits<value_type>::min());
        for (size_t i = 0; i < trans_lhs.raw.size(); ++i)
            trans_lhs._sizes[i] = std::max(trans_lhs.raw[i], trans_rhs.raw[i]);

        return trans_lhs;
    }
};

CLDNN_API_CLASS(tensor)

/// @brief Adds two @p tensors
inline tensor operator+(const tensor& lhs, const tensor& rhs) { return lhs.add(rhs); }
/// @brief Subtracts two @p tensors
inline tensor operator-(const tensor& lhs, const tensor& rhs) { return lhs.sub(rhs); }
/// @brief Multiplies a @p tensor to a @p scalar
inline tensor operator*(const tensor& lhs, tensor::value_type rhs) { return lhs.mul(rhs); }
/// @brief Divides a @p tensor by a @p scalar
inline tensor operator/(const tensor& lhs, tensor::value_type rhs) { return lhs.div(rhs); }

/// @}
/// @}
}
