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
#include <unordered_map>

#define TENSOR_DIM_MAX 8

namespace cldnn
{
struct format
{
    // FP32 (single precision float)
    enum type {
        x,
        yx,
        xy,
        xb,          // 1D+batch, float32
        bx,          // 1D+batch, float32
        yxfn,          // 3D + number of neurons - used in fully connected weights
        yxfb,          // 3D+batch, float32
        byxf,          // for convolution_cpu_jit_batch1
        bfyx,          // used in Caffe
        fyxb,          // used in Caffe
        oiyx,          // format used only for weights: o - output feature maps, i - input feature maps
        yxoi,          // format used only for weights: o - output feature maps, i - input feature maps
        oyxi,          // format used only for weights: o - output feature maps, i - input feature maps
        yxio,          // format used only for weights: o - output feature maps, i - input feature maps
        format_num,
        any = static_cast<int8_t>(-1)
    };

    struct format_traits
    {
        size_t batch_num;
        size_t feature_num;
        size_t spatial_num;
        std::string order;
    };

    static const format_traits& traits(type fmt)
    {
        const std::unordered_map<type, format_traits> traits
        {
            { x,   { 1, 1, 1, "x" } },
            { yx,  { 1, 1, 2, "yx" } },
            { xy,  { 1, 1, 2, "xy" } },
            { xb,  { 1, 1, 1, "xb" } },
            { bx,  { 1, 1, 1, "bx" } },
            { yxfn,{ 1, 1, 2, "yxfn" } },
            { yxfb,{ 1, 1, 2, "yxfb" } },
            { byxf,{ 1, 1, 2, "byxf" } },
            { bfyx,{ 1, 1, 2, "bfyx" } },
            { fyxb,{ 1, 1, 2, "fyxb" } },
            { oiyx,{ 1, 2, 2, "oiyx" } },
            { yxoi,{ 1, 2, 2, "yxoi" } },
            { oyxi,{ 1, 2, 2, "oyxi" } },
            { yxio,{ 1, 2, 2, "yxio" } },
        };
        return traits.at(fmt);
    }

    static size_t batch_num(type fmt) { return traits(fmt).batch_num; }
    static size_t feature_num(type fmt) { return traits(fmt).feature_num; }
    static size_t spatial_num(type fmt) { return traits(fmt).spatial_num; }
    static const std::string& order(type fmt) { return traits(fmt).order; }

    type value;
    format(type t) :value(t) {}
    operator type() const { return value; }
    size_t batch_num() const { return traits(value).batch_num; }
    size_t feature_num() const { return traits(value).feature_num; }
    size_t spatial_num() const { return traits(value).spatial_num; }
    const std::string& order() const { return traits(value).order; }
};

/**
 * \brief 
 */
struct tensor
{
    //TODO find the way to prevent direct change of following fields.
    format format;
    array_ref<int32_t> raw;
    array_ref<int32_t> batch;
    array_ref<int32_t> feature;
    array_ref<int32_t> spatial;

    /**
     * \brief Internal storage for tensor's data.
     * had to keep it public to support "Standard Layout"
     * please do not access this field directly
     */
    int32_t _sizes[TENSOR_DIM_MAX];

    tensor(cldnn::format fmt, int32_t default_size, const std::vector<int32_t>& sizes)
        : format(fmt)
        , raw(_sizes, fmt.batch_num() + fmt.feature_num() + fmt.spatial_num())
        , batch  (_sizes, fmt.batch_num())
        , feature(_sizes+ fmt.batch_num(), fmt.feature_num())
        , spatial(_sizes+ fmt.batch_num() + fmt.feature_num(), fmt.spatial_num())
    {
        auto order = fmt.order();
        std::fill_n(_sizes, TENSOR_DIM_MAX, default_size);

        if (sizes.size() != order.length())
            throw std::invalid_argument("number of sizes does not match format");

        size_t batch_idx = 0;
        size_t feature_idx = 0;
        size_t spatial_idx = 0;
        for (size_t i = 0; i < sizes.size(); i++)
        {
            switch (order[i])
            {
            case 'b':
            case 'n':
                _sizes[batch_idx++] = sizes[i];
                break;
            case 'f':
            case 'i':
            case 'o':

                _sizes[batch.size() + (feature_idx++)] = sizes[i];
                break;
            case 's':
            case 'x':
            case 'y':
            case 'z':
                _sizes[batch.size() + feature.size() + (spatial_idx++)] = sizes[i];
                break;
            default:
                throw std::domain_error(std::string("unknown coord type: ") + order[i]);
            }
        }
    }

    tensor(cldnn::format fmt, const std::vector<int32_t>& sizes)
        :tensor(fmt, 1, sizes)
    {}

    tensor() :tensor(format::x, 0, { 0 }) {}

    tensor(const tensor& other)
        : format(other.format)
        ,     raw(_sizes, format.batch_num() + format.feature_num() + format.spatial_num())
        ,   batch(_sizes, format.batch_num())
        , feature(_sizes + format.batch_num(), format.feature_num())
        , spatial(_sizes + format.batch_num() + format.feature_num(), format.spatial_num())
    {
        std::copy_n(other._sizes, TENSOR_DIM_MAX, _sizes);
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
        std::copy_n(other._sizes, TENSOR_DIM_MAX, _sizes);
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

    std::vector<int32_t> sizes() const {
        auto order = format.order();
        std::vector<int32_t> sizes(order.size());
        size_t batch_idx = 0;
        size_t feature_idx = 0;
        size_t spatial_idx = 0;
        for (size_t i = 0; i < sizes.size(); i++)
        {
            switch (order[i])
            {
            case 'b':
            case 'n':
                sizes[i] = batch[batch_idx++];
                break;
            case 'f':
            case 'i':
            case 'o':
                sizes[i] = feature[feature_idx++];
                break;
            case 's':
            case 'x':
            case 'y':
            case 'z':
                sizes[i] = spatial[spatial_idx++];
                break;
            default:
                throw std::domain_error(std::string("unknown coord type: ") + order[i]);
            }
        }
        return sizes;
    }

    size_t get_linear_size() const
    {
        return std::accumulate(
            sizes().begin(),
            sizes().end(),
            static_cast<size_t>(1),
            std::multiplies<size_t>()
        );
    }

    tensor transform(cldnn::format new_fmt, int32_t default_size) const
    {
        if (format == new_fmt) return *this;
        auto val_order = format.order();
        auto new_order = new_fmt.order();
        std::vector<int32_t> old_sizes = sizes();
        std::vector<int32_t> new_sizes(new_order.size(), default_size);
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

API_CLASS(tensor)
}