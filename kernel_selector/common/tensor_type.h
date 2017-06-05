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

#pragma once

#include "common_types.h"
#include <map>
#include <vector>
#include <assert.h>
#include <numeric>
#include <algorithm>

namespace KernelSelector
{
    namespace Tensor
    {
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // DataLayout
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        enum DataLayout
        {
            bf,                 // 1D+batch
            fb,                 // 1D+batch
            bfyx,               // 3D+batch
            yxfb,               // 3D+batch
            byxf,               // 3D+batch
            fyxb,               // 3D+batch

            // TODO: most of the kernel doesn't support ROI. we need to handle it correctly.
            brfyx,              // 4D+batch
        };

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // WeightsLayout
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        enum WeightsLayout
        {
            oi,
            io,
            oiyx,
            oyxi,
            iyxo,
            yxio,
            os_iyx_osv16,
            os_is_isv8_osv8,
            os_i_osv16,
            iyxo_om16x2_axy,
            iyxo_om16x2_ax_g32,
            iyxo_om8x2_ax_g32,
        };

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Dim
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        struct Dim
        {
            size_t v;
            size_t pitch;
        };

        using NDims = std::vector<Dim>;

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // extract code
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        namespace
        {
            enum class DataChannelName
            {
                NAME_X       = 0,
                NAME_Y       = 1,
                NAME_FEATURE = 2,
                NAME_ROI     = 3,
                NAME_BATCH   = 4,
            };

            enum class WeightsChannelName
            {
                NAME_X   = 0,
                NAME_Y   = 1,
                NAME_IFM = 2,
                NAME_OFM = 3,
            };

            using ChannelLocation = std::vector<int>;

            std::map<DataLayout, ChannelLocation> dataChannelMap
            {
                { DataLayout::bf,   { -1,-1, 0,-1, 1 }},
                { DataLayout::fb,   { -1,-1, 1,-1, 0 }},
                { DataLayout::bfyx, {  0, 1, 2,-1, 3 }},
                { DataLayout::yxfb, {  2, 3, 1,-1, 0 }},
                { DataLayout::byxf, {  1, 2, 0,-1, 3 }},
                { DataLayout::fyxb, {  1, 2, 3,-1, 0 }},
                { DataLayout::brfyx,{  0, 1, 2, 3, 4 }},
            };

            std::map<WeightsLayout, ChannelLocation> weightsChannelMap
            {
                { WeightsLayout::oi,                    { -1,-1, 0, 1 } },
                { WeightsLayout::io,                    { -1,-1, 1, 0 } },
                { WeightsLayout::oiyx,                  {  0, 1, 2, 3 } },
                { WeightsLayout::oyxi,                  {  1, 2, 0, 3 } },
                { WeightsLayout::iyxo,                  {  1, 2, 3, 0 } },
                { WeightsLayout::yxio,                  {  2, 3, 1, 0 } },
                { WeightsLayout::os_iyx_osv16,          {  0, 1, 2, 3 } },
                { WeightsLayout::os_is_isv8_osv8,       { -1,-1, 0, 1 } },
                { WeightsLayout::os_i_osv16,            { -1,-1, 0, 1 } },
                { WeightsLayout::iyxo_om16x2_axy,       {  1, 2, 3, 0 } },
                { WeightsLayout::iyxo_om16x2_ax_g32,    {  1, 2, 3, 0 } },
                { WeightsLayout::iyxo_om8x2_ax_g32,     {  1, 2, 3, 0 } },
            };

            template <typename MapT, typename Layout, typename ChannelName>
            int channelndex(MapT channelMap, Layout l, ChannelName channelName)
            {
                size_t channel = static_cast<size_t>(channelName);
                assert(channelMap.find(l) != channelMap.end());
                assert(channel < channelMap[l].size());

                return channelMap[l][channel];
            }

            template <typename MapT, typename Layout, typename ChannelName>
            Dim extract(MapT channelMap, Layout l, ChannelName channelName, const NDims& dims)
            {
                const int i = channelndex(channelMap, l, channelName);
                return ((i < 0) || (i >= (int)dims.size())) ? Dim{1, 1} : dims[i];
            }

            template <typename MapT, typename Layout, typename ChannelName>
            void set(MapT channelMap, Layout l, ChannelName channelName, NDims& dims, const Dim d)
            {
                const int i = channelndex(channelMap, l, channelName);
                if ((i >= 0) && (i < (int)dims.size()))
                {
                    dims[i] = d;
                }
            }

            template <typename MapT, typename Layout>
            uint32_t channelsCount(MapT channelMap, Layout l)
            {
                const auto& entry = channelMap[l];
                return std::accumulate(entry.begin(), entry.end(), 0U, [](uint32_t count, int v) {return count + ((v != -1) ? 1 : 0); });
            }

            Dim extract(DataLayout l, DataChannelName channel, const NDims& d)
            {
                return extract(dataChannelMap, l, channel, d);
            }

            Dim extract(WeightsLayout l, WeightsChannelName channel, const NDims& d)
            {
                return extract(weightsChannelMap, l, channel, d);
            }

            void set(DataLayout l, DataChannelName channel, NDims& dims, const Dim d)
            {
                set(dataChannelMap, l, channel, dims, d);
            }

            void set(WeightsLayout l, WeightsChannelName channel, NDims& dims, const Dim d)
            {
                set(weightsChannelMap, l, channel, dims, d);
            }

            int channelndex(DataLayout l, DataChannelName channel)
            {
                return channelndex(dataChannelMap, l, channel);
            }

            int channelndex(WeightsLayout l, WeightsChannelName channel)
            {
                return channelndex(weightsChannelMap, l, channel);
            }

            uint32_t channelsCount(DataLayout l)
            {
                return channelsCount(dataChannelMap, l);
            }

            uint32_t channelsCount(WeightsLayout l)
            {
                return channelsCount(weightsChannelMap, l);
            }
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // PADDED_VAL
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        enum class PADDED_VAL
        {
            UNDEFINED,
            ZERO,
            ONE,
            HIGHEST_VAL,
            LOWEST_VAL,
        };

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Tensor
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        struct DataTensor
        {
            Datatype            dtype = Datatype::F16;
            DataLayout          layout = DataLayout::bfyx;
            PADDED_VAL          paddedVal = PADDED_VAL::UNDEFINED;
            size_t              offset = 0;
            NDims               dims;

            DataTensor() = default;
            DataTensor(Datatype dt, DataLayout l, PADDED_VAL pv = PADDED_VAL::UNDEFINED, size_t of = 0, const NDims& nd = {}) :
                dtype(dt), layout(l), paddedVal(pv), offset(of), dims(nd) {}
            DataTensor(Datatype dt, DataLayout l, PADDED_VAL pv, size_t of, const std::vector<size_t>& d) :
                dtype(dt), layout(l), paddedVal(pv), offset(of), dims(CalcPitches(d)) {}
            DataTensor(const DataTensor&) = default;
            DataTensor& operator=(const DataTensor&) = default;

            NDims CalcPitches(const std::vector<size_t>& d) const
            {
                NDims ret(d.size());
                size_t pitch = 1;

                for (size_t i = 0; i < d.size(); i++)
                {
                    ret[i] = { d[i], pitch };
                    pitch *= d[i];
                }

                return ret;
            }

            Dim x() const { return extract(layout, DataChannelName::NAME_X, dims); }
            Dim y() const { return extract(layout, DataChannelName::NAME_Y, dims); }
            Dim feature() const { return extract(layout, DataChannelName::NAME_FEATURE, dims); }
            Dim roi() const { return extract(layout, DataChannelName::NAME_ROI, dims); }
            Dim batch() const { return extract(layout, DataChannelName::NAME_BATCH, dims); }

            void SetX(const Dim d) { set(layout, DataChannelName::NAME_X, dims, d); }
            void SetY(const Dim d) { set(layout, DataChannelName::NAME_Y, dims, d); }
            void SetFeature(const Dim d) { set(layout, DataChannelName::NAME_FEATURE, dims, d); }
            void SetRoi(const Dim d) { set(layout, DataChannelName::NAME_ROI, dims, d); }
            void SetBatch(const Dim d) { set(layout, DataChannelName::NAME_BATCH, dims, d); }

            size_t Length() const
            {
                return std::accumulate(dims.cbegin(), dims.cend(), (size_t)1, [](size_t val, const Dim& d) {return val*d.v; });
            }

            size_t LengthWithPadding() const
            {
                return std::accumulate(dims.cbegin(), dims.cend(), (size_t)1, [](size_t val, const Dim& d) {return std::max(val, d.pitch*d.v); });
            }

            size_t PhysicalSize() const
            {
                return (offset + LengthWithPadding()) * BytesPerElement(dtype);
            }

            size_t Dimentions() const
            {
                return channelsCount(layout);
            }

            bool PaddingExists() const
            {
                return (Length() != LengthWithPadding());
            }

            bool operator==(const DataTensor& t)
            {
                bool same = 
                    dtype == t.dtype &&
                    layout == t.layout &&
                    paddedVal == t.paddedVal &&
                    offset == t.offset &&
                    dims.size() == t.dims.size();
                if (same)
                {
                    for (size_t i = 0; i < dims.size(); i++)
                    {
                        same &= 
                            dims[i].v == t.dims[i].v &&
                            dims[i].pitch == t.dims[i].pitch; // TODO: do we need it
                    }
                }

                return same;
            }

            bool SameDims(const DataTensor& t) const
            {
                bool same =
                    dtype == t.dtype &&
                    layout == t.layout &&
                    dims.size() == t.dims.size();
                if (same)
                {
                    for (size_t i = 0; i < dims.size(); i++)
                    {
                        same &= dims[i].v == t.dims[i].v;
                    }
                }

                return same;
            }

            std::vector<size_t> LogicalDims()
            {
                std::vector<size_t> res(dims.size());
                std::transform(dims.begin(), dims.end(), res.begin(), [](const Dim& d) { return d.v; });
                return res;
            }

            DataTensor transform(DataLayout l) const
            {
                const uint32_t src_channels = channelsCount(layout);
                const uint32_t dst_channels = channelsCount(l);

                const size_t src_x = x().v;
                const size_t src_y = y().v;

                std::vector<size_t> vec(dst_channels);
                if (src_channels == 2 && dst_channels == 2)
                {
                    vec[channelndex(l, DataChannelName::NAME_FEATURE)] = feature().v;
                    vec[channelndex(l, DataChannelName::NAME_BATCH)] = batch().v;
                }
                else if (src_channels == 4 && dst_channels == 4)
                {
                    vec[channelndex(l, DataChannelName::NAME_X)] = x().v;
                    vec[channelndex(l, DataChannelName::NAME_Y)] = y().v;
                    vec[channelndex(l, DataChannelName::NAME_FEATURE)] = feature().v;
                    vec[channelndex(l, DataChannelName::NAME_BATCH)] = batch().v;
                }
                else if (src_channels == 2 && dst_channels == 4)
                {
                    const size_t dst_ifm = feature().v / (src_x*src_y);
                    const size_t dst_xy = feature().v % (src_x*src_y);
                    const size_t dst_y = dst_xy / src_x;
                    const size_t dst_x = dst_xy % src_x;
                    vec[channelndex(l, DataChannelName::NAME_X)] = dst_x;
                    vec[channelndex(l, DataChannelName::NAME_Y)] = dst_y;
                    vec[channelndex(l, DataChannelName::NAME_FEATURE)] = dst_ifm;
                    vec[channelndex(l, DataChannelName::NAME_BATCH)] = batch().v;
                }
                else if (src_channels == 4 && dst_channels == 2)
                {
                    const size_t dst_ifm = feature().v * src_x * src_y;
                    vec[channelndex(l, DataChannelName::NAME_FEATURE)] = dst_ifm;
                    vec[channelndex(l, DataChannelName::NAME_BATCH)] = batch().v;
                }
                else
                {
                    // TODO: implement ROI
                    assert(0);
                }

                return{ dtype, l, PADDED_VAL::UNDEFINED, 0, vec };
            }
        };

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // WeightsTensor
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        struct WeightsTensor
        {
            WeightsType         wtype = WeightsType::F16;
            WeightsLayout       layout = WeightsLayout::oiyx;
            PADDED_VAL          paddedVal = PADDED_VAL::UNDEFINED;
            size_t              offset = 0;
            NDims               dims;

            WeightsTensor() = default;
            WeightsTensor(WeightsType wt, WeightsLayout l, PADDED_VAL pv, size_t of, const NDims& nd) :
                wtype(wt), layout(l), paddedVal(pv), offset(of), dims(nd) {}
            WeightsTensor(WeightsType wt, WeightsLayout l, PADDED_VAL pv, size_t of, const std::vector<size_t>& d) :
                wtype(wt), layout(l), paddedVal(pv), offset(of), dims(CalcPitches(d)) {}
            WeightsTensor(const WeightsTensor&) = default;
            WeightsTensor& operator=(const WeightsTensor&) = default;

            NDims CalcPitches(const std::vector<size_t>& d) const
            {
                std::vector<size_t> newDims = d;

                // TOOD: it's not the right pitches. it's here in order to calculate physical size
                switch (layout)
                {
                case os_iyx_osv16:
                    assert(newDims.size() == 4);
                    newDims[3] = cldnn::round_up_to(newDims[3], 16);
                    break;
                case os_i_osv16:
                    assert(newDims.size() == 2);
                    newDims[1] = cldnn::round_up_to(newDims[1], 16);
                    break;
                case os_is_isv8_osv8:
                    assert(newDims.size() == 2);
                    newDims[1] = cldnn::round_up_to(newDims[1], 8);
                    newDims[0] = cldnn::round_up_to(newDims[0], 8);
                    break;
                case iyxo_om16x2_axy:
                    assert(newDims.size() == 4);
                    newDims[0] = cldnn::round_up_to(newDims[0], 16);
                    break;
                case iyxo_om16x2_ax_g32:
                case iyxo_om8x2_ax_g32:
                    assert(newDims.size() == 4);
                    newDims[0] = cldnn::round_up_to(newDims[0], 32);
                    break;
                default:
                    break;
                }

                NDims ret(newDims.size());
                size_t pitch = 1;

                for (size_t i = 0; i < newDims.size(); i++)
                {
                    ret[i] = { newDims[i], pitch };
                    pitch *= newDims[i];
                }

                if (layout == iyxo_om16x2_axy)
                {
                    ret[3].pitch = cldnn::round_up_to(newDims[1] * newDims[2], 2) * newDims[0];
                }
                else if (layout == iyxo_om16x2_ax_g32 || layout == iyxo_om8x2_ax_g32)
                {
                    ret[2].pitch = cldnn::round_up_to(newDims[1], 2) * newDims[0];
                    ret[3].pitch = newDims[2] * ret[2].pitch;
                }

                return ret;
            }

            Dim x()   const { return extract(layout, WeightsChannelName::NAME_X, dims); }
            Dim y()   const { return extract(layout, WeightsChannelName::NAME_Y, dims); }
            Dim ifm() const { return extract(layout, WeightsChannelName::NAME_IFM, dims); }
            Dim ofm() const { return extract(layout, WeightsChannelName::NAME_OFM, dims); }

            void SetX(const Dim d) { set(layout, WeightsChannelName::NAME_X, dims, d); }
            void SetY(const Dim d) { set(layout, WeightsChannelName::NAME_Y, dims, d); }
            void SetIFM(const Dim d) { set(layout, WeightsChannelName::NAME_IFM, dims, d); }
            void SetOFM(const Dim d) { set(layout, WeightsChannelName::NAME_OFM, dims, d); }

            size_t Length() const
            {
                return std::accumulate(dims.cbegin(), dims.cend(), (size_t)1, [](size_t val, const Dim& d) {return val*d.v; });
            }

            size_t LengthWithPadding() const
            {
                return std::accumulate(dims.cbegin(), dims.cend(), (size_t)1, [](size_t val, const Dim& d) {return std::max(val, d.pitch*d.v); });
            }

            size_t PhysicalSize() const
            {
                return (offset + LengthWithPadding()) * BytesPerElement(wtype);
            }

            bool PaddingExists() const
            {
                return (Length() != LengthWithPadding());
            }

            WeightsTensor transform(WeightsLayout l) const
            {
                const uint32_t src_channels = channelsCount(layout);
                const uint32_t dst_channels = channelsCount(l);

                const size_t src_x = x().v;
                const size_t src_y = y().v;

                std::vector<size_t> vec(dst_channels);
                if (src_channels == 2 && dst_channels == 2)
                {
                    vec[channelndex(l, WeightsChannelName::NAME_IFM)] = ifm().v;
                    vec[channelndex(l, WeightsChannelName::NAME_OFM)] = ofm().v;
                }
                else if (src_channels == 4 && dst_channels == 4)
                {
                    vec[channelndex(l, WeightsChannelName::NAME_X)] = x().v;
                    vec[channelndex(l, WeightsChannelName::NAME_Y)] = y().v;
                    vec[channelndex(l, WeightsChannelName::NAME_IFM)] = ifm().v;
                    vec[channelndex(l, WeightsChannelName::NAME_OFM)] = ofm().v;
                }
                else if (src_channels == 2 && dst_channels == 4)
                {
                    const size_t dst_ifm = ifm().v / (src_x*src_y);
                    const size_t dst_xy = ifm().v % (src_x*src_y);
                    const size_t dst_y = dst_xy / src_x;
                    const size_t dst_x = dst_xy % src_x;
                    vec[channelndex(l, WeightsChannelName::NAME_X)] = dst_x;
                    vec[channelndex(l, WeightsChannelName::NAME_Y)] = dst_y;
                    vec[channelndex(l, WeightsChannelName::NAME_IFM)] = dst_ifm;
                    vec[channelndex(l, WeightsChannelName::NAME_OFM)] = ofm().v;
                }
                else if (src_channels == 4 && dst_channels == 2)
                {
                    const size_t dst_ifm = ifm().v * src_x * src_y;
                    vec[channelndex(l, WeightsChannelName::NAME_IFM)] = dst_ifm;
                    vec[channelndex(l, WeightsChannelName::NAME_OFM)] = ofm().v;
                }
                else
                {
                    assert(0);
                }

                return{wtype, l, PADDED_VAL::UNDEFINED, 0, vec};
            }
        };
    }
}