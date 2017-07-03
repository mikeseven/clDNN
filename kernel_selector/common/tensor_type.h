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
#include <cstddef>

namespace KernelSelector
{
#define KERNEL_SELECTOR_TENSOR_DIM_MAX 8

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
            bs_f_bsv8__af8,     // for optimized FC
            bs_f_bsv16__af8,    // for optimized FC
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
            os_i_osv16,
            os_i_osv8__ai8,         // TODO can we drop the alignment form layout name?
            os_i_osv16__ai8,
            i_yxs_os_yxsv2_osv16,
            iy_xs_os_xsv2_osv16__ao32,
            iy_xs_os_xsv2_osv8__ao32,
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
                X       = 0,
                Y       = 1,
                FEATURE = 2,
                ROI     = 3,
                BATCH   = 4,
            };

            enum class WeightsChannelName
            {
                X   = 0,
                Y   = 1,
                IFM = 2,
                OFM = 3,
            };

            using ChannelLocation = std::vector<int>;

            std::map<DataLayout, ChannelLocation> dataChannelMap
            {
                { DataLayout::bf,               { -1,-1, 0,-1, 1 } },
                { DataLayout::fb,               { -1,-1, 1,-1, 0 } },
                { DataLayout::bfyx,             {  0, 1, 2,-1, 3 } },
                { DataLayout::yxfb,             {  2, 3, 1,-1, 0 } },
                { DataLayout::byxf,             {  1, 2, 0,-1, 3 } },
                { DataLayout::fyxb,             {  1, 2, 3,-1, 0 } },
                { DataLayout::bs_f_bsv8__af8,   { -1,-1, 0,-1, 1 } },
                { DataLayout::bs_f_bsv16__af8,  { -1,-1, 0,-1, 1 } },
                { DataLayout::brfyx,            {  0, 1, 2, 3, 4 } },
            };

            std::map<WeightsLayout, ChannelLocation> weightsChannelMap
            {
                { WeightsLayout::oi,                            { -1,-1, 0, 1 } },
                { WeightsLayout::io,                            { -1,-1, 1, 0 } },
                { WeightsLayout::oiyx,                          {  0, 1, 2, 3 } },
                { WeightsLayout::oyxi,                          {  1, 2, 0, 3 } },
                { WeightsLayout::iyxo,                          {  1, 2, 3, 0 } },
                { WeightsLayout::yxio,                          {  2, 3, 1, 0 } },
                { WeightsLayout::os_iyx_osv16,                  {  0, 1, 2, 3 } },
                { WeightsLayout::os_i_osv8__ai8,                { -1,-1, 0, 1 } },
                { WeightsLayout::os_i_osv16__ai8,               { -1,-1, 0, 1 } },
                { WeightsLayout::os_i_osv16,                    { -1,-1, 0, 1 } },
                { WeightsLayout::i_yxs_os_yxsv2_osv16,          {  1, 2, 3, 0 } },
                { WeightsLayout::iy_xs_os_xsv2_osv16__ao32,     {  1, 2, 3, 0 } },
                { WeightsLayout::iy_xs_os_xsv2_osv8__ao32,      {  1, 2, 3, 0 } },
            };

            template <typename MapT, typename Layout, typename ChannelName>
            inline int Channelndex(MapT channelMap, Layout l, ChannelName channelName)
            {
                size_t channel = static_cast<size_t>(channelName);
                assert(channelMap.find(l) != channelMap.end());
                assert(channel < channelMap[l].size());

                return channelMap[l][channel];
            }

            template <typename MapT, typename Layout, typename ChannelName>
            inline Dim Extract(MapT channelMap, Layout l, ChannelName channelName, const NDims& dims)
            {
                const int i = Channelndex(channelMap, l, channelName);
                return ((i < 0) || (i >= (int)dims.size())) ? Dim{1, 1} : dims[i];
            }

            template <typename MapT, typename Layout>
            inline uint32_t ChannelsCount(MapT channelMap, Layout l)
            {
                const auto& entry = channelMap[l];
                return std::accumulate(entry.begin(), entry.end(), 0U, [](uint32_t count, int v) {return count + ((v != -1) ? 1 : 0); });
            }

            inline Dim Extract(DataLayout l, DataChannelName channel, const NDims& d)
            {
                return Extract(dataChannelMap, l, channel, d);
            }

            inline Dim Extract(WeightsLayout l, WeightsChannelName channel, const NDims& d)
            {
                return Extract(weightsChannelMap, l, channel, d);
            }

            inline int Channelndex(DataLayout l, DataChannelName channel)
            {
                return Channelndex(dataChannelMap, l, channel);
            }

            inline int Channelndex(WeightsLayout l, WeightsChannelName channel)
            {
                return Channelndex(weightsChannelMap, l, channel);
            }

            inline uint32_t ChannelsCount(DataLayout l)
            {
                return ChannelsCount(dataChannelMap, l);
            }

            inline uint32_t ChannelsCount(WeightsLayout l)
            {
                return ChannelsCount(weightsChannelMap, l);
            }

            inline bool SimpleLayout(WeightsLayout l)
            {
                switch (l)
                {
                case WeightsLayout::oi:
                case WeightsLayout::io:
                case WeightsLayout::oiyx:
                case WeightsLayout::oyxi:
                case WeightsLayout::iyxo:
                case WeightsLayout::yxio:
                    return true;
                default:
                    return false;
                }
            }

            inline bool SimpleLayout(DataLayout l)
            {
                switch (l)
                {
                case DataLayout::bf:
                case DataLayout::fb:
                case DataLayout::bfyx:
                case DataLayout::yxfb:
                case DataLayout::byxf:
                case DataLayout::fyxb:
                    return true;
                default:
                    return false;
                }
            }
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // PADDED_VAL
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        enum class PaddedVal
        {
            UNDEFINED,
            ZERO,
            ONE,
            HIGHEST_VAL,
            LOWEST_VAL,
        };

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // TensorBase
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        struct TensorBase
        {
        protected:
            PaddedVal    paddedVal = PaddedVal::UNDEFINED;
            size_t        offset = 0;
            NDims         dims;

        public:
            TensorBase() = default;
            TensorBase(const TensorBase&) = default;
            TensorBase& operator=(const TensorBase&) = default;

            TensorBase(PaddedVal pv, size_t of, const NDims& nd) :
                paddedVal(pv), offset(of), dims(nd) {}

            PaddedVal   GetPaddedVal()  const { return paddedVal; }
            size_t       GetOffset()     const { return offset; }
            const NDims& GetDims()       const { return dims; }

            virtual uint32_t    ElementSize() const = 0;

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
                return (offset + LengthWithPadding()) * ElementSize();
            }

            bool PaddingExists() const
            {
                return (Length() != LengthWithPadding());
            }

            std::vector<size_t> LogicalDims() const
            {
                std::vector<size_t> res(dims.size());
                std::transform(dims.begin(), dims.end(), res.begin(), [](const Dim& d) { return d.v; });
                return res;
            }
        };

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // TensorBaseT
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        template<typename DType, typename Layout>
        struct TensorBaseT : public TensorBase
        {
        protected:
            DType     dtype;
            Layout    layout;

        public:
            TensorBaseT() = default;
            TensorBaseT(const TensorBaseT&) = default;
            TensorBaseT& operator=(const TensorBaseT&) = default;

            TensorBaseT(DType dt, Layout l, PaddedVal pv, size_t of, const NDims& nd) :
                TensorBase(pv, of, nd), dtype(dt), layout(l) {}

            DType       GetDType()      const           { return dtype; }
            Layout      GetLayout()     const           { return layout; }
            uint32_t    ElementSize()   const override  { return BytesPerElement(dtype); }
            size_t      Dimentions()    const           { return ChannelsCount(layout); }
            bool        SimpleLayout()  const           { return Tensor::SimpleLayout(layout); }

            bool operator==(const TensorBaseT& t) const
            {
                bool same =
                    dtype == t.dtype      &&
                    layout == t.layout     &&
                    paddedVal == t.paddedVal  &&
                    offset == t.offset     &&
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

            bool SameDims(const TensorBaseT& t) const
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
        };

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // DataTensor
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        struct DataTensor : public TensorBaseT<Datatype, DataLayout>
        {
            DataTensor() = default;
            DataTensor(const DataTensor&) = default;
            DataTensor& operator=(const DataTensor&) = default;

            DataTensor(Datatype dt, DataLayout l, PaddedVal pv, size_t of, const NDims& nd) :
                TensorBaseT(dt, l, pv, of, nd) {}

            DataTensor(Datatype dt, DataLayout l, PaddedVal pv, size_t of, const std::vector<size_t>& d) :
                TensorBaseT<Datatype, DataLayout>(dt, l, pv, of, CalcPitches(d, l)) {}

            Dim X()         const { return Extract(layout, DataChannelName::X, dims); }
            Dim Y()         const { return Extract(layout, DataChannelName::Y, dims); }
            Dim Feature()   const { return Extract(layout, DataChannelName::FEATURE, dims); }
            Dim ROI()       const { return Extract(layout, DataChannelName::ROI, dims); }
            Dim Batch()     const { return Extract(layout, DataChannelName::BATCH, dims); }

            DataTensor  Transform(DataLayout l) const;
            DataTensor  FlattenFeatureAndSpatials() const;
        
        private:
            static NDims CalcPitches(const std::vector<size_t>& d, DataLayout l);
        };

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // WeightsTensor
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        struct WeightsTensor : TensorBaseT<WeightsType, WeightsLayout>
        {
            WeightsTensor() = default;
            WeightsTensor(const WeightsTensor&) = default;
            WeightsTensor& operator=(const WeightsTensor&) = default;

            WeightsTensor(WeightsType dt, WeightsLayout l, PaddedVal pv, size_t of, const NDims& nd) :
                TensorBaseT(dt, l, pv, of, nd) {}

            WeightsTensor(WeightsType wt, WeightsLayout l, PaddedVal pv, size_t of, const std::vector<size_t>& d) :
                TensorBaseT<WeightsType, WeightsLayout>(wt, l, pv, of, CalcPitches(d, l)) {}

            WeightsTensor Transform(WeightsLayout l) const { return Transform(l, dtype); }
            WeightsTensor Transform(WeightsLayout l, WeightsType t) const;

            Dim X()   const { return Extract(layout, WeightsChannelName::X, dims); }
            Dim Y()   const { return Extract(layout, WeightsChannelName::Y, dims); }
            Dim IFM() const { return Extract(layout, WeightsChannelName::IFM, dims); }
            Dim OFM() const { return Extract(layout, WeightsChannelName::OFM, dims); }

        private:
            static NDims CalcPitches(const std::vector<size_t>& d, WeightsLayout l);
        };
    }
}