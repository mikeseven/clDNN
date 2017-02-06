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

#include <cstddef>
#include "ocl_toolkit.h"
#include "common_types.h"

namespace KernelSelctor
{
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // DataLayout
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    enum DataLayout
    {
        x = 0,
        xb,                 // 1D+batch
        bx,                 // 1D+batch
        yxfn,               // 3D + number of neurons - used in fully connected weights
        yxfb,               // 3D+batch
        byxf,               // for convolution_cpu_jit_batch1
        bfyx,               // used in Caffe
        fyxb,               // used in Caffe
        bs_xs_xsv8_bsv8,    // format used only for Fully connected: bs - batch slice, xs - x slice, bsv8 - 8 values of single slice, xsv - 8 values of single slice 
        data_layoyt_count
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // ParamsKey
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    class ParamsKey
    {
    public:
        ParamsKey()
        {
            key.restrict.raw = 0;
            key.machine_info.raw = 0;
            key.input_layout = 0;
            key.output_layout = 0;
            key.num_of_dims = 0;
        }

        struct Key
        {
            union restrict_t
            {
                struct val_t
                {
                    uint F16 : 1;
                    uint F32 : 1;
                    uint offset : 1;
                    uint pitches : 1;

                    union dedicated_t
                    {
                        struct norm_t
                        {
                            uint across : 1;
                            uint within : 1;
                        } norm;
                        struct pooling_t
                        {
                            uint max : 1;
                            uint avg : 1;
                            uint floor : 1;
                            uint ceil : 1;
                        } pooling;
                        struct conv_t {} conv;
                        struct fc_t {} fc;
                        struct lc_t {} lc;
                        struct softmax_t {} softmax;
                    } dedicated;
                } val;
                uint64_t raw;
            } restrict;

            union machine_info_t
            {
                struct val_t
                {
                    uint subgroup : 1;
                } val;
                uint32_t raw;
            } machine_info;

            static_assert(sizeof(restrict_t) == sizeof(uint64_t), "problem with union");

            uint input_layout;
            uint output_layout;
            uint num_of_dims;
        };

        void SetDataType(Datatype dt)
        {
            switch (dt)
            {
            case Datatype::F16:
                key.restrict.val.F16 = 1;
                break;
            case Datatype::F32:
                key.restrict.val.F32 = 1;
                break;
            default:
                break;
            }
        }

        void SetInputLayout(DataLayout l)
        {
            key.input_layout |= (1 << l);
        }

        void SetOutputLayout(DataLayout l)
        {
            key.output_layout |= (1 << l);
        }

        void SetOffsetSupport()
        {
            key.restrict.val.offset = 1;
        }

        void SetPitchesSupport()
        {
            key.restrict.val.pitches = 1;
        }

        void SetSubGroupSupport()
        {
            key.machine_info.val.subgroup = 1;
        }

        void SetNumDims(uint n)
        {
            key.num_of_dims = n;
        }

        void SetNormalizationMode(NormalizationMode m)
        {
            switch (m)
            {
            case NormalizationMode::ACROSS_CHANNELS:
                key.restrict.val.dedicated.norm.across = 1;
                break;
            case NormalizationMode::WITHIN_CHANNEL:
                key.restrict.val.dedicated.norm.within = 1;
                break;
            default:
                break;
            }
        }

        void SetPoolType(PoolType t)
        {
            switch (t)
            {
            case PoolType::MAX:
                key.restrict.val.dedicated.pooling.max = 1;
                break;
            case PoolType::AVG:
                key.restrict.val.dedicated.pooling.avg = 1;
                break;
            default:
                break;
            }
        }

        void SetPoolRemainder(PoolRemainder r)
        {
            switch (r)
            {
            case PoolRemainder::FLOOR:
                key.restrict.val.dedicated.pooling.floor = 1;
                break;
            case PoolRemainder::CEIL:
                key.restrict.val.dedicated.pooling.ceil = 1;
                break;
            default:
                break;
            }
        }

        bool Support(const ParamsKey& k) const
        {
            return 
                ((key.restrict.raw & k.key.restrict.raw) == k.key.restrict.raw) && // check if this kernel supports this params
                ((key.machine_info.raw & k.key.machine_info.raw) == key.machine_info.raw) && // check if machine supports this kernel
                ((key.input_layout & k.key.input_layout) != 0) &&
                ((key.output_layout & k.key.output_layout) != 0) &&
                (key.num_of_dims >= k.key.num_of_dims);
        }

        ParamsKey Merge(const ParamsKey& k) const
        {
            ParamsKey ret;
            ret.key.restrict.raw = key.restrict.raw | k.key.restrict.raw;
            ret.key.machine_info.raw = key.machine_info.raw | k.key.machine_info.raw;
            ret.key.input_layout = key.input_layout | k.key.input_layout;
            ret.key.output_layout = key.output_layout | k.key.output_layout;
            ret.key.num_of_dims = std::max(key.num_of_dims, k.key.num_of_dims);
            return ret;
        }

    private:
        Key key;
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Params
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct Params
    {
        virtual ~Params() {}

        KernelType GetType() const { return kType; }
        virtual ParamsKey GetParamsKey() const = 0;

    protected:
        Params(KernelType kt) : kType(kt) {}
        KernelType kType;
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // BaseParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct BaseParams : public Params
    {
        virtual ~BaseParams() {}

        Datatype            inputType = Datatype::F16;
        DataLayout          inputLayout = DataLayout::bfyx;
        DataLayout          outputLayout = DataLayout::bfyx;
        ActivationFunction  activationFunc = ActivationFunction::NONE;
        NonLinearParams     nlParams;
        uDims               inDims;
        TensorDesc          inDesc;
        uDims               outDims;
        TensorDesc          outDesc;

        virtual std::string to_string() const;

        virtual ParamsKey GetParamsKey() const
        {
            ParamsKey k;

            k.SetDataType(inputType);
            k.SetInputLayout(inputLayout);
            k.SetOutputLayout(outputLayout);

            uint num_of_dims =
                inDims.w != 1 ? 4 :
                inDims.z != 1 ? 3 :
                inDims.y != 1 ? 2 : 1;
            k.SetNumDims(num_of_dims);

            if (num_of_dims > 1)
            {
                if (inDims.x != inDesc.pitches.x ||
                    inDims.y != inDesc.pitches.y ||
                    inDims.z != inDesc.pitches.z ||
                    outDims.x != outDesc.pitches.x ||
                    outDims.y != outDesc.pitches.y ||
                    outDims.z != outDesc.pitches.z)
                {
                    k.SetPitchesSupport();
                }
            }

            if (inDesc.offset != 0 || outDesc.offset != 0)
            {
                k.SetOffsetSupport();
            }

            // TODO: moved to engine
            if (neural::gpu::gpu_toolkit::get()->extension_supported("cl_intel_subgroups_short"))
            {
                k.SetSubGroupSupport();
            }

            return k;
        }

    protected:

        BaseParams(KernelType kt) : Params(kt) {}
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // ConvolutionParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct ConvolutionParams : public BaseParams
    {
        ConvolutionParams() : BaseParams(KernelType::CONVOLUTION) {}
    
        struct DedicatedParams
        {
            uSize filterSize;
            uSize stride;
            uSize padding;
        };

        DedicatedParams convParams;

        virtual std::string to_string() const override;

        virtual ParamsKey GetParamsKey() const
        {
            return BaseParams::GetParamsKey();
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // NormalizationParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct NormalizationParams : public BaseParams
    {
        NormalizationParams() : BaseParams(KernelType::NORMALIZATION) {}

        struct DedicatedParams
        {
            NormalizationMode normMode = NormalizationMode::ACROSS_CHANNELS;
            float             alpha = 0.f;
            float             beta = 0.f;
            uint              localSize = 0;
        };

        DedicatedParams normParams;

        virtual ParamsKey GetParamsKey() const
        {
            ParamsKey k = BaseParams::GetParamsKey();

            k.SetNormalizationMode(normParams.normMode);

            return k;
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // PoolingParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct PoolingParams : public BaseParams
    {
        PoolingParams() : BaseParams(KernelType::POOLING) {}

        struct DedicatedParams
        {
            PoolType      poolType = PoolType::MAX;
            uSize         poolSize;
            uSize         poolStride;
            uSize         poolPad;
            PoolRemainder remainderAction = PoolRemainder::FLOOR;
        };

        DedicatedParams poolParams;

        virtual ParamsKey GetParamsKey() const
        {
            ParamsKey k = BaseParams::GetParamsKey();

            k.SetPoolType(poolParams.poolType);
            k.SetPoolRemainder(poolParams.remainderAction);

            return k;
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // ROIPoolingParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct ROIPoolingParams : public BaseParams
    {
        ROIPoolingParams() : BaseParams(KernelType::ROI_POOLING) {}

        size_t rois = 0;
        size_t pitch_rois_r = 0;
        size_t pitch_rois_b = 0;

        virtual ParamsKey GetParamsKey() const
        {
            return BaseParams::GetParamsKey();
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // FullyConnectedParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct FullyConnectedParams : public BaseParams
    {
        FullyConnectedParams() : BaseParams(KernelType::FULLY_CONNECTED) {}

        virtual ParamsKey GetParamsKey() const
        {
            return BaseParams::GetParamsKey();
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // LocallyConnectedParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct LocallyConnectedParams : public BaseParams
    {
        LocallyConnectedParams() : BaseParams(KernelType::LOCALLY_CONNECTED) {}

        struct DedicatedParams
        {
            uSize filterSize;
            uSize stride;
            uSize padding;
        };

        DedicatedParams lcParams;

        virtual ParamsKey GetParamsKey() const
        {
            return BaseParams::GetParamsKey();
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // ActivationParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct ActivationParams : public BaseParams
    {
        ActivationParams() : BaseParams(KernelType::ACTIVATION) {}

        virtual ParamsKey GetParamsKey() const
        {
            return BaseParams::GetParamsKey();
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // SoftMaxParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct SoftMaxParams : public BaseParams
    {
        SoftMaxParams() : BaseParams(KernelType::SOFT_MAX) {}

        virtual ParamsKey GetParamsKey() const
        {
            return BaseParams::GetParamsKey();
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // SoftMaxParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct EltwiseParams : public BaseParams
    {
        EltwiseParams() : BaseParams(KernelType::ELTWISE) {}

        struct DedicatedParams
        {
            EltwiseMode mode = EltwiseMode::ADD;
            float scalar = 0;
        };

        DedicatedParams eltwiseParams;

        virtual ParamsKey GetParamsKey() const
        {
            return BaseParams::GetParamsKey();
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // SoftMaxParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct ReorderVxParams : public BaseParams
    {
        ReorderVxParams() : BaseParams(KernelType::REORDER) {}

        struct DedicatedParams
        {
            ReorderMode mode = ReorderMode::xyzw;
        };

        DedicatedParams reorderParams;

        virtual ParamsKey GetParamsKey() const
        {
            return BaseParams::GetParamsKey();
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // ConvertParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct ConvertParams : public BaseParams
    {
        ConvertParams() : BaseParams(KernelType::CONVERT) {}

        struct DedicatedParams
        {
            ConvertTypes covertType = ConvertTypes::U16;
        };

        DedicatedParams convertParams;

        virtual ParamsKey GetParamsKey() const
        {
            return BaseParams::GetParamsKey();
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // TableLookupParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct TableLookupParams : public BaseParams
    {
        TableLookupParams() : BaseParams(KernelType::TABLE_LOOKUP) {}

        struct DedicatedParams
        {
            Datatype tableFormat = Datatype::F16;
            std::size_t tableSize = 0;
        };

        DedicatedParams lookupParams;

        virtual ParamsKey GetParamsKey() const
        {
            return BaseParams::GetParamsKey();
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // OptionalParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct OptionalParams
    {
        virtual ~OptionalParams() {}

        KernelType GetType() const { return kType; }

        std::vector<DataLayout> input_layouts;
        std::vector<DataLayout> output_layouts;

        virtual ParamsKey GetSupportedKey() const
        {
            ParamsKey k;

            for (auto l : input_layouts)
            {
                k.SetInputLayout(l);
            }

            for (auto l : output_layouts)
            {
                k.SetOutputLayout(l);
            }

            return k;
        }

    protected:
        OptionalParams(KernelType kt) : kType(kt) {}
        KernelType kType;
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // ConvolutionOptionalParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct ConvolutionOptionalParams : OptionalParams
    {
        ConvolutionOptionalParams() : OptionalParams(KernelType::CONVOLUTION) {}
        bool allow_padding = false;
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // NormalizationOptionalParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct NormalizationOptionalParams : OptionalParams
    {
        NormalizationOptionalParams() : OptionalParams(KernelType::NORMALIZATION) {}
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // PoolingOptionalParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct PoolingOptionalParams : OptionalParams
    {
        PoolingOptionalParams() : OptionalParams(KernelType::POOLING) {}
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // ROIPoolingOptionalParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct ROIPoolingOptionalParams : OptionalParams
    {
        ROIPoolingOptionalParams() : OptionalParams(KernelType::ROI_POOLING) {}
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // FullyConnectedOptionalParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct FullyConnectedOptionalParams : OptionalParams
    {
        FullyConnectedOptionalParams() : OptionalParams(KernelType::FULLY_CONNECTED) {}
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // LocallyConnectedOptionalParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct LocallyConnectedOptionalParams : OptionalParams
    {
        LocallyConnectedOptionalParams() : OptionalParams(KernelType::LOCALLY_CONNECTED) {}
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // ActivationOptionalParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct ActivationOptionalParams : OptionalParams
    {
        ActivationOptionalParams() : OptionalParams(KernelType::ACTIVATION) {}
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // SoftmaxOptionalParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct SoftmaxOptionalParams : OptionalParams
    {
        SoftmaxOptionalParams() : OptionalParams(KernelType::SOFT_MAX) {}
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // EltwiseOptionalParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct EltwiseOptionalParams : OptionalParams
    {
        EltwiseOptionalParams() : OptionalParams(KernelType::ELTWISE) {}
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // TableLookupOptionalParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct TableLookupOptionalParams : OptionalParams
    {
        TableLookupOptionalParams() : OptionalParams(KernelType::TABLE_LOOKUP) {}
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // ReorderVxOptionalParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct ReorderVxOptionalParams : OptionalParams
    {
        ReorderVxOptionalParams() : OptionalParams(KernelType::REORDER) {}
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // ConvertOptionalParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct ConvertOptionalParams : OptionalParams
    {
        ConvertOptionalParams() : OptionalParams(KernelType::CONVERT) {}
    };
}