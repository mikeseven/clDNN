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
#include "ks_ocl_toolkit.h"
#include "common_types.h"
#include "tensor_type.h"

namespace KernelSelector
{
    using DataTensor = Tensor::DataTensor;
    using WeightsTensor = Tensor::WeightsTensor;
    using DataLayout = Tensor::DataLayout;
    using WeightsLayout = Tensor::WeightsLayout;
    using PADDED_VAL = Tensor::PADDED_VAL;
    using MultiDataTensor = std::vector<DataTensor>;
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
            key.weights_layout = 0;
        }

        struct Key
        {
            union restrict_t
            {
                struct val_t
                {
                    uint32_t inputF16 : 1;
                    uint32_t inputF32 : 1;
                    uint32_t outputF16 : 1;
                    uint32_t outputF32 : 1;
                    uint32_t inputWeightsF16 : 1;
                    uint32_t inputWeightsF32 : 1;
                    uint32_t inputWeightsINT8 : 1;
                    uint32_t outputWeightsF16 : 1;
                    uint32_t outputWeightsF32 : 1;
                    uint32_t outputWeightsINT8 : 1;
                    uint32_t different_types : 1;
                    uint32_t offset : 1;
                    uint32_t pitches : 1;
                    uint32_t batching : 1;
                    uint32_t biasPerFeatureMap : 1;
                    uint32_t biasPerOutput : 1;
                    uint32_t nonBias : 1;
                    uint32_t prelu : 1;

                    union dedicated_t
                    {
                        struct norm_t
                        {
                            uint32_t across : 1;
                            uint32_t within : 1;
                            uint32_t fixed_kenrel_divider : 1;
                            uint32_t dynamic_kenrel_divider : 1;
                        } norm;
                        struct pooling_t
                        {
                            uint32_t max : 1;
                            uint32_t avg : 1;
                            uint32_t floor : 1;
                            uint32_t ceil : 1;
                            uint32_t fixed_kenrel_divider : 1;
                            uint32_t dynamic_kenrel_divider : 1;
                        } pooling;
                        struct conv_t 
                        {
                            uint32_t split : 1;
                            uint32_t dilation : 1;
                        } conv;
                        struct fc_t {} fc;
                        struct lc_t {} lc;
                        struct softmax_t 
                        {
                            uint32_t dim_x : 1;
                            uint32_t dim_y : 1;
                            uint32_t dim_feature : 1;
                        } softmax;
                    } dedicated;
                } val;
                uint64_t raw;
            } restrict;

            union machine_info_t
            {
                struct val_t
                {
                    uint32_t subgroup : 1;
                } val;
                uint32_t raw;
            } machine_info;

            static_assert(sizeof(restrict_t) == sizeof(uint64_t), "problem with union");

            uint32_t input_layout;
            uint32_t output_layout;
            uint32_t weights_layout;
        };

        void SetInputDataType(Datatype dt)
        {
            switch (dt)
            {
            case Datatype::F16:
                key.restrict.val.inputF16 = 1;
                break;
            case Datatype::F32:
                key.restrict.val.inputF32 = 1;
                break;
            default:
                break;
            }
        }

        void SetOutputDataType(Datatype dt)
        {
            switch (dt)
            {
            case Datatype::F16:
                key.restrict.val.outputF16 = 1;
                break;
            case Datatype::F32:
                key.restrict.val.outputF32 = 1;
                break;
            default:
                break;
            }
        }

        void SetInputWeightsType(WeightsType wt)
        {
            switch (wt)
            {
            case WeightsType::F16:
                key.restrict.val.inputWeightsF16 = 1;
                break;
            case WeightsType::F32:
                key.restrict.val.inputWeightsF32 = 1;
                break;
            case WeightsType::INT8:
                key.restrict.val.inputWeightsINT8 = 1;
                break;
            default:
                break;
            }
        }

        void SetOutputWeightsType(WeightsType wt)
        {
            switch (wt)
            {
            case WeightsType::F16:
                key.restrict.val.outputWeightsF16 = 1;
                break;
            case WeightsType::F32:
                key.restrict.val.outputWeightsF32 = 1;
                break;
            case WeightsType::INT8:
                key.restrict.val.outputWeightsINT8 = 1;
                break;
            default:
                break;
            }
        }

        void SetDifferentTypesSupport()
        {
            key.restrict.val.different_types = 1;
        }

        void SetInputLayout(DataLayout l)
        {
            key.input_layout |= (1 << l);
        }

        void EnableAllInputLayout()
        {
            key.input_layout = 0xffffffff;
        }

        void SetOutputLayout(DataLayout l)
        {
            key.output_layout |= (1 << l);
        }

        void EnableAllOutputLayout()
        {
            key.output_layout = 0xffffffff;
        }

        void SetWeightsLayout(WeightsLayout l)
        {
            key.weights_layout |= (1 << l);
        }

        void EnableAllWeightsLayout()
        {
            key.weights_layout = 0xffffffff;
        }

        void SetOffsetSupport()
        {
            key.restrict.val.offset = 1;
        }

        void SetPitchesSupport()
        {
            key.restrict.val.pitches = 1;
        }

        void SetBatchingSupport()
        {
            key.restrict.val.batching = 1;
        }

        void SetSubGroupSupport()
        {
            key.machine_info.val.subgroup = 1;
        }

        void SetNonBiasSupport()
        {
            key.restrict.val.nonBias = 1;
        }

        void SetBiasPerFeatureMap()
        {
            key.restrict.val.biasPerFeatureMap = 1;
        }

        void SetBiasPerOutput()
        {
            key.restrict.val.biasPerOutput = 1;
        }

        void SetPReluSupport()
        {
            key.restrict.val.prelu = 1;
        }

        void SetLRNMode(LRNMode m)
        {
            switch (m)
            {
            case LRNMode::ACROSS_CHANNEL:
                key.restrict.val.dedicated.norm.across = 1;
                break;
            case LRNMode::WITHIN_CHANNEL:
                key.restrict.val.dedicated.norm.within = 1;
                break;
            default:
                break;
            }
        }

        void SetNormalizeMode(NormalizeMode m)
        {
            switch (m)
            {
            case NormalizeMode::ACROSS_SPATIAL:
                key.restrict.val.dedicated.norm.across = 1;
                break;
            case NormalizeMode::WITHIN_SPATIAL:
                key.restrict.val.dedicated.norm.within = 1;
                break;
            default:
                break;
            }
        }

        void SetLRNKernelDividerMode(KernelDividerMode m)
        {
            switch (m)
            {
            case KernelDividerMode::FIXED:
                key.restrict.val.dedicated.norm.fixed_kenrel_divider = 1;
                break;
            case KernelDividerMode::DYNAMIC:
                key.restrict.val.dedicated.norm.dynamic_kenrel_divider = 1;
                break;
            default:
                break;
            }
        }

        void SetPoolKernelDividerMode(KernelDividerMode m)
        {
            switch (m)
            {
            case KernelDividerMode::FIXED:
                key.restrict.val.dedicated.pooling.fixed_kenrel_divider = 1;
                break;
            case KernelDividerMode::DYNAMIC:
                key.restrict.val.dedicated.pooling.dynamic_kenrel_divider = 1;
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

        void SetSplitSupport()
        {
            key.restrict.val.dedicated.conv.split = 1;
        }

        void SetDilationSupport()
        {
            key.restrict.val.dedicated.conv.dilation = 1;
        }

        void SetSoftmaxDim(SoftmaxDim d)
        {
            switch (d)
            {
            case SoftmaxDim::X:
                key.restrict.val.dedicated.softmax.dim_x = 1;
                break;
            case SoftmaxDim::Y:
                key.restrict.val.dedicated.softmax.dim_y = 1;
                break;
            case SoftmaxDim::FEATURE:
                key.restrict.val.dedicated.softmax.dim_feature = 1;
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
                ((key.input_layout & k.key.input_layout) != 0 || key.input_layout == k.key.input_layout) &&
                ((key.output_layout & k.key.output_layout) != 0 || key.output_layout == k.key.output_layout) &&
                ((key.weights_layout & k.key.weights_layout) != 0 || key.weights_layout == k.key.weights_layout);
        }

        ParamsKey Merge(const ParamsKey& k) const
        {
            ParamsKey ret;
            ret.key.restrict.raw = key.restrict.raw | k.key.restrict.raw;
            ret.key.machine_info.raw = key.machine_info.raw | k.key.machine_info.raw;
            ret.key.input_layout = key.input_layout | k.key.input_layout;
            ret.key.output_layout = key.output_layout | k.key.output_layout;
            ret.key.weights_layout = key.weights_layout | k.key.weights_layout;
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
        Params(KernelType kt, const std::string& id) : kType(kt), layerID(id) {}
        KernelType kType;

    public:
        std::string layerID;
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // BaseParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct BaseParams : public Params
    {
        virtual ~BaseParams() {}

        ActivationFunction  activationFunc = ActivationFunction::NONE;
        NonLinearParams     nlParams;
        MultiDataTensor     inputs;
        DataTensor          output;

        virtual std::string to_string() const;

        virtual ParamsKey GetParamsKey() const
        {
            ParamsKey k;

            bool bBatching = false;
            bool bPitches = false;
            bool bOffests = false;
            bool bDifferentTypes = false;

            for (const auto& i : inputs)
            {
                k.SetInputDataType(i.dtype);
                k.SetInputLayout(i.layout);

                bBatching       |= (i.batch().v > 1);
                bPitches        |= (i.PaddingExists());
                bOffests        |= (i.offset != 0);
                bDifferentTypes |= (i.dtype != output.dtype);
            }

            k.SetOutputDataType(output.dtype);
            k.SetOutputLayout(output.layout);

            if (bBatching)
            {
                k.SetBatchingSupport();
            }

            if (bPitches ||
                output.PaddingExists())
            {
                k.SetPitchesSupport();
            }

            if (bDifferentTypes)
            {
                k.SetDifferentTypesSupport();
            }

            if (bOffests ||
                output.offset != 0)
            {
                k.SetOffsetSupport();
            }

            if (activationFunc == ActivationFunction::PRELU)
            {
                k.SetPReluSupport();
            }

            return k;
        }

    protected:

        BaseParams(KernelType kt) : Params(kt, ""), inputs(1){}
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // ConvolutionParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct WeightBiasParams : public BaseParams
    {
        WeightBiasParams(KernelType kt) : BaseParams(kt) {}

        WeightsTensor weights;
        MultiDataTensor bias;

        virtual ParamsKey GetParamsKey() const
        {
            ParamsKey k = BaseParams::GetParamsKey();

            k.SetInputWeightsType(weights.wtype);
            
            // not needed - can be changed by reorder params
            //k.SetWeightsLayout(weights.layout);

            assert(bias.size() <= 1);

            if (bias.empty())
            {
                k.SetNonBiasSupport();
            }
            else if (bias[0].layout == DataLayout::bf ||
                     bias[0].layout == DataLayout::fb)
            {
                k.SetBiasPerFeatureMap();
            }
            else if (bias[0].layout == output.layout)
            {
                k.SetBiasPerOutput();
            }

            return k;
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // ConvolutionParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct ConvolutionParams : public WeightBiasParams
    {
        ConvolutionParams() : WeightBiasParams(KernelType::CONVOLUTION), convParams() {}
    
        struct DedicatedParams
        {
            uSize    filterSize;
            uSize    stride;
            uSize    dilation;
            uSize    padding;
            uint32_t split = 1;
        };

        DedicatedParams convParams;

        virtual std::string to_string() const override;

        virtual ParamsKey GetParamsKey() const
        {
            ParamsKey k = WeightBiasParams::GetParamsKey();

            if (convParams.split > 1)
            {
                k.SetSplitSupport();
            }

            if (convParams.dilation.x != 1 ||
                convParams.dilation.y != 1)
            {
                k.SetDilationSupport();
            }

            return k;
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // DeconvolutionParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct DeconvolutionParams : public WeightBiasParams
    {
        DeconvolutionParams() : WeightBiasParams(KernelType::DECONVOLUTION), deconvParams() {}

        struct DedicatedParams
        {
            uSize    filterSize;
            uSize    stride;
            uSize    dilation;
            uSize    padding;
            uint32_t split = 1;
        };

        DedicatedParams deconvParams;

        virtual std::string to_string() const override;

        virtual ParamsKey GetParamsKey() const
        {
            ParamsKey k = WeightBiasParams::GetParamsKey();

            if (deconvParams.split > 1)
            {
                k.SetSplitSupport();
            }

            if (deconvParams.dilation.x != 1 ||
                deconvParams.dilation.y != 1)
            {
                k.SetDilationSupport();
            }

            return k;
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // LRNParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct LRNParams : public BaseParams
    {
        LRNParams() : BaseParams(KernelType::LRN), lrnParams() {}

        struct DedicatedParams
        {
            LRNMode             normMode    = LRNMode::ACROSS_CHANNEL;
            KernelDividerMode   divMode     = KernelDividerMode::DONT_CARE;
            float               alpha       = 0.f;
            float               beta        = 0.f;
            float               k           = 0.f;
            uint32_t            localSize   = 0;
        };

        DedicatedParams lrnParams;

        virtual ParamsKey GetParamsKey() const
        {
            ParamsKey k = BaseParams::GetParamsKey();

            k.SetLRNMode(lrnParams.normMode);
            k.SetLRNKernelDividerMode(lrnParams.divMode);

            return k;
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // NormalizeParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct NormalizeParams : public BaseParams
    {
        NormalizeParams() : BaseParams(KernelType::NORMALIZE), normParams() {}

        struct DedicatedParams
        {
            NormalizeMode normMode = NormalizeMode::ACROSS_SPATIAL;
            float         epsilon  = 1e-10f;
            DataTensor    scale_table;
        };

        DedicatedParams normParams;

        virtual ParamsKey GetParamsKey() const
        {
            ParamsKey k = BaseParams::GetParamsKey();

            k.SetNormalizeMode(normParams.normMode);

            return k;
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // PoolingParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct PoolingParams : public BaseParams
    {
        PoolingParams() : BaseParams(KernelType::POOLING), poolParams() {}

        struct DedicatedParams
        {
            PoolType            poolType        = PoolType::MAX;
            PoolRemainder       remainderAction = PoolRemainder::FLOOR;
            KernelDividerMode   divMode         = KernelDividerMode::DONT_CARE;
            uSize               poolSize;
            uSize               poolStride;
            uSize               poolPad;
        };

        DedicatedParams poolParams;

        virtual ParamsKey GetParamsKey() const
        {
            ParamsKey k = BaseParams::GetParamsKey();

            k.SetPoolType(poolParams.poolType);
            k.SetPoolRemainder(poolParams.remainderAction);
            k.SetPoolKernelDividerMode(poolParams.divMode);

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
    struct FullyConnectedParams : public WeightBiasParams
    {
        FullyConnectedParams() : WeightBiasParams(KernelType::FULLY_CONNECTED) {}

        virtual ParamsKey GetParamsKey() const
        {
            return WeightBiasParams::GetParamsKey();
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // LocallyConnectedParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct LocallyConnectedParams : public BaseParams
    {
        LocallyConnectedParams() : BaseParams(KernelType::LOCALLY_CONNECTED), lcParams() {}

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
    struct SoftmaxParams : public BaseParams
    {
        SoftmaxParams() : BaseParams(KernelType::SOFT_MAX) {}

        struct DedicatedParams
        {
            SoftmaxDim dim = SoftmaxDim::FEATURE;
        };

        DedicatedParams smParams;

        virtual ParamsKey GetParamsKey() const
        {
            auto k = BaseParams::GetParamsKey();
            k.SetSoftmaxDim(smParams.dim);
            return k;
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // SoftMaxParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct EltwiseParams : public BaseParams
    {
        EltwiseParams() : BaseParams(KernelType::ELTWISE) {}

        struct InputType
        {
            EltwiseInputMode mode   = EltwiseInputMode::INPUT_BUFFER;
            uint32_t         index  = 0; // for inputs/temp results;
            float            scalar = 0.f;

            static InputType Buffer(uint32_t index) 
            {
                EltwiseParams::InputType input;
                input.mode = EltwiseInputMode::INPUT_BUFFER;
                input.index = index;
                return input;
            }

            static InputType Intermediate(uint32_t index)
            {
                EltwiseParams::InputType input;
                input.mode = EltwiseInputMode::INTERMEDIATE_RESULTS_INDEX;
                input.index = index;
                return input;
            }

            static InputType Scalar(float s)
            {
                EltwiseParams::InputType input;
                input.mode = EltwiseInputMode::SCALAR;
                input.scalar = s;
                return input;
            }
        };

        struct Node
        {
            std::vector<InputType> inputs;
            EltwiseMode mode;
        };

        std::vector<EltwiseParams::Node> eltwiseParams;

        virtual ParamsKey GetParamsKey() const
        {
            return BaseParams::GetParamsKey();
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // ReorderParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct ReorderBaseParams : public BaseParams
    {
        ReorderBaseParams() : BaseParams(KernelType::REORDER) {}

        virtual ParamsKey GetParamsKey() const
        {
            return BaseParams::GetParamsKey();
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // PermuteParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct PermuteParams : public ReorderBaseParams
    {
        PermuteParams() {}
        
        struct DedicatedParams
        {
            std::vector<uint16_t> order;
        };

        DedicatedParams permuteParams;

        virtual ParamsKey GetParamsKey() const
        {
            return BaseParams::GetParamsKey();
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // ReorderVxParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct ReorderVxParams : public ReorderBaseParams
    {
        ReorderVxParams() : reorderParams() {}

        struct DedicatedParams
        {
            ReorderMode mode = ReorderMode::xyzw;
        };

        DedicatedParams reorderParams;

        virtual ParamsKey GetParamsKey() const
        {
            return ReorderBaseParams::GetParamsKey();
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // ReorderParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct ReorderParams : public ReorderBaseParams
    {
        ReorderParams() : reorderParams() {}

        struct DedicatedParams
        {
            MeanSubtructMode    mode = MeanSubtructMode::NONE;
            std::vector<float>  mean_values;
            DataTensor          mean;
        };

        DedicatedParams reorderParams;

        virtual ParamsKey GetParamsKey() const
        {
            return ReorderBaseParams::GetParamsKey();
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // ReorderWeightsParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct ReorderWeightsParams : public Params
    {
        ReorderWeightsParams() : Params(KernelType::REORDER, ""), reorderParams() {}

        struct DedicatedParams
        {
            WeightsTensor input;
            WeightsTensor output;
        };

        DedicatedParams reorderParams;

        virtual ParamsKey GetParamsKey() const
        {
            ParamsKey k;
            const auto& input = reorderParams.input;
            const auto& output = reorderParams.output;
            k.SetWeightsLayout(input.layout);
            k.SetWeightsLayout(output.layout);
            k.SetInputWeightsType(input.wtype);
            k.SetOutputWeightsType(output.wtype);

            if (input.PaddingExists() ||
                output.PaddingExists())
            {
                k.SetPitchesSupport();
            }

            if (input.offset != 0 || output.offset != 0)
            {
                k.SetOffsetSupport();
            }
            return k;
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // ConvertParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct ConvertParams : public BaseParams
    {
        ConvertParams() : BaseParams(KernelType::CONVERT), convertParams() {}

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
        TableLookupParams() : BaseParams(KernelType::TABLE_LOOKUP), lookupParams() {}

        struct DedicatedParams
        {
            Datatype tableFormat = Datatype::F16;
            size_t tableSize = 0;
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
        bool bSupportSubGroupExt = false;
        uint64_t maxWorkGroupSize = 1;
        uint64_t maxLocalMemSize = 16*1024*1024;

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

            if (bSupportSubGroupExt)
            {
                k.SetSubGroupSupport();
            }

            return k;
        }

    protected:
        OptionalParams(KernelType kt) : kType(kt) {}
        KernelType kType;
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // WeightsBiasOptionalParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct WeightsBiasOptionalParams : OptionalParams
    {
        bool allow_weights_reorder = true;
    protected:
        WeightsBiasOptionalParams(KernelType kt) : OptionalParams(kt) {}
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // ConvolutionOptionalParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct ConvolutionOptionalParams : WeightsBiasOptionalParams
    {
        ConvolutionOptionalParams() : WeightsBiasOptionalParams(KernelType::CONVOLUTION) {}
        bool allow_padding = false;
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // DeconvolutionOptionalParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct DeconvolutionOptionalParams : WeightsBiasOptionalParams
    {
        DeconvolutionOptionalParams() : WeightsBiasOptionalParams(KernelType::DECONVOLUTION) {}
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // LRNOptionalParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct LRNOptionalParams : OptionalParams
    {
        LRNOptionalParams() : OptionalParams(KernelType::LRN) {}
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // NormalizeOptionalParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct NormalizeOptionalParams : OptionalParams
    {
        NormalizeOptionalParams() : OptionalParams(KernelType::NORMALIZE) {}
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
    struct FullyConnectedOptionalParams : WeightsBiasOptionalParams
    {
        FullyConnectedOptionalParams() : WeightsBiasOptionalParams(KernelType::FULLY_CONNECTED) {}
        bool allow_reorder_input = false;
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
    // ReorderVxOptionalParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct ReorderOptionalParams : OptionalParams
    {
        ReorderOptionalParams() : OptionalParams(KernelType::REORDER) {}
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // ConvertOptionalParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct ConvertOptionalParams : OptionalParams
    {
        ConvertOptionalParams() : OptionalParams(KernelType::CONVERT) {}
    };
}
