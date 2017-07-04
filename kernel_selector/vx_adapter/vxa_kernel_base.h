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

#include <cmath>
#include <assert.h>
#include <sstream>
#include "cache/vxa_ocl_toolkit.h"
#include "api/vx_cldnn_adapter.h"
#include "vxa_common.h"
#include "kernel_selector_common.h"
#include "kernel_selector.h"

namespace clDNN
{
    class ArgumentsInfo : public ArgumentsInfoBase
    {
    public:
        virtual size_t size() const { return data.size(); }
        virtual const Args& operator[](size_t i) const { return data[i]; }
        std::vector<Args> data;
    };

    using context_holder = clDNN::gpu::context_holder;
    class BaseKernelBinary : public KernelBinary, public context_holder
    {
    public:
        BaseKernelBinary(KernelType kType) : m_kType(kType) {}

        virtual const CLKernelData& GetKernelData() const override;
        virtual KernelType          GetKernelType() const  override { return m_kType; }

        virtual bool                ShouldChangeInputTensor() const override;
        virtual TensorDesc          GetNewInputTensorDesc() const override;

        virtual bool                ShouldReorderWeights() const override;
        virtual size_t              GetNewWeightBufferSizeInBytes() const override;
        virtual bool                ReorderWeightsWithKernel() const override;
        virtual const CLKernelData& GetWeightsReorderKernelData() const override;
        virtual void                ReorderWeights(void*, size_t, void*, size_t) const override;

protected:
        WorkGroups GetWorkGroups(const KernelSelector::WorkGroupSizes&) const;
        void InitBaseParams(const BaseParams& vxParams, KernelSelector::BaseParams& clDNNParams);
        void HandleBestKernels(const KernelSelector::KernelSelctorBase& ks, const KernelSelector::Params& params, const KernelSelector::OptionalParams& options);
        void UpdateBinary(const KernelSelector::clKernelData& cldnn_data, CLKernelData& data, binary_data& binary, std::shared_ptr<ArgumentsInfoBase>& args);
        std::shared_ptr<ArgumentsInfoBase> SetupArguments(const KernelSelector::Arguments& cldnn_args);

        void UpdateTensor(Datatype dt, DataLayout layout, const uDims& srcDims, const TensorDesc& srcDesc, KernelSelector::DataTensor& dst) const;

        const KernelType m_kType = KernelType::UNKNOWN;
        binary_data m_Binary;
        binary_data m_WeightsReorderBinary;
        std::shared_ptr<ArgumentsInfoBase> m_ArgInfo;
        std::shared_ptr<ArgumentsInfoBase> m_WeightsReorderArgInfo;
        CLKernelData m_KernelData;
        CLKernelData m_WeightsKernelData;
        KernelSelector::KernelData m_cldnnKernelData;
    };
}