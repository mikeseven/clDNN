﻿/*
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

#include "vxa_kernel_base.h"

namespace clDNN
{
    const CLKernelData& BaseKernelBinary::GetKernelData() const
    {
        return m_KernelData;
    }

    const CLKernelData& BaseKernelBinary::GetWeightsReorderKernelData() const
    {
        return m_WeightsKernelData;
    }

    bool BaseKernelBinary::ShouldChangeInputTensor() const 
    {
        return m_cldnnKernelData.reorder_input;
    }

    TensorDesc BaseKernelBinary::GetNewInputTensorDesc() const
    {
        const KernelSelctor::BaseParams* pParams = static_cast<const KernelSelctor::BaseParams*>(m_cldnnKernelData.params.get());
        return pParams->inDesc;
    }

    WorkGroups BaseKernelBinary::GetWorkGroups(const KernelSelctor::WorkGroupSizes& work_groups) const
    {
        WorkGroups res;
        
        cl::NDRange gws = work_groups.global;
        cl::NDRange lws = work_groups.local;

        if (gws.size() != 0)
        {
            res.global = {
                gws.size() >= 1 ? gws[0] : 1,
                gws.size() >= 2 ? gws[1] : 1,
                gws.size() >= 3 ? gws[2] : 1
            };
        }

        if (lws.size() != 0)
        {
            res.local = {
                lws.size() >= 1 ? lws[0] : 1,
                lws.size() >= 2 ? lws[1] : 1,
                lws.size() >= 3 ? lws[2] : 1
            };
        }

        return res;
    }

    bool BaseKernelBinary::ShouldReorderWeights() const
    {
        return m_cldnnKernelData.weights_reorder_params.engine != KernelSelctor::WeightsReorderParams::Engine::NONE;
    }

    std::size_t BaseKernelBinary::GetNewWeightBufferSizeInBytes() const
    {
        if (ShouldReorderWeights())
        {
            return m_cldnnKernelData.weights_reorder_params.new_buffer_size;
        }

        return 0;
    }

    bool BaseKernelBinary::ReorderWeightsWithKernel() const
    {
        return m_cldnnKernelData.weights_reorder_params.engine == KernelSelctor::WeightsReorderParams::Engine::GPU;
    }

    void BaseKernelBinary::ReorderWeights(void* org, std::size_t orgBufSize, void* newBuf, std::size_t newBufSize) const
    {
        if (ShouldReorderWeights() &&
            m_cldnnKernelData.weights_reorder_params.cpu_kernel)
        {
            m_cldnnKernelData.weights_reorder_params.cpu_kernel->Execute(org, orgBufSize, newBuf, newBufSize);
        }
    }

    void BaseKernelBinary::InitBaseParams(const BaseParams& vxParams, KernelSelctor::BaseParams& ksParams)
    {
        ksParams.inputType = vxParams.inputType;

        if (vxParams.inputLayout == DataLayout::bx)
        {
            ksParams.inputLayout = KernelSelctor::DataLayout::bx;
            ksParams.outputLayout = KernelSelctor::DataLayout::bx;
        }
        else
        {
            ksParams.inputLayout = KernelSelctor::DataLayout::bfyx;
            ksParams.outputLayout = KernelSelctor::DataLayout::bfyx;
        }

        ksParams.activationFunc = vxParams.activationFunc;
        ksParams.nlParams = vxParams.nlParams;
        ksParams.inDims = vxParams.inDims;
        ksParams.inDesc = vxParams.inDesc;
        ksParams.outDims = vxParams.outDims;
        ksParams.outDesc = vxParams.outDesc;
    }

    std::shared_ptr<ArgumentsInfoBase> BaseKernelBinary::SetupArguments(const KernelSelctor::ArgumentDescpirtor& cldnn_args)
    {
        //m_ArgInfo = std::make_shared<ArgumentsInfo>();
        auto args_ptr = std::shared_ptr<ArgumentsInfoBase>(new ArgumentsInfo());
        ArgumentsInfo* args = static_cast<ArgumentsInfo*>(args_ptr.get());
        const auto& cldnn_data = cldnn_args.data;

        args->data.resize(cldnn_data.size());

        for (size_t i = 0; i < cldnn_data.size(); i++)
        {
            switch (cldnn_data[i].t)
            {
            case KernelSelctor::ArgumentDescpirtor::Types::INPUT:
                args->data[i].t = ArgumentsInfoBase::Types::INPUT;
                break;
            case KernelSelctor::ArgumentDescpirtor::Types::OUTPUT:
                args->data[i].t = ArgumentsInfoBase::Types::OUTPUT;
                break;
            case KernelSelctor::ArgumentDescpirtor::Types::WEIGHTS:
                args->data[i].t = ArgumentsInfoBase::Types::WEIGHTS;
                break;
            case KernelSelctor::ArgumentDescpirtor::Types::BIAS:
                args->data[i].t = ArgumentsInfoBase::Types::BIAS;
                break;
            case KernelSelctor::ArgumentDescpirtor::Types::LOOKUP_TABLE:
                args->data[i].t = ArgumentsInfoBase::Types::LOOKUP_TABLE;
                break;
            case KernelSelctor::ArgumentDescpirtor::Types::UINT8:
                args->data[i].t = ArgumentsInfoBase::Types::UINT8;
                args->data[i].v.u8 = cldnn_data[i].v.u8;
                break;
            case KernelSelctor::ArgumentDescpirtor::Types::UINT16:
                args->data[i].t = ArgumentsInfoBase::Types::UINT16;
                args->data[i].v.u16 = cldnn_data[i].v.u16;
                break;
            case KernelSelctor::ArgumentDescpirtor::Types::UINT32:
                args->data[i].t = ArgumentsInfoBase::Types::UINT32;
                args->data[i].v.u32 = cldnn_data[i].v.u32;
                break;
            case KernelSelctor::ArgumentDescpirtor::Types::UINT64:
                args->data[i].t = ArgumentsInfoBase::Types::UINT64;
                args->data[i].v.u64 = cldnn_data[i].v.u64;
                break;
            case KernelSelctor::ArgumentDescpirtor::Types::INT8:
                args->data[i].t = ArgumentsInfoBase::Types::INT8;
                args->data[i].v.s8 = cldnn_data[i].v.s8;
                break;
            case KernelSelctor::ArgumentDescpirtor::Types::INT16:
                args->data[i].t = ArgumentsInfoBase::Types::INT16;
                args->data[i].v.s16 = cldnn_data[i].v.s16;
                break;
            case KernelSelctor::ArgumentDescpirtor::Types::INT32:
                args->data[i].t = ArgumentsInfoBase::Types::INT32;
                args->data[i].v.s32 = cldnn_data[i].v.s32;
                break;
            case KernelSelctor::ArgumentDescpirtor::Types::INT64:
                args->data[i].t = ArgumentsInfoBase::Types::INT64;
                args->data[i].v.s64 = cldnn_data[i].v.s64;
                break;
            case KernelSelctor::ArgumentDescpirtor::Types::FLOAT32:
                args->data[i].t = ArgumentsInfoBase::Types::FLOAT32;
                args->data[i].v.f32 = cldnn_data[i].v.f32;
                break;
            case KernelSelctor::ArgumentDescpirtor::Types::FLOAT64:
                args->data[i].t = ArgumentsInfoBase::Types::FLOAT64;
                args->data[i].v.f64 = cldnn_data[i].v.f64;
                break;
            default:
                break;
            }
        }

        return args_ptr;
    }

    void BaseKernelBinary::UpdateBinary(const KernelSelctor::clKernelData& cldnn_data, CLKernelData& data, binary_data& binary, std::shared_ptr<ArgumentsInfoBase>& args)
    {
        binary = cldnn_data.GetBinary(context().get(), *GetKernelCache());
        args = SetupArguments(cldnn_data.args_desc);

        if (binary.size() > 0 && args != nullptr)
        {
            data.desc.binary = binary.data();
            data.desc.size = binary.size();
            data.entry_point = cldnn_data.kernel_string.entry_point.c_str();
            data.args = args.get();
            data.workGroup = GetWorkGroups(cldnn_data.work_groups);
        }
    }

    void BaseKernelBinary::HandleBestKernels(const KernelSelctor::KernelSelctorBase& ks, const KernelSelctor::Params& params, const KernelSelctor::OptionalParams& options)
    {
        const KernelSelctor::KernelsData& kds = ks.GetBestKernels(params, options);

        if (kds.size() && kds[0].kernels.size())
        {
            m_cldnnKernelData = kds[0];

            UpdateBinary(
                m_cldnnKernelData.kernels[0], 
                m_KernelData, 
                m_Binary, 
                m_ArgInfo);

            if (m_cldnnKernelData.weights_reorder_params.engine == KernelSelctor::WeightsReorderParams::Engine::GPU)
            {
                UpdateBinary(
                    m_cldnnKernelData.weights_reorder_params.cl_kernel, 
                    m_WeightsKernelData, 
                    m_WeightsReorderBinary, 
                    m_WeightsReorderArgInfo);
            }
        }
        else
        {
            //printf("Internal Error\n");
        }
    }
}