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

#include "api/vx_cldnn_adapter.h"
#include "vxa_activation_kernel.h"
#include "vxa_convolution_kernel.h"
#include "vxa_fully_connected_kernel.h"
#include "vxa_locally_connected_kernel.h"
#include "vxa_lrn_kernel.h"
#include "vxa_pooling_kernel.h"
#include "vxa_roi_pooling_kernel.h"
#include "vxa_softmax_kernel.h"
#include "vxa_eltwise_kernel.h"
#include "vxa_table_lookup_kernel.h"
#include "vxa_reorder_kernel.h"
#include "vxa_convert_kernel.h"
#include "vxa_common.h"

namespace clDNN
{
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // CreateKernelBinaryT
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    template <typename DNNKernelBinaryT, typename ParamsT>
    KernelBinary* CreateKernelBinaryT(const Params& params)
    {
        KernelBinary* pProgram = nullptr;

        DNNKernelBinaryT* pDNNProgram = new DNNKernelBinaryT(*static_cast<const ParamsT*>(&params));

        const CLKernelData data = pDNNProgram->GetKernelData();
        if (data.desc.size != 0 &&
            data.args != nullptr &&
            data.entry_point != nullptr &&
            data.workGroup.global.NullRange == false)
        {
            pProgram = pDNNProgram;
        }
        else
        {
            ReleaseKernelBinary(pDNNProgram);
        }

        return pProgram;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // CreateKernelBinary
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    KernelBinary* CreateKernelBinary(const Params& params)
    {
        KernelBinary* pKernelBinary = nullptr;

        if (IsSupported(params))
        {
            switch (params.GetType())
            {
            case KernelType::CONVOLUTION:
                pKernelBinary = CreateKernelBinaryT<ConvolutionKernelBinary, ConvolutionParams>(params);
                break;
            case KernelType::LRN:
                pKernelBinary = CreateKernelBinaryT<LRNKernelBinary, LRNParams>(params);
                break;
            case KernelType::POOLING:
                pKernelBinary = CreateKernelBinaryT<PoolingKernelBinary, PoolingParams>(params);
                break;
            case KernelType::ROI_POOLING:
                pKernelBinary = CreateKernelBinaryT<ROIPoolingKernelBinary, ROIPoolingParams>(params);
                break;
            case KernelType::FULLY_CONNECTED:
                pKernelBinary = CreateKernelBinaryT<FullyConnectedKernelBinary, FullyConnectedParams>(params);
                break;
            case KernelType::LOCALLY_CONNECTED:
                pKernelBinary = CreateKernelBinaryT<LocallyConnectedKernelBinary, LocallyConnectedParams>(params);
                break;
            case KernelType::ACTIVATION:
                pKernelBinary = CreateKernelBinaryT<ActivationKernelBinary, ActivationParams>(params);
                break;
            case KernelType::SOFT_MAX:
                pKernelBinary = CreateKernelBinaryT<SoftMaxKernelBinary, SoftMaxParams>(params);
                break;
            case KernelType::ELTWISE:
                pKernelBinary = CreateKernelBinaryT<EltwiseKernelBinary, EltwiseParams>(params);
                break;
            case KernelType::TABLE_LOOKUP:
                pKernelBinary = CreateKernelBinaryT<TableLookupKernelBinary, TableLookupParams>(params);
                break;
            case KernelType::REORDER:
                pKernelBinary = CreateKernelBinaryT<ReorderKernelBinary, ReorderParams>(params);
                break;
            case KernelType::CONVERT:
                pKernelBinary = CreateKernelBinaryT<ConvertKernelBinary, ConvertParams>(params);
                break;
            default:
                assert(0);
                break;
            }
        }

        if (pKernelBinary == nullptr)
        {
#ifndef NDEBUG
            printf("----- clDNN not in use ------- %s\n", toString(params.GetType()).c_str());
#endif
        }

        return pKernelBinary;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // ReleaseKernelBinary
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    void ReleaseKernelBinary(KernelBinary* pKernelBinary)
    {
        if (pKernelBinary)
        {
            delete pKernelBinary;
        }
    }
}