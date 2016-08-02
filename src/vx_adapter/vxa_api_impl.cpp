#include "api/vx_cldnn_adapter.h"
#include "vxa_activation_kernel.h"
#include "vxa_convolution_kernel.h"
#include "vxa_fully_connected_kernel.h"
#include "vxa_locally_connected_kernel.h"
#include "vxa_normalization_kernel.h"
#include "vxa_pooling_kernel.h"
#include "vxa_softmax_kernel.h"
#include "vxa_binary_op_kernel.h"
#include "vxa_table_lookup_kernel.h"
#include "vxa_reorder_kernel.h"
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

        if (pDNNProgram->GetBinaryDesc().size != 0)
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
            case KernelType::NORMALIZATION:
                pKernelBinary = CreateKernelBinaryT<NormalizationKernelBinary, NormalizationParams>(params);
                break;
            case KernelType::POOLING:
                pKernelBinary = CreateKernelBinaryT<PoolingKernelBinary, PoolingParams>(params);
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
            case KernelType::BINARY_OP:
                pKernelBinary = CreateKernelBinaryT<BinaryOpKernelBinary, BinaryOpParams>(params);
                break;
            case KernelType::TABLE_LOOKUP:
                pKernelBinary = CreateKernelBinaryT<TableLookupKernelBinary, TableLookupParams>(params);
                break;
            case KernelType::REORDER:
                pKernelBinary = CreateKernelBinaryT<ReorderKernelBinary, ReorderParams>(params);
                break;
            default:
                assert(0);
                break;
            }
        }

        if (pKernelBinary == nullptr)
        {
            //printf("----- clDNN not in use ------- %s\n", toString(params.GetType()).c_str());
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