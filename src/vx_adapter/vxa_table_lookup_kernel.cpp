#include "vxa_table_lookup_kernel.h"

namespace clDNN
{
    TableLookupKernelBinary::TableLookupKernelBinary(
        const TableLookupParams& params) :
        BaseKernelBinary(KernelType::TABLE_LOOKUP, "cnn_table_lookup"),
        m_Params(params)
    {
        InitInputOutputArgsLocations(1);

        m_EntryPoint = "table_lookup";
        const auto& out = m_Params.outDims;
        m_kernelInfo.SetGlobalWGS(out.x, out.y, out.z);

        std::stringstream jit;
        jit << GetBaseJit(m_Params);

        neural::gpu::manager::primitive_id id = GetPrimitiveID(m_Params.inputType);

        m_Binary = m_Selector->get(context().get(), jit.str(), id);
    }
}