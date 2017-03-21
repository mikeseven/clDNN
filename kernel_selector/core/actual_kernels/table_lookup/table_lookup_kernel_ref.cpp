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

#include "table_lookup_kernel_ref.h"
 
namespace KernelSelctor {

    ParamsKey TableLookupKernelRef::GetSupportedKey() const
    {
        ParamsKey k;
        k.SetDataType(Datatype::F16);
        k.SetDataType(Datatype::F32);
        k.EnableAllInputLayout();
        k.EnableAllOutputLayout();
        k.SetOffsetSupport();
        k.SetPitchesSupport();
        k.SetNumDims(4);
        return k;
    }

    KernelsData TableLookupKernelRef::GetKernelsData(const Params& params, const OptionalParams&) const
    {
        assert(params.GetType() == KernelType::TABLE_LOOKUP);

        KernelData kd = KernelData::Default<TableLookupParams>(params, 1);

        TableLookupParams& newParams = *static_cast<TableLookupParams*>(kd.params.get());
        newParams.inputLayout = newParams.outputLayout = bfyx;

        std::stringstream jit;
        jit << GetBaseJit(newParams)
            << "#define TABLE_SIZE (" << newParams.lookupParams.tableSize << ")\n";

        if (newParams.lookupParams.tableFormat == Datatype::F16)
        {
            jit << "#define LUT_TYPE half\n";
        }
        else
        {
            jit << "#define LUT_TYPE float\n";
        }

        kd.estimated_time = DONT_USE_IF_HAVE_SOMETHING_ELSE;

        const auto& out = newParams.outDims;
        auto& kernel = kd.kernels[0];
        kernel.work_groups.global = cl::NDRange(out.x, out.y, out.z*out.w);
        kernel.kernel_string = GetKernelString(kernel_name, jit.str(), "table_lookup");
        kernel.args_desc = GetArgumentDesc(1, false, false);
        kernel.args_desc.data.push_back({ ArgumentDescpirtor::Types::LOOKUP_TABLE, 0 });

        return{ kd };
    }
}