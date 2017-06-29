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

#include "vxa_table_lookup_kernel.h"
#include "table_lookup/table_lookup_kernel_selector.h"

namespace clDNN
{
    TableLookupKernelBinary::TableLookupKernelBinary(
        const TableLookupParams& params) :
        BaseKernelBinary(KernelType::TABLE_LOOKUP),
        m_Params(params)
    {
        KernelSelector::TableLookupParams ksParams;

        InitBaseParams(params, ksParams);
        ksParams.lookupParams.tableFormat = params.lookupParams.tableFormat;
        ksParams.lookupParams.tableSize = params.lookupParams.tableSize;

        KernelSelector::TableLookupOptionalParams ksOptParams;

        HandleBestKernels(KernelSelector::TableLookupKernelSelctor::Instance(), ksParams, ksOptParams);
    }
}