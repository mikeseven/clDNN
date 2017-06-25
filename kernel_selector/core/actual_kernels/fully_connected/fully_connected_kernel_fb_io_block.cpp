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

#include "fully_connected_kernel_fb_io_block.h"
#include "kernel_selector_utils.h"

namespace KernelSelector 
{
    ParamsKey FullyConnected_fb_io_block::GetSupportedKey() const
    {
        ParamsKey k;
        k.SetInputDataType(Datatype::F16);
        k.SetOutputDataType(Datatype::F16);
        k.SetInputWeightsType(WeightsType::F16);
        k.EnableAllInputLayout();
        k.SetOutputLayout(DataLayout::fb);
        k.SetBatchingSupport();
        k.SetBiasPerFeatureMap();
        k.SetNonBiasSupport();
        k.SetSubGroupSupport();
        return k;
    }

    FullyConnected_fb_io_block::DispatchData FullyConnected_fb_io_block::SetDefault(const FullyConnectedParams& arg) const
    {
        DispatchData kd = SetKernelData(arg);

        const auto& output = arg.output;
        
        auto batch_size = output.Batch().v;
        auto response_size = output.Feature().v;
        //bool batch_size_pow_2 = batch_size > 0 && (batch_size & (batch_size - 1)) == 0;

        constexpr uint32_t unit_byte_size = sizeof(cl_half);
        const char* chunk_type = "uint";
        constexpr uint32_t chunk_byte_size = sizeof(cl_uint);
        constexpr uint32_t sub_group_size = 16;
        constexpr uint32_t units_per_chunk = chunk_byte_size / unit_byte_size;
        constexpr uint32_t units_per_sg_read = sub_group_size * units_per_chunk;

        if (batch_size    > 0 && batch_size % units_per_sg_read == 0 &&
            response_size > 0 && response_size * unit_byte_size % 4 == 0) // Temporary: response size must be compatible with block read.
        {
            // Number of response groups. Each group (except last) writes units_per_sg_read responses
            // for at least one input data set from batch.
            auto rg_count = cldnn::ceil_div(response_size, units_per_sg_read);

            kd.lws0 = sub_group_size;
            // Number of work items needed to process all response groups.
            kd.gws0 = rg_count * sub_group_size;
            kd.lws1 = 1;
            kd.gws1 = batch_size / units_per_sg_read;

            kd.data_xb_xb_fp16.unit_byte_size       = unit_byte_size;
            kd.data_xb_xb_fp16.chunk_type           = chunk_type;
            kd.data_xb_xb_fp16.chunk_byte_size      = chunk_byte_size;
            kd.data_xb_xb_fp16.units_per_chunk      = units_per_chunk;
            kd.data_xb_xb_fp16.bytes_per_sg_read    = sub_group_size * chunk_byte_size;
            kd.data_xb_xb_fp16.units_per_sg_read    = units_per_sg_read;
            kd.data_xb_xb_fp16.rg_count             = (uint32_t)rg_count;
            kd.data_xb_xb_fp16.last_rg_size         = response_size % units_per_sg_read;
        }
        else
        {
            throw std::runtime_error("Unsupported");
        }

        return kd;
    }

    JitConstants FullyConnected_fb_io_block::GetJitConstants(const FullyConnectedParams& params, const DispatchData& run_info) const
    {
        auto cldnn_jit = IGKFullyConnectedKernelBase::GetJitConstants(params, run_info);
        cldnn_jit.AddConstants({
            MakeJitConstant("SUB_GROUP_SIZE",        run_info.lws0),
            MakeJitConstant("WORK_ITEMS_PER_BATCH",  run_info.gws1),
            MakeJitConstant("UNIT_BYTE_SIZE",        run_info.data_xb_xb_fp16.unit_byte_size),
            MakeJitConstant("CHUNK_TYPE",            run_info.data_xb_xb_fp16.chunk_type),
            MakeJitConstant("CHUNK_BYTE_SIZE",       run_info.data_xb_xb_fp16.chunk_byte_size),
            MakeJitConstant("UNITS_PER_CHUNK",       run_info.data_xb_xb_fp16.units_per_chunk),
            MakeJitConstant("BYTES_PER_SG_READ",     run_info.data_xb_xb_fp16.bytes_per_sg_read),
            MakeJitConstant("UNITS_PER_SG_READ",     run_info.data_xb_xb_fp16.units_per_sg_read),
            MakeJitConstant("RG_COUNT",              run_info.data_xb_xb_fp16.rg_count),
            MakeJitConstant("LAST_RG_SIZE",          run_info.data_xb_xb_fp16.last_rg_size),
        });
        return cldnn_jit;
    }

    KernelsData FullyConnected_fb_io_block::GetKernelsData(const Params& params, const OptionalParams& optParams) const
    {
        assert(params.GetType() == KernelType::FULLY_CONNECTED);

        const auto& orgParams = static_cast<const FullyConnectedParams&>(params);

        const auto& output = orgParams.output;
        const auto batches = output.Batch().v;
        const auto x_size = output.Length() / batches;

        const bool bSupportedBatch = (batches % 8) == 0;
        const bool bSupportedFeature = (x_size % 8) == 0;

        if (!bSupportedBatch || 
            !bSupportedFeature)
        {
            return KernelsData();
        }

        float estimated_time =
            orgParams.inputs[0].GetDType() == Datatype::F16 && batches >= 16 ?
            FORCE_PRIORITY_3 : FORCE_PRIORITY_5;

        // TODO: it should be fb_io. but the original code use this kernel with yxfb and yxio 
        //       (fb == fyxb flatten fyx, not yxfb flatten yxf).
        //       the order of the add operation cause some numeric changes. in order to avoid them right now we use yxfb/oiyx instead.
        // return GetCommonKernelsData(params, optParams, DataLayout::fb, WeightsLayout::io, estimated_time);
        return GetCommonKernelsData(params, optParams, DataLayout::yxfb, WeightsLayout::yxio, estimated_time);
    }
}