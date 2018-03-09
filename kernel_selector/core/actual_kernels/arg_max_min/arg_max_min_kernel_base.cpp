/*
// Copyright (c) 2018 Intel Corporation
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

#include "arg_max_min_kernel_base.h"

namespace KernelSelector
{
	bool ArgMaxMinKernelBase::Validate(const Params& p, const OptionalParams& o) const
	{
		if (p.GetType() != KernelType::ARG_MAX_MIN ||
			o.GetType() != KernelType::ARG_MAX_MIN)
		{
			return false;
		}

		return true;
	}

	JitConstants ArgMaxMinKernelBase::GetJitConstants(const ArgMaxMinParams& params) const
	{
		JitConstants mem_consts = MakeArgMaxJitConstants(params);

		return mem_consts;
	}

	ArgMaxMinKernelBase::DispatchData ArgMaxMinKernelBase::SetDefault(const ArgMaxMinParams& params) const
	{
		DispatchData kd;

		kd.fp16UnitUsed = params.inputs[0].GetDType() == Datatype::F16;

		// Determine global work sizes.
		kd.gws0 = 128;     // X
		kd.gws1 = params.inputs[0].Batch().v;                   // Y
		kd.gws2 = 1; // output.Batch().v * output.Feature().v;    // B, F

															// Find largest positive local work size that is divider for global work size.
		kd.lws0 = 128;
		kd.lws1 = 1;
		kd.lws2 = 1;

		return kd;
	}

	KernelsData ArgMaxMinKernelBase::GetCommonKernelsData(const Params& params, const OptionalParams& options, float estimatedTime) const
	{
		if (!Validate(params, options))
		{
			return{};
		}

		const ArgMaxMinParams& orgParams = static_cast<const ArgMaxMinParams&>(params);

		DispatchData runInfo = SetDefault(orgParams);

		KernelData kd = KernelData::Default<ArgMaxMinParams>(params);

		auto cldnn_jit = GetJitConstants(orgParams);
		auto entry_point = GetEntryPoint(kernelName, orgParams.layerID, options);
		auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

		auto& kernel = kd.kernels[0];
		FillCLKernelData(kernel, runInfo, kernelName, jit, entry_point);

		kd.estimatedTime = estimatedTime;

		return{ kd };
	}
}