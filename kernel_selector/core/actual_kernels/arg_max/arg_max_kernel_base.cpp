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

#include "arg_max_kernel_base.h"

namespace KernelSelector
{
	bool ArgMaxKernelBase::Validate(const Params& p, const OptionalParams& o) const
	{
		if (p.GetType() != KernelType::ARGMAX ||
			o.GetType() != KernelType::ARGMAX)
		{
			return false;
		}

		return true;
	}

	JitConstants ArgMaxKernelBase::GetJitConstants(const ArgMaxParams& params) const
	{
		JitConstants mem_consts = MakeBaseParamsJitConstants(params);

		return mem_consts;
	}

	ArgMaxKernelBase::DispatchData ArgMaxKernelBase::SetDefault(const ArgMaxParams& params) const
	{
		const auto& output = params.output;

		DispatchData kd;

		kd.fp16UnitUsed = params.inputs[0].GetDType() == Datatype::F16;

		if (output.GetLayout() == DataLayout::bfyx || output.GetLayout() == DataLayout::byxf)
		{
			// Determine global work sizes.
			kd.gws2 = output.Batch().v * output.Feature().v;    // B, F
			kd.gws0 = Align(output.X().v, 32);        // X
			kd.gws1 = output.Y().v;                             // Y

																// Find largest positive local work size that is divider for global work size.
			kd.lws0 = 32;
			kd.lws1 = 1;
			kd.lws2 = 1;
		}
		else
		{
			// Determine global work sizes.
			kd.gws0 = output.Batch().v * output.Feature().v;    // B, F
			kd.gws1 = output.X().v;                             // X
			kd.gws2 = output.Y().v;                             // Y

			kd.lws0 = std::min(std::max(kd.gws0, static_cast<size_t>(1)), static_cast<size_t>(32));
			while (kd.gws0 % kd.lws0 != 0)
			{
				--kd.lws0;
			}
			kd.lws1 = 1;
			kd.lws2 = 1;
		}

		return kd;
	}

	KernelsData ArgMaxKernelBase::GetCommonKernelsData(const Params& params, const OptionalParams& options, float estimatedTime) const
	{
		if (!Validate(params, options))
		{
			return{};
		}

		const ArgMaxParams& orgParams = static_cast<const ArgMaxParams&>(params);

		DispatchData runInfo = SetDefault(orgParams);

		KernelData kd = KernelData::Default<ArgMaxParams>(params);

		auto cldnn_jit = GetJitConstants(orgParams);
		auto entry_point = GetEntryPoint(kernelName, orgParams.layerID, options);
		auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

		auto& kernel = kd.kernels[0];
		FillCLKernelData(kernel, runInfo, kernelName, jit, entry_point);

		kd.estimatedTime = estimatedTime;

		return{ kd };
	}
}