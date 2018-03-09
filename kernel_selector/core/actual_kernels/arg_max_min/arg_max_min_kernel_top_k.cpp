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

#include "arg_max_min_kernel_top_k.h"

namespace KernelSelector
{
	ParamsKey ArgMaxMinKernelTopK::GetSupportedKey() const
	{
		ParamsKey k;
		k.EnableInputDataType(Datatype::F16);
		k.EnableInputDataType(Datatype::F32);
		k.EnableInputDataType(Datatype::INT8);
		k.EnableOutputDataType(Datatype::F32);
		k.EnableInputLayout(DataLayout::bfyx);
		k.EnableOutputLayout(DataLayout::bfyx);
		k.EnableArgMaxMinAxis(ArgMaxMinAxis::NONE);
		k.EnableArgMaxMinOutVal(ArgMaxMinOut::MAX);
		k.EnableArgMaxMinOutVal(ArgMaxMinOut::MIN);
		k.EnableBatching();
		return k;
	}

	KernelsData ArgMaxMinKernelTopK::GetKernelsData(const Params& params, const OptionalParams& options) const
	{
		return GetCommonKernelsData(params, options, FORCE_PRIORITY_8);
	}
}