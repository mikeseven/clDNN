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
#include "gpu_linker.h"

namespace neural { namespace gpu { namespace manager {

gpu_program gpu_linker::link(context * context, const std::vector<cache::binary_data>& kernels)
{
	auto& clContext = context->context();
	auto& device = context->device();
	cl::vector<cl::vector<unsigned char>> binaries;
	for (const auto & k : kernels)
	{
		binaries.emplace_back(k.begin(), k.end());
	}
	cl::Program program(clContext, cl::vector<cl::Device>(1, device), binaries);
	program.build();
    return program;
}

} } }
