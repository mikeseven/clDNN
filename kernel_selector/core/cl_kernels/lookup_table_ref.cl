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


#include "include/include_all.cl"

KERNEL(lookup_table)(const __global UNIT_TYPE* input0, const __global uint* input1, __global UNIT_TYPE* output)
{
    const uint x    = (uint)get_global_id(0);
    const uint b    = (uint)get_global_id(1);
	const uint size = INPUT0_SIZE_X * INPUT0_SIZE_Y * INPUT0_FEATURE_NUM;
	const uint global_index = b * VAL_NUM + x;
    
    output[global_index] = input0[input1[global_index] + b*size];
}
	