// Copyright (c) 2016-2017 Intel Corporation
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

#define GLOBAL_SIZE 16
#define LOCAL_SIZE 16


typedef struct /* Index and Value type that holds index and value used in this kernel */
{
    uint index; 
    UNIT_TYPE value; 
} iav_type;

__attribute__((reqd_work_group_size(LOCAL_SIZE, INPUT0_BATCH_NUM, 1)))
KERNEL(arg_max_gpu)(const __global UNIT_TYPE* input, __global UNIT_TYPE* output)
{
	__local iav_type scratch[INPUT0_BATCH_NUM][LOCAL_SIZE];
    const uint x    = (uint)get_global_id(0);
    const uint y    = (uint)get_global_id(1);
    const uint bf   = (uint)get_global_id(2);
    const uint f    = bf % INPUT0_FEATURE_NUM;
    const uint b    = bf / INPUT0_FEATURE_NUM;
	const uint size = INPUT0_SIZE_X * INPUT0_SIZE_Y * INPUT0_FEATURE_NUM;
	uint global_index = b * size + f * INPUT0_SIZE_X * INPUT0_SIZE_Y + y * INPUT0_SIZE_X + x;
	const uint current_batch = get_local_id(1);
	const uint batch_offset = current_batch * size;
	uint local_index = get_local_id(0);
	iav_type accumulator;
	while(global_index < batch_offset){
		global_index += size;
	}
	while(global_index > batch_offset + size)
		global_index -= size;
	accumulator.index = global_index;
	accumulator.value = input[global_index];
#ifdef MAX_OUT
	__attribute__((opencl_unroll_hint))
	while (global_index < size + batch_offset) 
	{
		iav_type element;
		element.value = input[global_index];
		element.index = global_index;

		if(accumulator.value < element.value)
		{
			accumulator.value = element.value;
			accumulator.index = element.index;
		}
		global_index += GLOBAL_SIZE;
	}

	scratch[current_batch][local_index] = accumulator;

	barrier(CLK_LOCAL_MEM_FENCE);

	__attribute__((opencl_unroll_hint))
	for(uint offset = LOCAL_SIZE / 2; offset > 0; offset /= 2) 
	{
		if (local_index < offset) 
		{
			iav_type other = scratch[current_batch][local_index + offset];
			iav_type mine = scratch[current_batch][local_index];

			if(mine.value < other.value)
			{
				scratch[current_batch][local_index] = other;
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (local_index == 0) 
	{
		output[current_batch] = scratch[current_batch][0].index % size;
	}
}
#else
__attribute__((opencl_unroll_hint))
	while (global_index < size + batch_offset) 
	{
		iav_type element;
		element.value = input[global_index];
		element.index = global_index;

		if(accumulator.value > element.value)
		{
			accumulator.value = element.value;
			accumulator.index = element.index;
		}
		global_index += GLOBAL_SIZE;
	}

	scratch[current_batch][local_index] = accumulator;

	barrier(CLK_LOCAL_MEM_FENCE);

	__attribute__((opencl_unroll_hint))
	for(uint offset = LOCAL_SIZE / 2; offset > 0; offset /= 2) 
	{
		if (local_index < offset) 
		{
			iav_type other = scratch[current_batch][local_index + offset];
			iav_type mine = scratch[current_batch][local_index];

			if(mine.value > other.value)
			{
				scratch[current_batch][local_index] = other;
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (local_index == 0) 
	{
		output[current_batch] = scratch[current_batch][0].index % size;
	}
}
#endif