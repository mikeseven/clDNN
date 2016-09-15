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

#include "api/neural.h"
#include "memory_utils.h"

int winograd_example(size_t input_sz, size_t output_sz, size_t iFM, size_t oFM, float* Kernels, float* Data, float* bias, float* ref_out, FILE* out);
void runWinograd();

void CopyToMemoryObject(const neural::memory& mem, float* data)
{
	float* p = (float*)mem.get_buffer()->lock();
	for (size_t i = 0; i < mem.count(); i++)
	{
		p[i] = data[i];
	}

	mem.get_buffer()->release();
}

void GenerateImageData(const neural::memory& mem, float* data, size_t input_z)
{
	size_t count = mem.count();

	for (size_t i = 0; i < count / input_z; i++)
	{
		for (size_t z = 0; z < input_z; z++)
		{
			data[z + i* input_z] = i * z / (float)input_z;
		}
	}
}

void GenerateKernelData(const neural::memory& mem, float* data)
{
	size_t count = mem.count();

	for (size_t i = 0; i < count; i++)
	{
		data[i] = 1.0;
	}
}

size_t getMemoryObjectCount(const neural::memory& mem)
{
	return mem.count();
}

void example_convolution_winograd_ref() {
	using namespace neural;

	const uint32_t output_y = 16,
		output_x = 16,
		output_z = 16,
		output_b = 1,    // size of whole output buffer

		input_y = output_y,
		input_x = output_x,
		input_z = 9,
		input_b = 1,    // size of whole input buffer

		stride_y = 1,
		stride_x = 1,
		stride_z = 1,
		stride_b = 1,

		conv_size_y = 3,
		conv_size_x = 3,
		conv_size_ifm = input_z,
		conv_size_ofm = output_z;    // size of convolution window

									 //    const int32_t in_off_y = 0, in_off_x = 0, in_off_z = 0, in_off_b = 0;

	auto eng = engine::gpu;
	auto input = memory::allocate({ eng, memory::format::yxfb_f32 ,{ input_b  ,{ input_y    , input_x }, input_z } });
	auto output = memory::allocate({ eng, memory::format::yxfb_f32,{ output_b ,{ output_y   , output_x }, output_z } });
	auto weights = memory::allocate({ eng, memory::format::oiyx_f32,{ 1        ,{ conv_size_y, conv_size_x },{ conv_size_ofm, conv_size_ifm } } });
	auto biases = memory::allocate({ eng, memory::format::x_f32,{ 1        ,{ { output_z } }              , 1 } });

	float* in_data = new float[getMemoryObjectCount(input.as<const memory&>())];
	float* out_data = new float[getMemoryObjectCount(output.as<const memory&>())];
	float* w_data = new float[getMemoryObjectCount(weights.as<const memory&>())];
	float* b_data = new float[getMemoryObjectCount(biases.as<const memory&>())];

	GenerateImageData(input.as<const memory&>(), in_data, input_z);
	GenerateKernelData(weights.as<const memory&>(), w_data);
	GenerateKernelData(biases.as<const memory&>(), b_data);

	CopyToMemoryObject(input.as<const memory&>(), in_data);
	CopyToMemoryObject(weights.as<const memory&>(), w_data);
	CopyToMemoryObject(biases.as<const memory&>(), b_data);

	/*const memory& memBuffer = output.as<const memory&>();
	float* p = (float*)memBuffer.get_buffer()->lock();

	for (size_t i = 0; i < memBuffer.count(); i++)
	{

	}*/


	convolution::arguments arg(eng, output, { input, weights, biases }, { stride_b,{ stride_y, stride_x }, stride_z }, padding::zero);
	auto conv = convolution::create(arg);
	execute({ conv }).wait();

	const memory& memBuffer = output.as<const memory&>();
	float* p = (float*)memBuffer.get_buffer()->lock();

	winograd_example(input_x, output_x, input_z, output_z, w_data, in_data, b_data, p, NULL);

	int stop = 0;
	system("pause");
}

void example_convolution_winograd_perf()
{
	runWinograd();
}
