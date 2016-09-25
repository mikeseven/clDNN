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
#include "multidimensional_counter.h"
#include "implementation_map.h"
#include "kernel.h"
#include "relu_gpu.h"

namespace neural 
{

const char inline_utils_float[] = R"__CC(
#ifdef RELU
#define ACTIVATION(output, input) output = max(input, 0.0f) + NEGATIVE_SLOPE * min(input, 0.0f);
#else
#define ACTIVATION(output, input) output = input;
#endif

#define ACTIVATION_8(_result) \
{ \
        ACTIVATION(_result.s0, _result.s0); \
        ACTIVATION(_result.s1, _result.s1); \
        ACTIVATION(_result.s2, _result.s2); \
        ACTIVATION(_result.s3, _result.s3); \
        ACTIVATION(_result.s4, _result.s4); \
        ACTIVATION(_result.s5, _result.s5); \
        ACTIVATION(_result.s6, _result.s6); \
        ACTIVATION(_result.s7, _result.s7); \
}
)__CC";

const char inline_utils_float_end[] = R"__CC(
#undef ACTIVATION_8
#undef ACTIVATION
)__CC";

const std::string kernelName = "Relu_GPU";
const std::string kernelCode = R"__krnl(
KERNEL(Relu_GPU)(const __global neural_memory* input_mem, __global neural_memory* output_mem)
{
    const __global float* input = (const __global float*)get_data(input_mem);
    __global float* output = (__global float*)get_data(output_mem);
    
    const int global_id = get_global_id(0);
    ACTIVATION(output[global_id], input[global_id]);
}
)__krnl";

struct relu_gpu : is_an_implementation {
    relu &outer;
    gpu::kernel _kernel;

    relu_gpu(relu &arg) : is_an_implementation(neural::type_id<relu_gpu>())
        , outer(arg)
        , _kernel(kernelName, get_jit_constants()) {}

	gpu::jit_constants get_jit_constants() const
	{
		gpu::jit_constants mem_consts
		{
			gpu::make_jit_constant("RELU", ""),
			gpu::make_jit_constant("NEGATIVE_SLOPE", outer.argument.negative_slope),
		};

		return mem_consts;
	}

    static void implementation(const void *ptr) 
	{
        auto me = static_cast<const relu_gpu *>(ptr);
        auto& outer = me->outer;

        auto& input_mem = outer.input_memory(0);
        auto& output_mem = outer.output_memory(0);

        size_t dstSize = output_mem.count();

        int lws = 16;
        while (dstSize % lws)
        {
            lws--;
        }

        me->_kernel.run<gpu::input_mem, gpu::output_mem>
            ({ dstSize, std::min(dstSize, static_cast<size_t>(lws)) }, input_mem, output_mem);
    }

    static is_an_implementation *create(relu &arg) { return new relu_gpu(arg); };
    task_group work() override { return{ { task{ implementation, this } }, schedule::unordered }; };
};


namespace {
struct attach {
    attach() {
        gpu::kernel_templates::add(kernelName, inline_utils_float + kernelCode + inline_utils_float_end);
        auto key_fw = std::make_tuple(engine::gpu, memory::format::yxfb_f32, memory::format::yxfb_f32);
        auto val_fw = relu_gpu::create;

        implementation_map<relu>::add(key_fw, val_fw); //todo keys should be different
    }
    ~attach() {}
};

#ifdef __GNUC__
    __attribute__((visibility("default"))) //todo meybe dll_sym?
#elif _MSC_VER
#   pragma section(".nn_init$m", read, write)
#endif
attach attach_impl;

}
}
