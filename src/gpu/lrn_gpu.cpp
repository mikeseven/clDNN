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

const std::string kernelName = "lrn_GPU";
const std::string kernelCode = R"__krnl(
KERNEL (lrn_GPU)(__global float* input, __global float* output)
{
    const uint global_id = get_global_id(0);
    const uint batch_offset = global_id % INPUT_BATCH_NUM;
    const uint ifm_offset = (global_id / INPUT_BATCH_NUM) % INPUT_FEATURE_NUM;
    const uint x = (global_id / INPUT_BATCH_NUM) / INPUT_FEATURE_NUM;

    float acc = 0;

	int input_offset_f = ifm_offset + HELP_INPUT_OFFSET;
	int input_idx = batch_offset + INPUT_BATCH_NUM * ( input_offset_f + x * INPUT_FEATURE_NUM);
    for (int i = 0; i < P_SIZE; i++)
    {
        bool zero = input_offset_f < 0 || input_offset_f >= INPUT_FEATURE_NUM;

        float value = zero ? 0 : input[input_idx];
        acc = mad(value, value, acc);
		input_offset_f++;
		input_idx += INPUT_BATCH_NUM;
    }
    acc = mad(acc, ALPHA, K);
    acc = pow(acc, -BETA);

    output[global_id] = acc * input[global_id];
}
)__krnl";

namespace neural {
struct lrn_gpu : is_an_implementation {
    normalization::response& outer;
        gpu::kernel _kernel;

    lrn_gpu(normalization::response &arg): is_an_implementation(neural::type_id<lrn_gpu>())
        , outer(arg)
        , _kernel(kernelName, get_jit_constants())
    {}

    gpu::jit_constants get_jit_constants() const {
        auto size = outer.argument.size;

        gpu::jit_constants mem_consts{
            gpu::make_jit_constant("INPUT", outer.input_memory(0).argument.size),
            gpu::make_jit_constant("P_SIZE", size),
            gpu::make_jit_constant("ALPHA", outer.argument.alpha),
            gpu::make_jit_constant("BETA", outer.argument.beta),
            gpu::make_jit_constant("K", outer.argument.k),
            gpu::make_jit_constant("HELP_INPUT_OFFSET", outer.argument.input_offset.feature[0] - static_cast<cl_int>(size / 2))
        };

        return mem_consts;
    }

    static void implementation(const void *ptr) {
        auto me = static_cast<const lrn_gpu*>(ptr);
        auto& outer = me->outer;

        auto& input_mem = outer.input_memory(0);
        auto& output_mem = outer.output_memory(0);

        auto padding = outer.argument.padding;

        size_t dstSize = output_mem.count();

        int lws = 32;
        while (dstSize % lws)
        {
            lws--;
        }

        switch (padding) {
        case padding::zero:
        {
            me->_kernel.run<gpu::input_mem, gpu::output_mem>
                ({ dstSize, std::min(dstSize, static_cast<size_t>(lws)) },
                    input_mem,
                    output_mem);
            break;
        }
        default:
            throw std::runtime_error("Unknown padding mode in lrn");
        }
    }


    static is_an_implementation *create(normalization::response &arg) {
        auto input_arg = arg.input_memory(0).argument;
        auto output_arg = arg.output_memory(0).argument;

        if (input_arg.size.raw.size() != output_arg.size.raw.size())
            throw std::runtime_error("lrn input/output number of dimension does not match [iput size=" + std::to_string(input_arg.size.raw.size())
                + ", output size=" + std::to_string(output_arg.size.raw.size()));
        return new lrn_gpu(arg);
    }

    task_group work() override { return{ { task{ implementation, this } }, schedule::single }; }

};

    namespace {
        struct attach {
            attach() {
                gpu::kernel_templates::add(kernelName, kernelCode);
                auto key = std::make_tuple(engine::gpu, memory::format::yxfb_f32, memory::format::yxfb_f32);
                auto val_fw = lrn_gpu::create;

                implementation_map<normalization::response>::add(key, val_fw); 
            }
            ~attach() {}
        };

#ifdef __GNUC__
        __attribute__((visibility("default"))) //todo maybe dll_sym?
#elif _MSC_VER
#   pragma section(".nn_init$m", read, write)
#endif
        attach attach_impl;

    }

}