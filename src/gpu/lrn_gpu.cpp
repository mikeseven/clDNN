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
KERNEL (lrn_GPU)(__global neural_memory* input_mem, __global neural_memory* dst_mem, uint pSize, float k, float alpha, float beta, int helper_input_offset_feature)
{
    __global float* input = (__global float*)get_data(input_mem);
    __global float* pDst = (__global float*)get_data(dst_mem);

    const int global_id = get_global_id(0);

    const int batch_num = INPUT_BATCH_NUM;
    const int batch_offset = global_id % batch_num;

    const int ifm_num = INPUT_FEATURE_NUM_0;
    const int ifm_offset = (global_id / batch_num) % ifm_num;

    const int x = (global_id / batch_num) / ifm_num;

    float acc = 0;

    for (int i = 0; i < pSize; i++)
    {
        int input_offset_f = i + ifm_offset + helper_input_offset_feature;
        bool zero = false;
        
        zero = input_offset_f < 0 ? true : zero;
        zero = input_offset_f >= ifm_num ? true : zero;

        int input_idx = input_offset_f * batch_num + x * ifm_num * batch_num + batch_offset;
        
        float value = zero ? 0 : input[input_idx];
        acc += value * value;
        
        //if(i==0)
        //pDst[global_id] = input_offset_f;
    }
    acc = acc * alpha + k;
    acc = pow(acc, -beta);

    pDst[global_id] = acc * input[global_id];
}
)__krnl";

namespace neural {
struct lrn_gpu : is_an_implementation {
    normalization::response& outer;
        gpu::kernel _kernel;

    lrn_gpu(normalization::response &arg): is_an_implementation(neural::type_id<lrn_gpu>())
        , outer(arg)
        , _kernel(kernelName, {gpu::make_jit_constant("INPUT", outer.input_memory(0).argument.size)})
    {}

    static void implementation(const void *ptr) {
        auto me = static_cast<const lrn_gpu*>(ptr);
        auto& outer = me->outer;

        auto& input_mem = outer.input_memory(0);
        auto& output_mem = outer.output_memory(0);

        auto size = outer.argument.size;

        cl_int help_input_offset = outer.argument.input_offset.feature[0] - static_cast<cl_int>(size / 2);

        auto k = outer.argument.k;
        auto alpha = outer.argument.alpha;
        auto beta = outer.argument.beta;

        auto padding = outer.argument.padding;

        size_t dstSize = output_mem.count();

        int lws = 16;
        while (dstSize % lws)
        {
            lws--;
        }

        switch (padding) {
        case padding::zero:
        {
            me->_kernel.run<gpu::input_mem, gpu::output_mem, cl_uint, cl_float, cl_float, cl_float, cl_int>
                ({ dstSize, std::min(dstSize, static_cast<size_t>(lws)) },
                    input_mem,
                    output_mem,
                    size,
                    k,
                    alpha,
                    beta,
                    help_input_offset);
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

                implementation_map<normalization::response>::add(key, val_fw); //todo keys should be different
                //lrn_bw_implementation_map.insert({ key, val_bw });
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