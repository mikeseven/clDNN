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

#include "multidimensional_counter.h"
#include "lrn_gpu.h"
#include "kernel.h"

const std::string kernelName = "lrn_GPU";
const std::string kernelCode = R"__krnl(
KERNEL (lrn_GPU)(__global neural_memory* input_mem, __global neural_memory* dst_mem, uint pSize, float k, float alpha, float beta, int helper_input_offset_feature)
{
    __global uint* input_size = get_raw(input_mem);
    __global float* input = (__global float*)get_data(input_mem);
    __global float* pDst = (__global float*)get_data(dst_mem);

    const int global_id = get_global_id(0);

    const int batch_num = input_size[0];
    const int batch_offset = global_id % batch_num;

    const int ifm_num = input_size[1];
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

    lrn_gpu::lrn_gpu(normalization::response &arg)
        : is_an_implementation(neural::type_id<lrn_gpu>())
        , outer(arg) {};

    lrn_gpu::~lrn_gpu() {};

    void lrn_gpu::implementation(const void *ptr) {

        auto this_lrn = static_cast<const normalization::response *>(ptr);

        auto& input_offset = this_lrn->argument.input_offset;
        auto& padding = this_lrn->argument.padding;
        auto& size = this_lrn->argument.size;

        auto& k = this_lrn->argument.k;
        auto& alpha = this_lrn->argument.alpha;
        auto& beta = this_lrn->argument.beta;

        auto input_arg  = this_lrn->input_memory(0).argument;
        auto output_arg = this_lrn->output_memory(0).argument;

        if (input_arg.size.raw.size() != output_arg.size.raw.size())
            throw std::runtime_error("lrn input/output number of dimension does not match [iput size=" + std::to_string(input_arg.size.raw.size())
                                     + ", output size=" + std::to_string(output_arg.size.raw.size()));

        vector<int32_t> help_input_offset({ input_offset });
        help_input_offset.feature[0] -= static_cast<int32_t>(size / 2);

        auto& input_mem = this_lrn->input_memory(0);
        auto& output_mem = this_lrn->output_memory(0);
        size_t dstSize = output_mem.count();

        int lws = 16;
        while (dstSize % lws)
        {
            lws--;
        }

        switch (padding) {
        case padding::zero:
        {
            gpu::kernel<gpu::input_mem, gpu::output_mem, cl_uint, cl_float, cl_float, cl_float, cl_int> kernel(kernelName);
            kernel({ dstSize, std::min(dstSize, (size_t)lws) }, input_mem, output_mem, size, k, alpha, beta, help_input_offset.feature[0]);
            break;
        }
        default:
            throw std::runtime_error("Unknown padding mode in lrn");
        }        
    }

    namespace {
        struct attach {
            attach() {
                gpu::kernel_templates::add(kernelName, kernelCode);
                auto key = std::make_tuple(engine::gpu, memory::format::yxfb_f32, memory::format::yxfb_f32);
                auto val_fw = lrn_gpu::create;
                //auto val_bw = lrn_backward_cpu_reference::create;

                lrn_fw_implementation_map::instance().insert({ key, val_fw }); //todo keys should be different
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