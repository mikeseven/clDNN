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

const std::string kernelName = "softmax_gpu";
const std::string kernelCode = R"__krnl(
float FUNC(find_max_value)(const int idx, __global float* input)
{
    __local float partial_max[LWS];
    float value = -FLT_MAX;
    for(int i = 0; i < ITEMS_NUM; i++)
    {
        value = max(value, input[LWS * i + idx]);
    }
    value = max(value, idx < LEFTOVERS? input[LWS * ITEMS_NUM + idx] : -FLT_MAX);
    partial_max[idx] = value;

#if (GWS != LWS) || (LWS > 32) 
    barrier(CLK_LOCAL_MEM_FENCE);
#endif
    if(idx == 0)
    {
        for(int i = 1; i < LWS; i++)
        {
            partial_max[0] = max(partial_max[0], partial_max[i]);
        };
    }
#if (GWS != LWS) || (LWS > 32) 
    barrier(CLK_LOCAL_MEM_FENCE);
#endif
    return partial_max[0];
}

KERNEL (softmax_gpu)(__global neural_memory* input_mem, __global neural_memory* dst_mem)
{
    __global float* input = (__global float*)get_data(input_mem);
    __global float* pDst = (__global float*)get_data(dst_mem);

    const int idx = get_global_id(0);

    const int batch_num = INPUT_BATCH_NUM;
    const int batch_offset = idx % batch_num;

    float max_value = FUNC_CALL(find_max_value)(idx, input);
    
    float tmp_vals[ITEMS_NUM + 1];
    for(int i = 0; i < ITEMS_NUM; i++)
    {
        tmp_vals[i] = exp(input[LWS * i + idx] - max_value);
    }
    tmp_vals[ITEMS_NUM] = idx < LEFTOVERS ? exp(input[LWS * ITEMS_NUM + idx] - max_value) : 0;

    // accumulate all values;
    __local float partial_acc[LWS]; // all values accumulated;
    partial_acc[idx] = 0;
    for(int i = 0; i < ITEMS_NUM + 1; i++)
    {
        partial_acc[idx] += tmp_vals[i];
    }

#if (GWS != LWS) || (LWS > 32) 
    barrier(CLK_LOCAL_MEM_FENCE); // we must be sure that all threads calculated max of elements(we can remove it if simd32 and GWS <= 32
#endif
    if(idx == 0)
    {
        for(int i = 1; i < LWS; i++)
        {
            partial_acc[0] += partial_acc[i];
        }
    }
#if (GWS != LWS) || (LWS > 32) 
    barrier(CLK_LOCAL_MEM_FENCE);
#endif
    for(int i = 0; i < ITEMS_NUM; i++)
    {
        pDst[LWS * i + idx] = tmp_vals[i] / partial_acc[0];
    }
    if(idx < LEFTOVERS)
        pDst[LWS * ITEMS_NUM + idx] = tmp_vals[ITEMS_NUM] / partial_acc[0];
}
)__krnl";

const std::string kernelName2 = "softmax_gpu_batches";
const std::string kernelCodeBatches = R"__krnl(
float FUNC(find_max_value)(const int global_id, const int idx, const int batch_offset, const int batch_num, __global float* input)
{
    __local float partial_max[LWS];
    float value = -FLT_MAX;
    for(int i = 0; i < ITEMS_NUM; i++)
    {
        value = max(value, input[LWS * i + global_id]);
    }
    value = max(value, global_id < LEFTOVERS? input[LWS * ITEMS_NUM + global_id] : -FLT_MAX);
    partial_max[global_id] = value;

#if (GWS != LWS) || (LWS > 32) 
    barrier(CLK_LOCAL_MEM_FENCE);
#endif
    if(global_id < batch_num)
    {
        for(int i = 1; i < LWS / batch_num; i++)
        {
            partial_max[batch_offset] = max(partial_max[0], partial_max[i*batch_num + batch_offset]);
        };
    }
#if (GWS != LWS) || (LWS > 32) 
    barrier(CLK_LOCAL_MEM_FENCE);
#endif
    return partial_max[batch_offset];
}

KERNEL (softmax_gpu_batches)(__global neural_memory* input_mem, __global neural_memory* dst_mem)
{
    __global float* input = (__global float*)get_data(input_mem);
    __global float* pDst = (__global float*)get_data(dst_mem);

    const int batch_num = INPUT_BATCH_NUM;
    const int global_id = get_global_id(0);
    const int idx = global_id / batch_num;

    const int batch_offset = global_id % batch_num;

    const float max_value = FUNC_CALL(find_max_value)(global_id, idx, batch_offset, batch_num, input);

    float tmp_vals[ITEMS_NUM + 1];
    for(int i = 0; i < ITEMS_NUM; i++)
    {
        tmp_vals[i] = exp(input[LWS * i + global_id] - max_value);
    }
    tmp_vals[ITEMS_NUM] = global_id < LEFTOVERS ? exp(input[LWS * ITEMS_NUM + global_id] - max_value) : 0;

    // accumulate all values;
    __local float partial_acc[LWS]; // all values accumulated;
    partial_acc[global_id] = 0;
    for(int i = 0; i < ITEMS_NUM + 1; i++)
    {
        partial_acc[global_id] += tmp_vals[i];
    }

#if (GWS != LWS) || (LWS > 32) 
    barrier(CLK_LOCAL_MEM_FENCE); // we must be sure that all threads calculated max of elements(we can remove it if simd32 and GWS <= 32
#endif
    if(global_id < batch_num)
    {
        for(int i = 1; i < LWS/batch_num; i++)
        {
            partial_acc[batch_offset] += partial_acc[i*batch_num + batch_offset];
        }
    }
#if (GWS != LWS) || (LWS > 32) 
    barrier(CLK_LOCAL_MEM_FENCE);
#endif
    for(int i = 0; i < ITEMS_NUM; i++)
    {
        pDst[LWS * i + global_id] = tmp_vals[i] / partial_acc[batch_offset];
    }
    if(global_id < LEFTOVERS)
        pDst[LWS * ITEMS_NUM + global_id] = tmp_vals[ITEMS_NUM] / partial_acc[batch_offset];
}
)__krnl";

// TODO: read this value from ocl device!!!
#define MAX_LWS 256

namespace neural {
namespace normalization {
struct softmax_gpu : is_an_implementation {
    softmax &outer;
    gpu::kernel _kernel;

    softmax_gpu(softmax &arg) : is_an_implementation(neural::type_id<softmax_gpu>())
        , outer(arg) 
        , _kernel(select_kernel_name(), get_jit_constants())
    {}

    const std::string& select_kernel_name() const {
        return outer.argument.output_size.batch[0] == 1 ? kernelName : kernelName2;
    }

    gpu::jit_constants get_jit_constants() const {
        auto batch_num = outer.argument.output_size.batch[0];

        auto dstSize = outer.output_memory(0).count();

        size_t items_num = 0;
        size_t preferred_lws = 0;
        size_t preferred_gws = 0;
        size_t leftovers = 0;

        if (batch_num == 1)
        {
            preferred_lws = dstSize < 32 ? dstSize : 32;
            preferred_gws = dstSize < 32 ? dstSize : dstSize - (dstSize % preferred_lws);
            items_num = preferred_gws / preferred_lws;
            leftovers = dstSize < 32 ? 0 : dstSize % preferred_lws;
        }
        else
        {
            preferred_lws = batch_num;
            items_num = dstSize / preferred_lws;
            while ( (items_num > 32 || preferred_lws < items_num) && ((preferred_lws << 1) <= MAX_LWS) )
            {
                preferred_lws <<= 1;
                items_num >>= 1;
            }
            preferred_gws = preferred_lws;
            leftovers = dstSize % preferred_lws;
        }

        assert(items_num > 0 && preferred_lws > 0 && preferred_gws > 0 && leftovers > 0);

        return gpu::jit_constants{
            gpu::make_jit_constant("INPUT", outer.input_memory(0).argument.size),
            gpu::make_jit_constant("ITEMS_NUM", std::to_string(items_num)),
            gpu::make_jit_constant("LWS", std::to_string(preferred_lws)),
            gpu::make_jit_constant("GWS", std::to_string(preferred_gws)),
            gpu::make_jit_constant("LEFTOVERS", std::to_string(leftovers))
        };
    }

    static void implementation(const void *ptr) {
        auto me = static_cast<const softmax_gpu*>(ptr);
        auto& outer = me->outer;

        auto& output_size = outer.argument.output_size;

        assert(1 == output_size.feature.size());
        assert(1 == output_size.batch.size());

        auto& input_mem = outer.input_memory(0);
        auto& output_mem = outer.output_memory(0);
        auto dstSize = output_mem.count();

        auto batch_num = output_size.batch[0];

        size_t preferred_lws = 0;
        size_t preferred_gws = 0;

        if (batch_num == 1)
        {
            preferred_lws = dstSize < 32 ? dstSize : 32;
            preferred_gws = dstSize < 32 ? dstSize : dstSize - (dstSize % preferred_lws);
        }
        else
        {
            preferred_lws = batch_num;
            auto items_num = dstSize / preferred_lws;
            while ( (items_num > 32 || preferred_lws < items_num) && ((preferred_lws << 1) <= MAX_LWS) )
            {
                preferred_lws <<= 1;
                items_num >>= 1;
            }
            preferred_gws = preferred_lws;
        }
        me->_kernel.run<gpu::input_mem, gpu::output_mem>
            ({ preferred_gws, preferred_lws }, input_mem, output_mem);
    }

    static is_an_implementation *create(softmax &arg) { return new softmax_gpu(arg); };
    task_group work() override { return{ { task{ implementation, this } }, schedule::single }; };

};

namespace {
struct attach {
    attach() {
        gpu::kernel_templates::add(kernelName, kernelCode);
        gpu::kernel_templates::add(kernelName2, kernelCodeBatches);

        auto key_fw = std::make_tuple(engine::gpu, memory::format::xb_f32, memory::format::xb_f32);
        auto val_fw = softmax_gpu::create;

        implementation_map<softmax>::add(key_fw, val_fw);
    }
    ~attach() {}
};
}

#ifdef __GNUC__
    __attribute__((visibility("default"))) //todo meybe dll_sym?
#elif _MSC_VER
#   pragma section(".nn_init$m", read, write)
#endif
attach attach_impl;

} // namespace normalization
} // namespace neural
