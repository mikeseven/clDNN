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

#include "neural_impl.h"
#include "engine_impl.h"
#include "network_impl.h"
#include "implementation_map.h"
#include "kernel.h"

#include <algorithm>
#include <stdexcept>
#include <string>


namespace neural
{
// Kernel names.
static const std::string kernel_name         = "softmax_gpu";
static const std::string kernel_name_batches = "softmax_gpu_batches";

namespace normalization
{
struct softmax_gpu : is_an_implementation
{
    const softmax& _outer;
    struct kernel_data
    {
        size_t gws0;
        size_t lws0;
        std::string kernel_name;
        size_t items_num, leftovers;
        bool fp16_unit_used;
        bool fp16_supported;
    } _kernel_data;
    gpu::kernel _kernel;


    softmax_gpu(const softmax& outer)
        : _outer(outer),
        _kernel_data(set_kernel_data(_outer)),
        _kernel(_outer.get_network().get_engine()->get_context(), _outer.id(), _kernel_data.kernel_name, get_jit_constants(_outer, _kernel_data))
    {}

    static kernel_data set_kernel_data(const softmax& outer)
    {
        auto engine_info = outer.get_network().get_engine()->get_context()->get_engine_info();

        const auto& input_mem  = outer.input_memory(0);  // input
        const auto& output_mem = outer.output_memory(); // output

        kernel_data kd;

        kd.fp16_unit_used = input_mem.get_layout().data_type == cldnn::data_types::f16;
        kd.fp16_supported = engine_info.supports_fp16 != 0;

        auto batch_num         = outer.output_memory().get_layout().size.batch[0];
        size_t out_buffer_size = output_mem.count();

        if (batch_num != 1 && input_mem.argument().format == memory::format::bx_f32)
        {
            throw std::runtime_error("Softmax not implemented for bx input when batch > 1!");
        }

        if (batch_num <= 1)
        {
            kd.lws0 = std::min(std::max(out_buffer_size, static_cast<size_t>(1)), static_cast<size_t>(32));
            kd.leftovers = out_buffer_size % kd.lws0;
            kd.gws0 = out_buffer_size - kd.leftovers;
            kd.items_num = kd.gws0 / kd.lws0;

            kd.kernel_name = kernel_name;
        }
        else
        {
            // We have two units of data per work item in current implementation.
            auto local_mem_per_wi = 2 * (kd.fp16_unit_used ? sizeof(half_t) : sizeof(float));
            // Combining device execution and local memory restrictions to compute maximum possible LWS.
            auto max_lws = std::min(engine_info.max_work_group_size, engine_info.max_local_mem_size / local_mem_per_wi);

            kd.lws0 = batch_num;
            kd.items_num = out_buffer_size / kd.lws0;
            // Compute maximum possible LWS that does not exceed device capabilities and optimizes number of global memory reads.
            while ((kd.items_num > 32 || kd.lws0 < kd.items_num) && (2 * kd.lws0 <= max_lws))
            {
                kd.lws0 *= 2;
                kd.items_num /= 2;
            }

            kd.gws0 = kd.lws0;
            kd.leftovers = out_buffer_size % kd.lws0;

            kd.kernel_name = kernel_name_batches;
        }

        assert(kd.items_num > 0 && kd.lws0 && kd.gws0 > 0);
        return kd;
    }

    static gpu::jit_constants get_jit_constants(const softmax& outer, const kernel_data& data)
    {
        if (!data.fp16_supported && data.fp16_unit_used)
            throw std::invalid_argument("GPU device does not support half precision floating-point formats (cl_khr_fp16 extension)");

        return gpu::jit_constants{
            gpu::make_jit_constant("INPUT",          outer.input_memory(0).argument().size),
            gpu::make_jit_constant("ITEMS_NUM",      data.items_num),
            gpu::make_jit_constant("LWS",            data.lws0),
            gpu::make_jit_constant("GWS",            data.gws0),
            gpu::make_jit_constant("LEFTOVERS",      data.leftovers),
            gpu::make_jit_constant("FP16_SUPPORTED", static_cast<int>(data.fp16_supported)),
            gpu::make_jit_constant("FP16_UNIT_USED", static_cast<int>(data.fp16_unit_used)),
            gpu::make_jit_constant("UNIT_TYPE",      data.fp16_unit_used ? "half" : "float"),
            gpu::make_jit_constant("UNIT_VAL_MAX",   data.fp16_unit_used ? "HALF_MAX" : "FLT_MAX"),
            gpu::make_jit_constant("UNIT_VAL_ZERO",  data.fp16_unit_used ? "0.0h" : "0.0f")
        };
    }

    cldnn::refcounted_obj_ptr<cldnn::event_impl> execute(const std::vector<cldnn::refcounted_obj_ptr<cldnn::event_impl>>& events) override
    {
        const auto& outer = _outer;
        const auto& kd    = _kernel_data;

        const auto& input_mem  = outer.input_memory(0);  // input
        const auto& output_mem = outer.output_memory(); // output

        assert(1 == output_mem.get_layout().size.feature.size());
        assert(1 == output_mem.get_layout().size.batch.size());

        return _kernel.run<gpu::input_mem, gpu::output_mem>({kd.gws0, kd.lws0}, events, input_mem, output_mem);
    }

    static is_an_implementation *create(softmax &arg) { return new softmax_gpu(arg); };
};

namespace {
struct attach {
    attach() {
        auto val_fw = softmax_gpu::create;
        implementation_map<softmax>::add(std::make_tuple(cldnn::engine_types::ocl, memory::format::xb_f32), val_fw);
        implementation_map<softmax>::add(std::make_tuple(cldnn::engine_types::ocl, memory::format::xb_f16), val_fw);
        implementation_map<softmax>::add(std::make_tuple(cldnn::engine_types::ocl, memory::format::bx_f32), val_fw);
        implementation_map<softmax>::add(std::make_tuple(cldnn::engine_types::ocl, memory::format::yxfb_f32), val_fw);
        implementation_map<softmax>::add(std::make_tuple(cldnn::engine_types::ocl, memory::format::yxfb_f16), val_fw);
        implementation_map<softmax>::add(std::make_tuple(cldnn::engine_types::ocl, memory::format::bfyx_f32), val_fw);
        implementation_map<softmax>::add(std::make_tuple(cldnn::engine_types::ocl, memory::format::bfyx_f16), val_fw);
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
