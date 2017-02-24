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
static const std::string kernel_name              = "softmax_gpu";
static const std::string kernel_name_batches      = "softmax_gpu_batches";
static const std::string kernel_name_batches_bx   = "softmax_gpu_batches_bx";
static const std::string kernel_name_batches_bfyx = "softmax_gpu_batches_bfyx";
static const std::string kernel_name_batches_yxfb = "softmax_gpu_batches_yxfb";


namespace normalization
{
struct softmax_gpu : is_an_implementation
{
    const softmax& _outer;
    struct kernel_data
    {
        size_t gws0;
        size_t gws1;
        size_t lws0;
        std::string kernel_name;
        size_t items_num, leftovers, elements_in_batch;
        bool fp16_unit_used;
        bool fp16_supported;
    } _kernel_data;
    gpu::kernel _kernel;


    softmax_gpu(const softmax& outer)
        : _outer(outer),
        _kernel_data(set_kernel_data(_outer)),
        _kernel(_outer.get_network().get_engine()->get_context(), _kernel_data.kernel_name, get_jit_constants(_outer, _kernel_data), _outer.id())
    {}

    static kernel_data set_kernel_data(const softmax& outer)
    {
        auto engine_info = outer.get_network().get_engine()->get_context()->get_engine_info();

        const auto& input_mem  = outer.input_memory(0);  // input
        const auto& output_mem = outer.output_memory(); // output

        kernel_data kd;

        kd.fp16_unit_used      = input_mem.get_layout().data_type == cldnn::data_types::f16;
        kd.fp16_supported      = engine_info.supports_fp16 != 0;
        auto batch_num         = outer.output_memory().get_layout().size.batch[0];
        auto feature_num       = outer.input_memory(0).get_layout().size.feature[0];
        size_t out_buffer_size = output_mem.count();
        auto input_size = input_mem.argument().size;
        
        if (input_mem.argument().format == memory::format::bfyx_f32 ||//floats
            input_mem.argument().format == memory::format::bfyx_f16 ||
            input_mem.argument().format == memory::format::yxfb_f32 ||//halfs
            input_mem.argument().format == memory::format::yxfb_f16)
        {
            kd.elements_in_batch = input_size.spatial[0] * input_size.spatial[1];
            kd.gws0 = align_to(kd.elements_in_batch, 32);
            kd.gws1 = batch_num;
            kd.lws0 = 32;
            kd.items_num = feature_num;
            if(kd.fp16_unit_used == true)
                kd.kernel_name = kernel_name_batches_yxfb;       
            else
                kd.kernel_name = kernel_name_batches_bfyx;             
        }
        else if (batch_num <= 1)
        {
            kd.lws0 = std::min(std::max(out_buffer_size, static_cast<size_t>(1)), static_cast<size_t>(32));
            kd.leftovers = out_buffer_size % kd.lws0;
            kd.gws0 = out_buffer_size - kd.leftovers;
            kd.gws1 = 1;
            kd.items_num = kd.gws0 / kd.lws0;

            kd.kernel_name = kernel_name;
        }
        else if (input_mem.argument().format == memory::format::bx_f32 || 
            input_mem.argument().format == memory::format::bx_f16)
        {
            // We have two units of data per work item in current implementation.
            auto local_mem_per_wi = 2 * (kd.fp16_unit_used ? sizeof(half_t) : sizeof(float));
            // Combining device execution and local memory restrictions to compute maximum possible LWS.
            auto max_lws = std::min(engine_info.max_work_group_size, engine_info.max_local_mem_size / local_mem_per_wi);

            kd.lws0 = 1;
            kd.items_num = out_buffer_size / batch_num;
            // Compute maximum possible LWS that does not exceed device capabilities and optimizes number of global memory reads.
            while ((kd.items_num > 32 || kd.lws0 < kd.items_num) && (2 * kd.lws0 <= max_lws))
            {
                kd.lws0 *= 2;
                kd.items_num /= 2;
            }

            assert((kd.items_num + 1) * kd.lws0 >= (out_buffer_size / batch_num) && "More than 'lws0' items per batch remains! Lws too small?");
            
            kd.gws0 = kd.lws0;
            kd.gws1 = batch_num;
            kd.leftovers = (out_buffer_size / batch_num) % kd.lws0;

            kd.kernel_name = kernel_name_batches_bx;
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
            kd.gws1 = 1;
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

        auto input_size = outer.input_memory(0).argument().size;

        //kernel relies on INPUT_SIZE_X being a number of values per batch, for bfyx format, when spatials == 1,1
        //and actual number of values is stored as fueatures count (squeezenet), swap feature[0] with spatial[0]
        if (input_size.format == cldnn::format::bfyx)
        {
            if (input_size.feature[0] > 1)
                input_size = cldnn::tensor(cldnn::format::bfyx, { input_size.batch[0], input_size.spatial[0], input_size.spatial[1], input_size.feature[0] });
        }

        else if (input_size.format == cldnn::format::yxfb)
        {
            if (input_size.feature[0] > 1)
                input_size = cldnn::tensor(cldnn::format::yxfb, { input_size.spatial[1], input_size.feature[0], input_size.spatial[0], input_size.batch[0] });
        }


        return gpu::jit_constants{
            gpu::make_jit_constant("INPUT",          input_size),
            gpu::make_jit_constant("ITEMS_NUM",      data.items_num),
            gpu::make_jit_constant("LWS",            data.lws0),
            gpu::make_jit_constant("GWS",            data.gws0),
            gpu::make_jit_constant("LEFTOVERS",      data.leftovers),
            gpu::make_jit_constant("FP16_SUPPORTED", static_cast<int>(data.fp16_supported)),
            gpu::make_jit_constant("FP16_UNIT_USED", static_cast<int>(data.fp16_unit_used)),
            gpu::make_jit_constant("UNIT_TYPE",      data.fp16_unit_used ? "half" : "float"),
            gpu::make_jit_constant("UNIT_VAL_MAX",   data.fp16_unit_used ? "HALF_MAX" : "FLT_MAX"),
            gpu::make_jit_constant("UNIT_VAL_ZERO",  data.fp16_unit_used ? "0.0h" : "0.0f"),
            gpu::make_jit_constant("ELEMENTS_NUM",   data.elements_in_batch)
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

        return _kernel.run<gpu::input_mem, gpu::output_mem>({ { kd.gws0, kd.gws1 }, { kd.lws0, 1 } }, events, input_mem, output_mem);
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
            implementation_map<softmax>::add(std::make_tuple(cldnn::engine_types::ocl, memory::format::bx_f16), val_fw);
            implementation_map<softmax>::add(std::make_tuple(cldnn::engine_types::ocl, memory::format::yxfb_f32), val_fw);
            implementation_map<softmax>::add(std::make_tuple(cldnn::engine_types::ocl, memory::format::yxfb_f16), val_fw);
            implementation_map<softmax>::add(std::make_tuple(cldnn::engine_types::ocl, memory::format::bfyx_f32), val_fw);
            implementation_map<softmax>::add(std::make_tuple(cldnn::engine_types::ocl, memory::format::bfyx_f16), val_fw);
        }
        ~attach() {}
    };
}
attach attach_impl;
} // namespace normalization
} // namespace neural
