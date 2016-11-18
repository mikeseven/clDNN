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
#include "cache/primitive_db.h"

#include <algorithm>


namespace neural
{
// Kernel names.
const std::string kernel_name_max            = "pooling_gpu_max";
const std::string kernel_name_max_offset     = "pooling_gpu_max_offset";
const std::string kernel_name_average        = "pooling_gpu_average";
const std::string kernel_name_average_offset = "pooling_gpu_average_offset";

// GPU engine information helpers.
namespace
{
struct gpu_info_helper : gpu::context_holder
{
    gpu::engine_info get_engine_info() const
    {
        return context()->get_engine_info();
    }
};
}

struct pooling_gpu : is_an_implementation {
    const pooling& _outer;
    struct kernel_data 
    {
        size_t gws0, gws1, gws2; ///< Local work sizes (3D).
        size_t lws0, lws1, lws2; ///< Global work sizes (3D).
        std::string kernel_name;
        bool fp16_unit_used;
    } _kernel_data;
    gpu::kernel _kernel;

    pooling_gpu(const pooling& outer)
        : is_an_implementation(neural::type_id<pooling_gpu>()),
        _outer(outer),
        _kernel_data(set_kernel_data(_outer)),
        _kernel(_kernel_data.kernel_name, get_jit_constants(_outer, _kernel_data))
    {}

    static kernel_data set_kernel_data(const pooling& outer)
    {
        const auto& input_mem  = outer.input_memory(0);  // input
        const auto& output_mem = outer.output_memory(0); // output

        kernel_data kd;

        // Determine global work sizes.
        kd.gws0 = output_mem.argument.size.batch[0] * output_mem.argument.size.feature[0];
        kd.gws1 = output_mem.argument.size.spatial[0];
        kd.gws2 = output_mem.argument.size.spatial[1];

        // Find largest positive local work size that is divider for global work size.
        kd.lws0 = std::min(std::max(kd.gws0, static_cast<size_t>(1)), static_cast<size_t>(32));
        while (kd.gws0 % kd.lws0 != 0)
        {
            --kd.lws0;
        }
        kd.lws1 = 1;
        kd.lws2 = 1;

        kd.fp16_unit_used = false;

        // Check for supported input formats.
        switch (input_mem.argument.format)
        {
        case memory::format::yxfb_f32:
            break;
        case memory::format::yxfb_f16:
            kd.fp16_unit_used = true;
            break;

        default:
            throw std::invalid_argument("Input memory format is not supported.");
        }

        // Select kernel name.
        auto needs_boundary = needs_boundary_check(outer);
        switch (outer.argument.mode)
        {
        case pooling::mode::max:
            kd.kernel_name = needs_boundary ? kernel_name_max_offset : kernel_name_max;
            break;
        case pooling::mode::average:
            kd.kernel_name = needs_boundary ? kernel_name_average_offset : kernel_name_average;
            break;

        default:
            throw std::runtime_error("Unknown pooling mode.");
        }

        return kd;
    }

    // Checks if we need boundary checking in kernel.
    static bool needs_boundary_check(const pooling& outer)
    {
        auto& input_mem = outer.input_memory(0);
        auto& input_offset = outer.argument.input_offset;
        
        if (input_offset.spatial[0] || input_offset.spatial[1])
            return true;

        auto& kernel_size = outer.argument.size;
        auto& stride = outer.argument.stride;

        // If modulo is not 0 that means it is not dividable by stride, so we would go out of boundary.
        auto mod_x = (input_mem.argument.size.spatial[0] - (2 * input_offset.spatial[0]) - kernel_size.spatial[0]) % stride.spatial[0];
        auto mod_y = (input_mem.argument.size.spatial[1] - (2 * input_offset.spatial[1]) - kernel_size.spatial[1]) % stride.spatial[1];

        return mod_x || mod_y;
    }

    static gpu::jit_constants get_jit_constants(const pooling& outer, const kernel_data& data) {
        gpu_info_helper gpu_info;
        auto engine_info = gpu_info.get_engine_info();

        if (!engine_info.supports_fp16 && data.fp16_unit_used)
            throw std::invalid_argument("GPU device does not support half precision floating-point formats (cl_khr_fp16 extension)");

        gpu::jit_constants mem_consts{
            gpu::make_jit_constant("INPUT",             outer.input_memory(0).argument.size),
            gpu::make_jit_constant("OUTPUT",            outer.output_memory(0).argument.size),
            gpu::make_jit_constant("WINDOW",            outer.argument.size),
            gpu::make_jit_constant("STRIDE",            outer.argument.stride),
            gpu::make_jit_constant("INPUT_OFFSET",      outer.argument.input_offset),
            gpu::make_jit_constant("FP16_SUPPORTED",    static_cast<int>(engine_info.supports_fp16)),
            gpu::make_jit_constant("FP16_UNIT_USED",    static_cast<int>(data.fp16_unit_used)),
            gpu::make_jit_constant("UNIT_TYPE",         data.fp16_unit_used ? "half" : "float"),
            gpu::make_jit_constant("UNIT_INIT_VAL_MAX", data.fp16_unit_used ? "-HALF_MAX" : "-FLT_MAX"),
            gpu::make_jit_constant("UNIT_INIT_VAL_AVG", data.fp16_unit_used ? "0.0h" : "0.0f")
        };
        return mem_consts;
    }

    static void implementation(const void *ptr) {
        auto me = static_cast<const pooling_gpu*>(ptr);
        const auto& outer = me->_outer;
        const auto& kd    = me->_kernel_data;

        const auto& input_mem  = outer.input_memory(0);  // input
        const auto& output_mem = outer.output_memory(0); // output

        me->_kernel.run<gpu::input_mem, gpu::output_mem>
            ({ { kd.gws0, kd.gws1, kd.gws2}, { kd.lws0, kd.lws1, kd.lws2 } }, input_mem, output_mem);
    }

    static is_an_implementation *create(pooling &arg) {
        auto& input_arg = arg.input_memory(0).argument;
        auto& input_offset = arg.argument.input_offset;

        auto& input_buffer_size = input_arg.size;
        auto& output_arg = arg.output_memory(0).argument;
        auto& output_buffer_size = output_arg.size;
        auto& output_size = arg.argument.output_size;
        auto& output_offset = arg.argument.output_offset;
        auto& stride = arg.argument.stride;
        auto& window = arg.argument.size;
        auto& padding = arg.argument.padding;

        if (padding::zero != padding)                                      throw std::logic_error("Pooling supports only zero padding.");
        if (input_buffer_size.raw.size() != output_buffer_size.raw.size()) throw std::invalid_argument("Pooling input/output number of dimension does not match.");
        if (stride.raw.size() != output_buffer_size.raw.size())            throw std::invalid_argument("Pooling stride/output number of dimension does not match.");
        if (window.raw.size() != output_buffer_size.raw.size())            throw std::invalid_argument("Pooling window_size/output number of dimension does not match.");
        if (input_arg.format != output_arg.format)                         throw std::invalid_argument("Pooling input/output data format does not match.");
        
        // general formula: output size = (input size - window size) / step + 1
        for (size_t i = 0; i < input_offset.raw.size(); ++i) {
            if (output_buffer_size.raw[i] < output_size.raw[i] + output_offset.raw[i])
                throw std::runtime_error("Pooling output buffer size is to small.");
        }

        return new pooling_gpu(arg);
    }

    task_group work() override { return{ { task{ implementation, this } }, schedule::single }; };

};



namespace
{

    struct attach
    {
        attach()
        {
            auto key_fw = std::make_tuple(engine::gpu, memory::format::yxfb_f32, memory::format::yxfb_f32);
            auto val_fw = pooling_gpu::create;

            implementation_map<pooling>::add(key_fw, val_fw);
        }

        ~attach()
        {
        }
    };

#ifdef __GNUC__
    __attribute__((visibility("default")))
#elif _MSC_VER
#   pragma section(".nn_init$m", read, write)
#endif
    attach attach_impl;

}
}
