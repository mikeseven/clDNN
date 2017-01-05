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
#include "implementation_map.h"
#include "kernel.h"
#include "ocl_toolkit.h"
#include "api/memory.hpp"
#include "api/event.hpp"
#include "reorder_arg.h"
#include "engine_impl.h"
#include "network_impl.h"
#include <string>

const std::string kernelName = "reorder_GPU";
const std::string kernelName_subtract = "reorder_subtract_GPU";
const std::string kernelName_subtract_values = "reorder_subtract_values_GPU";
const std::string kernel_name_1d_convert = "reorder_gpu_1d_convert";
const std::string kernel_name_1d_convert_subtract = "reorder_gpu_1d_convert_subtract";
const std::string kernel_name_1d_convert_subtract_values = "reorder_gpu_1d_convert_subtract_values";
const std::string kernel_name_reorder_padding_bfyx_f32 = "reorder_gpu_padding_bfyx_f32";
namespace neural {

struct reorder_gpu : is_an_implementation {
    const reorder& outer;
    bool have_subtraction;
    bool padding_only;
    gpu::kernel _kernel;
    gpu::kernel_execution_options _exec_options;

    reorder_gpu(reorder &arg)
    : outer(arg)
    , have_subtraction(arg.have_substract())
    , _kernel(arg.get_network().get_engine()->get_context(), select_kernel_name(), get_jit_constants())
    , _exec_options(get_execution_options())
    {
        auto& input_mem = outer.input_memory(0);
        auto& output_mem = outer.output_memory();

        padding_only = (!have_subtraction) && (input_mem.argument().format == output_mem.argument().format) && input_mem.argument().format == memory::format::type::bfyx_f32;
    }

    // We need to specify the output idx based on input position
    static std::string get_idx_calculation(memory::format::type type) {
        switch (type)
        {
        // Reorder and optional conversion cases.
        // For input formats:
        // 0 - batch (b), 1 - feature (f), 2, 3 - spatial (x -> 2, y -> 3)
        // For weights formats:
        // 0 - batch (b), 1, 2 - feature (o -> 1, i -> 2), 3, 4 - spatial (x -> 3, y -> 4)
        case memory::format::type::byxf_f32:
        case memory::format::type::byxf_f16:
            return "return pad[1] + pos[1] + (2 * pad[1] + size[1]) * (pad[2] + pos[2] + (2 * pad[2] + size[2]) * (pad[3] + pos[3] + (2 * pad[3] + size[3]) * (pad[0] + pos[0])));";
        case memory::format::type::yxfb_f32:
        case memory::format::type::yxfb_f16:
            return "return pad[0] + pos[0] + (2 * pad[0] + size[0]) * (pad[1] + pos[1] + (2 * pad[1] + size[1]) * (pad[2] + pos[2] + (2 * pad[2] + size[2]) * (pad[3] + pos[3])));";
        case memory::format::type::fyxb_f32:
        case memory::format::type::fyxb_f16:
            return "return pad[0] + pos[0] + (2 * pad[0] + size[0]) * (pad[2] + pos[2] + (2 * pad[2] + size[2]) * (pad[3] + pos[3] + (2 * pad[3] + size[3]) * (pad[1] + pos[1])));";
        case memory::format::type::bfyx_f32:
        case memory::format::type::bfyx_f16:
            return "return pad[2] + pos[2] + (2 * pad[2] + size[2]) * (pad[3] + pos[3] + (2 * pad[3] + size[3]) * (pad[1] + pos[1] + (2 * pad[1] + size[1]) * (pad[0] + pos[0])));";
        case memory::format::type::oiyx_f32:
        case memory::format::type::oiyx_f16:
            return "return pad[3] + pos[3] + (2 * pad[3] + size[3]) * (pad[4] + pos[4] + (2 * pad[4] + size[4]) * (pad[2] + pos[2] + (2 * pad[2] + size[2]) * (pad[1] + pos[1])));";
        case memory::format::type::yxio_f32:
        case memory::format::type::yxio_f16:
            return "return pad[1] + pos[1] + (2 * pad[1] + size[1]) * (pad[2] + pos[2] + (2 * pad[2] + size[2]) * (pad[3] + pos[3] + (2 * pad[3] + size[3]) * (pad[4] + pos[4])));";
        case memory::format::type::os_iyx_osv16_f32:
            return R"__C(uint _slice_id = pos[1] / 16; \
                        uint _id_in_slice = pos[1] % 16; \
                        return _id_in_slice + 16 * (pos[3] + size[3] * (pos[4] + size[4] * (pos[2] + _slice_id * size[2])));)__C";
        case memory::format::type::bx_f32:
        case memory::format::type::bx_f16:
            return "return pad[2] + pos[2] + (2 * pad[2] + size[2]) * (pad[0] + pos[0]);";
        case memory::format::type::xb_f32:
        case memory::format::type::xb_f16:
            return "return pad[0] + pos[0] + (2 * pad[0] + size[0]) * (pad[2] + pos[2]);";
        // No reorder, only conversion (use simpler 1D kernels for that).
        case memory::format::type::x_f32:
        case memory::format::type::x_f16:
            return "return pad[2] + pos[2];";

        default:
            throw std::invalid_argument("This format is not supported in GPU reorder");
        }
    }

    // To read input memory linearly we need to specify the order of reading
    static std::vector<uint32_t> get_calculation_order(memory::format::type type)
    {
        switch(type)
        {
        // Reorder and optional conversion cases.
        // For input formats:
        // 0 - batch (b), 1 - feature (f), 2, 3 - spatial (x -> 2, y -> 3)
        // For weights formats:
        // 0 - batch (b), 1, 2 - feature (o -> 1, i -> 2), 3, 4 - spatial (x -> 3, y -> 4)
        case memory::format::type::byxf_f32:
        case memory::format::type::byxf_f16:
            return { 1, 2, 3, 0 };
        case memory::format::type::yxfb_f32:
        case memory::format::type::yxfb_f16:
            return { 0, 1, 2, 3 };
        case memory::format::type::bfyx_f32:
        case memory::format::type::bfyx_f16:
            return { 2, 3, 1, 0 };
        case memory::format::type::oiyx_f32:
        case memory::format::type::oiyx_f16:
            return { 0, 3, 4, 2, 1 };
        case memory::format::type::yxio_f32:
        case memory::format::type::yxio_f16:
            return { 0, 1, 2, 3, 4 };
        case memory::format::type::fyxb_f32:
        case memory::format::type::fyxb_f16:
            return { 0, 2, 3, 1 };
        case memory::format::type::os_iyx_osv16_f32:
            return { 0, 1, 3, 4, 2 };
        case memory::format::type::bx_f32:
        case memory::format::type::bx_f16:
            return { 1, 2, 0 };
        case memory::format::type::xb_f32:
        case memory::format::type::xb_f16:
            return { 1, 0, 2 };
        // No reorder, only conversion (use simpler 1D kernels for that).
        case memory::format::type::x_f32:
        case memory::format::type::x_f16:
            return { 0, 1, 2 };

        default:
            throw std::invalid_argument("This format is not supported in GPU reorder");
        }
    }

    static std::string get_calculation_order_string(memory::format::type type)
    {
        std::ostringstream os;
        os << "(uint[]){ ";
        for(auto i : get_calculation_order(type)) {
            os << i << ", ";
        }
        os << " }";
        return os.str();
    }

    const std::string& select_kernel_name() const {
        auto& input_mem = outer.input_memory(0);

        auto& output_mem = outer.output_memory();

        bool _padding_only = (!have_subtraction) && (input_mem.argument().format == output_mem.argument().format) && input_mem.argument().format == memory::format::type::bfyx_f32;
        if (_padding_only)
        {
            return kernel_name_reorder_padding_bfyx_f32;
        }

        // 1d conversions (no reorder)
        if (memory::traits(input_mem.get_layout()).dimension == 1)
        {
            if (have_subtraction)
                return kernel_name_1d_convert_subtract;
            if (!outer.argument.substract_per_feature.empty())
                return kernel_name_1d_convert_subtract_values;
            return kernel_name_1d_convert;
        }
        // if we got values to subtract, then choose apropriate kernel
        if (have_subtraction)
            return kernelName_subtract;
        if (!outer.argument.substract_per_feature.empty())
            return kernelName_subtract_values;
        return kernelName;
    }

    gpu::jit_constants get_jit_constants() const {
        auto engine_info = outer.get_network().get_engine()->get_context()->get_engine_info();

        auto& input_mem = outer.input_memory(0);
        auto& output_mem = outer.output_memory();

        auto input_use_half = input_mem.get_layout().data_type == cldnn::data_types::f16;
        auto output_use_half = output_mem.get_layout().data_type == cldnn::data_types::f16;
        int input_output_type_cvt = input_use_half != output_use_half;
        auto padding = outer.desc()->output_offset().transform(output_mem.get_layout().size.format, 0);

        if (!engine_info.supports_fp16 && (input_use_half || output_use_half))
            throw std::invalid_argument("GPU device does not support half precision floating-point formats (cl_khr_fp16 extension)");

        gpu::jit_constants mem_consts{
            gpu::make_jit_constant("DIMENSIONS", std::to_string(input_mem.argument().size.raw.size())),
            gpu::make_jit_constant("OUT_FORMAT_IMPLEMENTATION", get_idx_calculation(output_mem.argument().format)),
            gpu::make_jit_constant("CALCULATION_ORDER", get_calculation_order_string(input_mem.argument().format)),
            gpu::make_jit_constant("SRC_TYPE", input_use_half ? std::string("half") : std::string("float")),
            gpu::make_jit_constant("DEST_TYPE", output_use_half ? std::string("half") : std::string("float")),
            gpu::make_jit_constant("SRC_DEST_TYPE_CVT", input_output_type_cvt),
            gpu::make_jit_constant("FP16_SUPPORTED", static_cast<int>(engine_info.supports_fp16))
        };
        {
            std::stringstream s;
            s << "(uint[]){ ";
            for (uint32_t i = 0; i < input_mem.argument().size.raw.size(); i++)
            {
                s << static_cast<uint32_t>(input_mem.argument().size.raw[i]) << ", ";
            }
            s << " }";
            mem_consts.add_constant(gpu::make_jit_constant("SIZE", s.str()));
        }
        {
            std::stringstream s;
            s << "(uint[]){ ";
            for (uint32_t i = 0; i < output_mem.argument().size.raw.size(); i++)
            {
                s << static_cast<uint32_t>(padding.raw[i]) << ", ";
            }
            s << " }";
            mem_consts.add_constant(gpu::make_jit_constant("PADDING", s.str()));
        }

        bool _padding_only = (!have_subtraction) && (input_mem.argument().format == output_mem.argument().format) && input_mem.argument().format == memory::format::type::bfyx_f32;

        if (_padding_only)
        {
            mem_consts.add_constant(gpu::make_jit_constant("INPUT", input_mem.argument().size));
            mem_consts.add_constant(gpu::make_jit_constant("OUTPUT", output_mem.argument().size));
        }
        else if (have_subtraction)
        {
            auto& subtract_mem = outer.input_memory(1);

            auto subtract_use_half = subtract_mem.get_layout().data_type == cldnn::data_types::f16;
            int subtract_input_type_cvt = subtract_use_half != input_use_half;

            if (!engine_info.supports_fp16 && subtract_use_half)
                throw std::invalid_argument("GPU device does not support half precision floating-point formats (cl_khr_fp16 extension)");

            mem_consts.add_constant(gpu::make_jit_constant("SUBTRACT_FORMAT_IMPLEMENTATION", get_idx_calculation(subtract_mem.argument().format)));
            mem_consts.add_constant(gpu::make_jit_constant("SUBTRACT_TYPE", subtract_use_half ? std::string("half") : std::string("float")));
            mem_consts.add_constant(gpu::make_jit_constant("SUBTRACT_SRC_TYPE_CVT", subtract_input_type_cvt));
            {
                std::stringstream s;
                s << "(uint[]){ ";
                for (uint32_t i = 0; i < subtract_mem.argument().size.raw.size(); i++)
                {
                    s << static_cast<uint32_t>(padding.raw[i]) << ", ";
                }
                s << " }";
                mem_consts.add_constant(gpu::make_jit_constant("SUBTRTACT_PADDING", s.str()));
            }

        }
        else if (!outer.argument.substract_per_feature.empty())
        {
            std::stringstream s;
            s << "(float[]){ ";
            for (uint32_t i = 0; i < outer.argument.substract_per_feature.size(); i++)
            {
                s << outer.argument.substract_per_feature[i] << ", ";
            }
            s << " }";
            mem_consts.add_constant(gpu::make_jit_constant("VALUE_TO_SUBTRACT", s.str()));
            mem_consts.add_constant(gpu::make_jit_constant("SUBTRACT_TYPE", "float"));
            mem_consts.add_constant(gpu::make_jit_constant("SUBTRACT_SRC_TYPE_CVT", input_use_half));
        }

        return mem_consts;
    }

    gpu::kernel_execution_options get_execution_options() const {
        auto& input_mem = outer.input_memory(0);
        auto& input_size_raw = input_mem.argument().size.raw;
        auto dimensions = input_size_raw.size();
        auto order = get_calculation_order(input_mem.argument().format);
        if (dimensions != order.size()) throw std::runtime_error("Reorder number of input dimensions != size of indices order");

        size_t gws_2 = input_size_raw[order[dimensions - 1]];
        size_t gws_1 = input_size_raw[order[dimensions - 2]];
        size_t gws_0 = 1;
        for (size_t i = 0; i < dimensions - 2; i++) {
            gws_0 *= input_size_raw[order[i]];
        }

        return { {gws_0, gws_1, gws_2} };
    }

    cldnn::refcounted_obj_ptr<cldnn::event_impl> execute(const std::vector<cldnn::refcounted_obj_ptr<cldnn::event_impl>>& events) override
    {
        auto me = this;

        auto& input_mem = outer.input_memory(0);
        auto& output_mem = outer.output_memory();

        if (input_mem.argument().size.raw.size() != output_mem.argument().size.raw.size() ||
            memory::traits(input_mem.get_layout()).dimension != memory::traits(output_mem.get_layout()).dimension)
        {
            throw std::runtime_error("Reorder input/output number of dimension does not match.");
        }

        if (me->have_subtraction)
        {
            return me->_kernel.run<gpu::input_mem, gpu::output_mem, gpu::input_mem>
                (me->_exec_options,
                    events,
                    input_mem,
                    output_mem,
                    outer.input_memory(1));
        }
        else if (me->padding_only)
        {
            if (input_mem.argument().size.spatial[1] > 255)
                throw std::runtime_error("We don't support padding reorder with Y > 256");
            gpu::kernel_execution_options exec_options{
                {
                    static_cast<size_t>(input_mem.argument().size.batch[0]),
                    static_cast<size_t>(input_mem.argument().size.feature[0]),
                    static_cast<size_t>(input_mem.argument().size.spatial[1])
                },
                {
                    1, 1, static_cast<size_t>(input_mem.argument().size.spatial[1])
                }
            };
            return me->_kernel.run<gpu::input_mem, gpu::output_mem>
                (exec_options ,
                    events,
                    input_mem,
                    output_mem);
        }
        else
        {
            return me->_kernel.run<gpu::input_mem, gpu::output_mem>
                (me->_exec_options, events,
                    input_mem,
                    output_mem);
        }
    }

    static is_an_implementation *create(reorder &arg) {
        return new reorder_gpu(arg);
    }
};


    namespace {
        struct attach {
            attach() {
                implementation_map<reorder>::add({
                    { cldnn::engine_types::ocl, reorder_gpu::create }
                });
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