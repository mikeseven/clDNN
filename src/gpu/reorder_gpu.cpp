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
#include "kd_selector.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace neural {
    const std::string kernelName = "reorder_GPU";
    const std::string kernelName_subtract = "reorder_subtract_GPU";
    const std::string kernelName_subtract_values = "reorder_subtract_values_GPU";
    const std::string kernel_name_1d_convert = "reorder_gpu_1d_convert";
    const std::string kernel_name_1d_convert_subtract = "reorder_gpu_1d_convert_subtract";
    const std::string kernel_name_1d_convert_subtract_values = "reorder_gpu_1d_convert_subtract_values";
    const std::string kernel_name_reorder_padding_bfyx_f32 = "reorder_gpu_padding_bfyx_f32";

    template <>
    struct kd_default_value_selector<neural::gpu::engine_info_internal::architectures>
    {
        static constexpr neural::gpu::engine_info_internal::architectures value = neural::gpu::engine_info_internal::architectures::GEN_UNKNOWN;
    };

    template <>
    struct kd_default_value_selector<neural::gpu::engine_info_internal::configurations>
    {
        static constexpr neural::gpu::engine_info_internal::configurations value = neural::gpu::engine_info_internal::configurations::GT_UNKNOWN;
    };

struct reorder_gpu : is_an_implementation {
    const reorder& _outer;
    gpu::engine_info_internal _engine_info;

    struct kernel_data
    {
        std::string kernel_name;
        bool have_subtraction;
        bool padding_only;
        bool is_flatten;
    } _kernel_data;
    gpu::kernel _kernel;
    gpu::kernel_execution_options _exec_options;

    static kd_selector_t<kernel_data, reorder, kd_optional_selector_t, size_t, neural::gpu::engine_info_internal::architectures, neural::gpu::engine_info_internal::configurations> ks;

    reorder_gpu(reorder &outer)
    : _outer(outer)
    , _engine_info(outer.get_network().get_engine()->get_context()->get_engine_info())
    , _kernel_data(ks.get_kernel(outer, memory::traits(outer.input_memory(0).get_layout()).dimension, _engine_info.architecture, _engine_info.configuration))
    , _kernel(_outer.get_network().get_engine()->get_context(), _kernel_data.kernel_name, get_jit_constants(_outer, _kernel_data), outer.id())
    , _exec_options(get_execution_options())
    {}

    static kernel_data set_kernel_data(const reorder& outer)
    {
        kernel_data kd;

        kd.have_subtraction = outer.have_substract(); //why 'false' when !subtract_per_feature.empty()? maybe rename to 'have_mean'?
        kd.padding_only = (!kd.have_subtraction) && outer.argument.substract_per_feature.empty() &&
            outer.input_memory(0).argument().format == outer.output_memory().argument().format &&
            (outer.input_memory(0).argument().format == memory::format::type::bfyx_f32 || outer.input_memory(0).argument().format == memory::format::type::bfyx_f16) &&
            outer.desc()->output_padding().lower_size().feature[0] == 0 &&
            outer.desc()->output_padding().lower_size().batch[0] == 0 &&
            outer.desc()->output_padding().upper_size().feature[0] == 0 &&
            outer.desc()->output_padding().upper_size().batch[0] == 0;
        kd.is_flatten = (outer.input_memory(0).argument().size.raw.size() != outer.output_memory().argument().size.raw.size());

        return kd;
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
            return "return lpad[1] + pos[1] + (lpad[1] + size[1] + upad[1]) * (lpad[2] + pos[2] + (lpad[2] + size[2] + upad[2]) * (lpad[3] + pos[3] + (lpad[3] + size[3] + upad[3]) * (lpad[0] + pos[0])));";
        case memory::format::type::yxfb_f32:
        case memory::format::type::yxfb_f16:
            return "return lpad[0] + pos[0] + (lpad[0] + size[0] + upad[0]) * (lpad[1] + pos[1] + (lpad[1] + size[1] + upad[1]) * (lpad[2] + pos[2] + (lpad[2] + size[2] + upad[2]) * (lpad[3] + pos[3])));";
        case memory::format::type::fyxb_f32:
        case memory::format::type::fyxb_f16:
            return "return lpad[0] + pos[0] + (lpad[0] + size[0] + upad[0]) * (lpad[2] + pos[2] + (lpad[2] + size[2] + upad[2]) * (lpad[3] + pos[3] + (lpad[3] + size[3] + upad[3]) * (lpad[1] + pos[1])));";
        case memory::format::type::bfyx_f32:
        case memory::format::type::bfyx_f16:
            return "return lpad[2] + pos[2] + (lpad[2] + size[2] + upad[2]) * (lpad[3] + pos[3] + (lpad[3] + size[3] + upad[3]) * (lpad[1] + pos[1] + (lpad[1] + size[1] + upad[1]) * (lpad[0] + pos[0])));";
        case memory::format::type::oiyx_f32:
        case memory::format::type::oiyx_f16:
            return "return lpad[3] + pos[3] + (lpad[3] + size[3] + upad[3]) * (lpad[4] + pos[4] + (lpad[4] + size[4] + upad[4]) * (lpad[2] + pos[2] + (lpad[2] + size[2] + upad[2]) * (lpad[1] + pos[1])));";
        case memory::format::type::yxio_f32:
        case memory::format::type::yxio_f16:
            return "return lpad[1] + pos[1] + (lpad[1] + size[1] + upad[1]) * (lpad[2] + pos[2] + (lpad[2] + size[2] + upad[2]) * (lpad[3] + pos[3] + (lpad[3] + size[3] + upad[3]) * (lpad[4] + pos[4])));";
        case memory::format::type::os_iyx_osv16_f32:
        case memory::format::type::os_iyx_osv16_f16:
            return R"__C(uint _slice_id = pos[1] / 16; \
                        uint _id_in_slice = pos[1] % 16; \
                        return _id_in_slice + 16 * (pos[3] + size[3] * (pos[4] + size[4] * (pos[2] + _slice_id * size[2])));)__C";
        case memory::format::type::bs_xs_xsv8_bsv8_f32:
            return R"__C(uint _b_slice_id = pos[0] / 8; \
                        uint _b_id_in_slice = pos[0] % 8; \
                        uint _x_slice_id = pos[2] / 8; \
                        uint _x_id_in_slice = pos[2] % 8; \
                        return _b_id_in_slice + 8 * (_x_id_in_slice + 8 * _x_slice_id + _b_slice_id * size[2]);)__C";
        case memory::format::type::bs_x_bsv16_f32:
        case memory::format::type::bs_x_bsv16_f16:
            return R"__C(uint _slice_id = pos[0] / 16; \
                        uint _id_in_slice = pos[0] % 16; \
                        return _id_in_slice + 16 * (pos[2] + size[2] * _slice_id);)__C";
        case memory::format::type::bx_f32:
        case memory::format::type::bx_f16:
            return "return lpad[2] + pos[2] + (lpad[2] + size[2] + upad[2]) * (lpad[0] + pos[0]);";
        case memory::format::type::xb_f32:
        case memory::format::type::xb_f16:
            return "return lpad[0] + pos[0] + (lpad[0] + size[0] + upad[0]) * (lpad[2] + pos[2]);";
        // No reorder, only conversion (use simpler 1D kernels for that).
        case memory::format::type::x_f32:
        case memory::format::type::x_f16:
            return "return lpad[2] + pos[2];";

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

    // output idx for flatten
    static std::string get_idx_calculation_flatten(memory::format::type OutType, memory::format::type InType) {
        cldnn::format OutFormat = memory::to_tensor_format(OutType);

        // Flatten cases
        // 0 - batch (b), 1 - feature (f), 2, 3 - spatial (x -> 2, y -> 3)
        switch (OutFormat)
        {
        case cldnn::format::bs_xs_xsv8_bsv8:
            return R"__C(uint _b_slice_id = pos[0] / 8; \
                        uint _b_id_in_slice = pos[0] % 8; \
                        uint _x_slice_id = (pos[2] + size[2] * (pos[3] + size[3] * pos[1])) / 8; \
                        uint _x_id_in_slice = (pos[2] + size[2] * (pos[3] + size[3] * pos[1])) % 8; \
                        return _b_id_in_slice + 8 * (_x_id_in_slice + 8 * _x_slice_id + _b_slice_id * (size[2] * size[3] * size[1]));)__C";
        case cldnn::format::bs_x_bsv16:
            return R"__C(uint _slice_id = pos[0] / 16; \
                        uint _id_in_slice = pos[0] % 16; \
                        return _id_in_slice + 16 * (pos[2] + size[2] * (pos[3] + size[3] * (pos[1] + size[1] * _slice_id)));)__C";
        //equivalent to axis = 1 (feature), end_axis = -1(x) in caffe
        case cldnn::format::bx:
            return "return pos[2] + size[2] * (pos[3] + size[3] * (pos[1] + size[1] * pos[0]));";
        //equivalent to axis = 0 (batch), end_axis = 2(y) in caffe
        case cldnn::format::xb:
            return "return pos[0] + size[0] * ((pos[1] * size[2] * size[3]) + size[1] * (pos[2] + size[2] * pos[3]) / size[2]);";
        //flatten all into one dimension - equivalent to axis = 0 (batch), end_axis = 3(x) in caffe
        case cldnn::format::x:
        {
            std::vector<uint32_t> calcOrder = get_calculation_order(InType);
            return "return pos[" + std::to_string(calcOrder[0]) + "] + size[" + std::to_string(calcOrder[0]) + "] * (pos[" + std::to_string(calcOrder[1]) +
                "] + size[" + std::to_string(calcOrder[1]) + "] * (pos[" + std::to_string(calcOrder[2]) + "] + size[" + std::to_string(calcOrder[2]) +
                "] * pos[" + std::to_string(calcOrder[3]) + "]));";
        }
        default:
            throw std::invalid_argument("This format is not supported in GPU reorder - flatten");
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

    static gpu::jit_constants get_jit_constants(const reorder& outer, const kernel_data& data) {
        auto engine_info = outer.get_network().get_engine()->get_context()->get_engine_info();

        auto& input_mem = outer.input_memory(0);
        auto& output_mem = outer.output_memory();

        auto input_use_half = input_mem.get_layout().data_type == cldnn::data_types::f16;
        auto output_use_half = output_mem.get_layout().data_type == cldnn::data_types::f16;
        int input_output_type_cvt = input_use_half != output_use_half;
        auto lower_padding = outer.desc()->output_padding().lower_size().transform(output_mem.get_layout().size.format, 0);
        auto upper_padding = outer.desc()->output_padding().upper_size().transform(output_mem.get_layout().size.format, 0);

        if (!engine_info.supports_fp16 && (input_use_half || output_use_half))
            throw std::invalid_argument("GPU device does not support half precision floating-point formats (cl_khr_fp16 extension)");

        gpu::jit_constants mem_consts{
            gpu::make_jit_constant("DIMENSIONS", std::to_string(input_mem.argument().size.raw.size())),
            gpu::make_jit_constant("OUT_FORMAT_IMPLEMENTATION", data.is_flatten ? get_idx_calculation_flatten(output_mem.argument().format, input_mem.argument().format) : get_idx_calculation(output_mem.argument().format)),
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
                s << static_cast<uint32_t>(lower_padding.raw[i]) << ", ";
            }
            s << " }";
            mem_consts.add_constant(gpu::make_jit_constant("LOWER_PADDING", s.str()));
        }
        {
            std::stringstream s;
            s << "(uint[]){ ";
            for (uint32_t i = 0; i < output_mem.argument().size.raw.size(); i++)
            {
                s << static_cast<uint32_t>(upper_padding.raw[i]) << ", ";
            }
            s << " }";
            mem_consts.add_constant(gpu::make_jit_constant("UPPER_PADDING", s.str()));
        }

        if (data.padding_only)
        {
            mem_consts.add_constant(gpu::make_jit_constant("INPUT", input_mem.argument().size));
            mem_consts.add_constant(gpu::make_jit_constant("OUTPUT", output_mem.argument().size));
        }
        else if (data.have_subtraction)
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
                    // TODO: get subtract padding from mean_subtract primitive.
                    s << 0/*static_cast<uint32_t>(padding.raw[i])*/ << ", ";
                }
                s << " }";
                mem_consts.add_constant(gpu::make_jit_constant("SUBTRACT_LOWER_PADDING", s.str()));
                mem_consts.add_constant(gpu::make_jit_constant("SUBTRACT_UPPER_PADDING", s.str()));
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
        auto& input_mem = _outer.input_memory(0);
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

        auto& input_mem = _outer.input_memory(0);
        auto& output_mem = _outer.output_memory();

        if (_kernel_data.have_subtraction)
        {
            return me->_kernel.run<gpu::input_mem, gpu::output_mem, gpu::input_mem>
                (me->_exec_options,
                    events,
                    input_mem,
                    output_mem,
                    _outer.input_memory(1));
        }
        else if (_kernel_data.padding_only)
        {
            gpu::kernel_execution_options exec_options{
                {
                    static_cast<size_t>(input_mem.argument().size.batch[0]),
                    static_cast<size_t>(input_mem.argument().size.feature[0]),
                    static_cast<size_t>(align_to(input_mem.argument().size.spatial[1], 32))
                },
                {
                    1, 1, 32
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

reorder_gpu::kernel_data set_default(const reorder& arg)
{
    reorder_gpu::kernel_data kd = reorder_gpu::set_kernel_data(arg);

    if (kd.padding_only)
    {
        kd.kernel_name = kernel_name_reorder_padding_bfyx_f32;
    }
    else
    {
        //if we got values to subtract, then choose apropriate kernel
        if (kd.have_subtraction)
            kd.kernel_name = kernelName_subtract;
        else if (!arg.argument.substract_per_feature.empty())
            kd.kernel_name = kernelName_subtract_values;
        else
            kd.kernel_name = kernelName;
    }

    return kd;
}

reorder_gpu::kernel_data set_default_dim1(const reorder& arg)
{
    reorder_gpu::kernel_data kd = reorder_gpu::set_kernel_data(arg);

    if (kd.have_subtraction)
        kd.kernel_name = kernel_name_1d_convert_subtract;
    else if (!arg.argument.substract_per_feature.empty())
        kd.kernel_name = kernel_name_1d_convert_subtract_values;
    else
        kd.kernel_name = kernel_name_1d_convert;

    return kd;
}

kd_selector_t<reorder_gpu::kernel_data, reorder, kd_optional_selector_t, size_t, neural::gpu::engine_info_internal::architectures, neural::gpu::engine_info_internal::configurations> reorder_gpu::ks = {
    { std::make_tuple(1, gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), set_default_dim1 },
    { std::make_tuple(0, gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), set_default },
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
    attach attach_impl;
}
}