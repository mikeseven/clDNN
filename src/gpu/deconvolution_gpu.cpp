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

#include "deconvolution_inst.h"
#include "kernel.h"
#include "kd_selector.h"
#include "network_impl.h"
#include "implementation_map.h"

#include <initializer_list>

using namespace cldnn;

namespace neural 
{

static const std::string kernel_name_yxfb_oiyx = "deconvolution_gpu_yxfb_oiyx";
static const std::string kernel_name_yxfb_yxio = "deconvolution_gpu_yxfb_yxio";
static const std::string kernel_name_bfyx_oiyx = "deconvolution_gpu_bfyx_oiyx";
static const std::string kernel_name_bfyx_yxio = "deconvolution_gpu_bfyx_yxio";

template <>
struct kd_default_value_selector<neural::gpu::engine_info_internal::architectures>
{
    static constexpr neural::gpu::engine_info_internal::architectures value = neural::gpu::engine_info_internal::architectures::GEN_UNKNOWN;
};

template <>
struct kd_default_value_selector<gpu::engine_info_internal::configurations>
{
    static constexpr gpu::engine_info_internal::configurations value = gpu::engine_info_internal::configurations::GT_UNKNOWN;
};

struct deconvolution_gpu : primitive_impl
{
    deconvolution_inst &outer;
    gpu::engine_info_internal _engine_info;

    struct kernel_data
    {
        size_t gws0, gws1, gws2;
        size_t lws0, lws1, lws2;
        size_t ofm_per_work_item; // how many output feature maps a single work item compute
        size_t batches_per_work_item; // how many batches will a single work item compute
        size_t block_width, block_height; // used for kernels processing blocks
        size_t prefetch;
        size_t input_block_array_size; ///< Number of elements in array of UNIT_TYPE that must be specified in kernel to store/cache input block.
        size_t input_block_width;      ///< Number of elements in X dimension stored/cached in input block.
        std::string kernel_name;       ///< Name of a kernel/algorithm to execute.
        bool fp16_unit_used;           ///< Value indicating that FP16 half precision floating point type will be used (instead of single precision).
        size_t leftovers;
    } _kernel_data;

    gpu::kernel _kernel;

    static kernel_data set_default(const deconvolution_inst& arg)
    {
        const auto& input_mem = arg.input_memory();  // input
        const auto& output_mem = arg.output_memory(); // output

        auto split = arg.argument.split();
        auto batch_size = output_mem.get_layout().size.batch[0];

        kernel_data kd;

        kd.fp16_unit_used = input_mem.get_layout().data_type == cldnn::data_types::f16;

        kd.gws0 = (output_mem.get_layout().size.feature[0] * batch_size) / split;
        kd.lws0 = std::min(kd.gws0, static_cast<size_t>(32));
        while (kd.gws0 % kd.lws0)
        {
            kd.lws0--;
        }
        kd.gws1 = output_mem.get_layout().size.spatial[0];
        kd.gws2 = output_mem.get_layout().size.spatial[1];
        kd.lws1 = 1;
        kd.lws2 = 1;
        kd.ofm_per_work_item = 1;
        kd.batches_per_work_item = 1;
        kd.block_width = 1;
        kd.block_height = 1;
        kd.prefetch = 0;
        kd.input_block_array_size = 0;
        kd.input_block_width = 0;
        kd.leftovers = 0;
        return kd;
    }

    typedef kd_selector_t<kernel_data, deconvolution_inst, data_types, format::type, data_types, format::type, kd_optional_selector_t, int, neural::gpu::engine_info_internal::architectures, gpu::engine_info_internal::configurations> ks_type;
    static ks_type ks;

    deconvolution_gpu(deconvolution_inst &arg)
        : outer(arg)
        , _engine_info(arg.get_network().get_engine()->get_context()->get_engine_info())
        , _kernel_data(ks.get_kernel(outer,
            outer.input_memory().get_layout().data_type,
            outer.input_memory().get_layout().size.format,
            outer.weights_memory(0).get_layout().data_type,
            outer.weights_memory(0).get_layout().size.format,
            outer.input_memory().get_layout().size.batch[0],
            _engine_info.architecture,
            _engine_info.configuration))
        , _kernel(arg.get_network().get_engine()->get_context(), _kernel_data.kernel_name, get_jit_constants(), outer.id())
    {}

    gpu::jit_constants get_jit_constants() const {

        auto& input_mem = outer.input_memory();
        auto input_offset = outer.desc()->input_offset().transform(input_mem.get_layout().size.format, 0);
        auto& output_mem = outer.output_memory();
        auto output_offset = outer.desc()->output_offset().transform(output_mem.get_layout().size.format, 0);
        auto& output_size = outer.output_memory().get_layout().size;
        auto& filter_mem = outer.weights_memory(0);
        auto split = outer.argument.split();
        auto input_size = outer.input().at(0)->non_padded_output_layout().size;
        cldnn::tensor stride(cldnn::format::yx, { outer.argument.stride.spatial[0], outer.argument.stride.spatial[1] });
        cldnn::padding input_padding = outer.input().at(0)->desc()->output_padding;

        gpu::jit_constants mem_consts{
            gpu::make_jit_constant("INPUT",                     input_size),
            gpu::make_jit_constant("OUTPUT",                    outer.non_padded_output_layout().size),
            gpu::make_jit_constant("STRIDE",                    stride),
            gpu::make_jit_constant("INPUT_OFFSET",              input_offset),
            gpu::make_jit_constant("OUTPUT_OFFSET",             output_offset),
            gpu::make_jit_constant("OUTPUT_LIMIT",              output_size),
            gpu::make_jit_constant("INPUT_PADDING",             input_padding),
            gpu::make_jit_constant("OUTPUT_PADDING",            outer.argument.output_padding),
            gpu::make_jit_constant("FILTER",                    filter_mem.get_layout().size),
            gpu::make_jit_constant("FILTER_ARRAY_NUM",          split),
            gpu::make_jit_constant("FILTER_OUTPUT_FEATURE_NUM", "FILTER_FEATURE_NUM_0"),
            gpu::make_jit_constant("FILTER_INPUT_FEATURE_NUM",  "FILTER_FEATURE_NUM_1"),
            gpu::make_jit_constant("FP16_SUPPORTED",            static_cast<int>(_engine_info.supports_fp16)),
            gpu::make_jit_constant("FP16_UNIT_USED",            static_cast<int>(_kernel_data.fp16_unit_used)),
            gpu::make_jit_constant("UNIT_TYPE",                 _kernel_data.fp16_unit_used ? "half" : "float"),
            gpu::make_jit_constant("UNIT_VAL_ZERO",             _kernel_data.fp16_unit_used ? "0.0h" : "0.0f"),
            gpu::make_jit_constant("RELU",                      static_cast<int>(outer.argument.with_activation)),
            gpu::make_jit_constant("NEGATIVE_SLOPE",            outer.argument.activation_negative_slope),
        };

        return mem_consts;
    }

    cldnn::refcounted_obj_ptr<cldnn::event_impl> execute(const std::vector<cldnn::refcounted_obj_ptr<cldnn::event_impl>>& events) override
    {
        auto me = this;

        auto split = outer.argument.split();

        auto& input_mem = outer.input_memory();
        auto& output_mem = outer.output_memory();
        auto& filter_mem = outer.weights_memory(0);

        if (outer.desc()->padding_filling_value() != 0.0f)
            throw std::invalid_argument("Unknown padding mode in deconvolution.");

        // Check whether all memory elements use the same unit type (FP16 or FP32).
        if (input_mem.get_layout().data_type != output_mem.get_layout().data_type)
            throw std::invalid_argument("Memory format of input is incompatible with memory format of output.");
        if (input_mem.get_layout().data_type != filter_mem.get_layout().data_type)
            throw std::invalid_argument("Memory format of input is incompatible with memory format of filter.");

        auto& kd = me->_kernel_data;

        std::vector<cldnn::refcounted_obj_ptr<cldnn::event_impl>> tmp_events(events);

        // execute kernels
        for (decltype(split) i = 0; i < split; i++) {
            assert(kd.gws0 % kd.lws0 == 0);
            auto event = me->_kernel.run<gpu::input_mem, gpu::output_mem, gpu::input_mem, gpu::input_mem, uint32_t>
                ({ { kd.gws0, kd.gws1, kd.gws2 },{ kd.lws0, kd.lws1, kd.lws2 } },
                    tmp_events,
                    input_mem,
                    output_mem,
                    outer.weights_memory(i), //filters
                    outer.bias_memory(i), //biases
                    i);
            tmp_events.clear();
            tmp_events.emplace_back(event);
        }
        return tmp_events.at(0);
    }

    static primitive_impl* create(deconvolution_inst &arg)
    {
        auto filter_arg = arg.weights_memory(0).get_layout(); //deconvolution filter

        assert(arg.output_memory().get_layout().size.feature[0] / arg.argument.split() == filter_arg.size.feature[0]); // memory::format oixy
        
        switch (filter_arg.fused_format())
        {
        // FP32 (float)
        case fuse(data_types::f32, format::oiyx):
        case fuse(data_types::f32, format::yxio):
            break;
        default:
            throw std::runtime_error("deconvolution weights format unsupported");
        }

        return new deconvolution_gpu(arg);
    }
};

deconvolution_gpu::kernel_data default_oiyx_f32(const deconvolution_inst& arg)
{
    deconvolution_gpu::kernel_data kd = deconvolution_gpu::set_default(arg);
    kd.kernel_name = (arg.input_memory().get_layout().size.format == cldnn::format::bfyx) ? kernel_name_bfyx_oiyx : kernel_name_yxfb_oiyx;
    return kd;
}

deconvolution_gpu::kernel_data default_yxio_f32(const deconvolution_inst& arg)
{
    deconvolution_gpu::kernel_data kd = deconvolution_gpu::set_default(arg);
    kd.kernel_name = (arg.input_memory().get_layout().size.format == cldnn::format::bfyx) ? kernel_name_bfyx_yxio : kernel_name_yxfb_yxio;
    return kd;
}

deconvolution_gpu::ks_type deconvolution_gpu::ks = {
    { std::make_tuple(data_types::f32, format::yxfb, data_types::f32, format::oiyx, 0, gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), default_oiyx_f32 },
    { std::make_tuple(data_types::f32, format::yxfb, data_types::f32, format::yxio, 0, gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), default_yxio_f32 },
    { std::make_tuple(data_types::f32, format::bfyx, data_types::f32, format::oiyx, 0, gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), default_oiyx_f32 },
    { std::make_tuple(data_types::f32, format::bfyx, data_types::f32, format::yxio, 0, gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), default_yxio_f32 },
};

namespace{
    struct attach {
        attach() {
            implementation_map<deconvolution_inst>::add(std::make_tuple(cldnn::engine_types::ocl, data_types::f32, format::yxfb), deconvolution_gpu::create);
            implementation_map<deconvolution_inst>::add(std::make_tuple(cldnn::engine_types::ocl, data_types::f32, format::bfyx), deconvolution_gpu::create);
        }
        ~attach() {}
    };
    attach attach_impl;
}
}
