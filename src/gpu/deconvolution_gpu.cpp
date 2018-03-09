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
#include "primitive_gpu_base.h"
#include "implementation_map.h"
#include "error_handler.h"
#include "kernel_selector_helper.h"

namespace cldnn { namespace gpu {

struct deconvolution_gpu : typed_primitive_gpu_impl<deconvolution>
{
    using parent = typed_primitive_gpu_impl<deconvolution>;
    using parent::parent;

protected:

    // TODO: share it with convolution and fully connected
    virtual bool validate(typed_primitive_inst<deconvolution>& instance) const override
    {
        bool res = parent::validate(instance);

        CLDNN_ERROR_NOT_EQUAL(_outer.id(), "deconvolution filling value", _outer.get_output_layout().data_padding.filling_value(), "padding mode", 0.0f, "Unknown padding mode in deconvolution.");
        // Check whether all memory elements use the same unit type (FP16 or FP32).
        CLDNN_ERROR_DATA_TYPES_MISMATCH(_outer.id(), "Input memory", instance.input_memory().get_layout().data_type, "output memory", instance.output_memory().get_layout().data_type, "");
        CLDNN_ERROR_DATA_TYPES_MISMATCH(_outer.id(), "Input memory", instance.input_memory().get_layout().data_type, "filter memory", instance.weights_memory(0).get_layout().data_type, "");

        return res;
    }

    virtual kernel::kernel_arguments_data get_arguments(typed_primitive_inst<deconvolution>& instance, int32_t split) const override
    {
        kernel::kernel_arguments_data args = parent::get_arguments(instance, split);

        args.weights    = &instance.weights_memory(split);
        args.bias       = instance.bias_term() ? &instance.bias_memory(split) : nullptr;

        return args;
    }

    virtual int32_t get_split() const override
    { 
        return _outer.get_split(); 
    }

public:

    static primitive_impl* create(const deconvolution_node& arg)
    {
        const auto& primitive = arg.get_primitive();
        const auto& weights_layout = arg.weights(0).get_output_layout();

        assert(arg.get_output_layout().size.feature[0] / arg.get_primitive()->split() == weights_layout.size.batch[0]); // memory::format oixy

        switch (weights_layout.fused_format())
        {
            // FP32 (float)
        case fuse(data_types::f32, format::bfyx):
        case fuse(data_types::f32, format::yxfb):
        case fuse(data_types::f16, format::bfyx):
        case fuse(data_types::f16, format::yxfb):
            break;
        default:
            throw std::runtime_error("deconvolution weights format unsupported");
        }

        const auto& weights_size = weights_layout.size;

        const auto& split = primitive->split();
        const auto& stride = primitive->stride;
#if 0 // TODO: support dilation
        const auto& dilation = primitive->dilation;
#else
        const tensor dilation = {0,0,1,1};
#endif
        const auto depthwise_separable_opt = arg.get_depthwise_sep_opt();

        const auto& input_offset = primitive->input_offset;

        assert(arg.get_output_layout().size.feature[0] / primitive->split() == weights_layout.size.batch[0]);

        auto deconv_params = get_weights_bias_default_params<kernel_selector::deconvolution_params>(arg, depthwise_separable_opt ? 1 : split);
        auto deconv_optional_params = get_default_weights_bias_optional_params<kernel_selector::deconvolution_optional_params>(arg.get_program());

        if(primitive->with_activation)
            convert_activation_func_params(primitive, deconv_params);

        deconv_params.deconvParams.depthwiseSeparableOpt = depthwise_separable_opt;

        deconv_params.deconvParams.split = split;
        deconv_params.deconvParams.filterSize = {
            (uint32_t)weights_size.spatial[0],
            (uint32_t)weights_size.spatial[1],
        };

        deconv_params.deconvParams.padding = {
            (uint32_t)std::max(-input_offset.spatial[0], 0),
            (uint32_t)std::max(-input_offset.spatial[1], 0)
        };

        deconv_params.deconvParams.stride = {
            (uint32_t)stride.spatial[0],
            (uint32_t)stride.spatial[1]
        };

        deconv_params.deconvParams.dilation = {
            (uint32_t)dilation.spatial[0],
            (uint32_t)dilation.spatial[1]
        };

        auto& kernel_selector = kernel_selector::deconvolution_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(deconv_params, deconv_optional_params);

        CLDNN_ERROR_BOOL(arg.id(), "Best_kernel.empty()", best_kernels.empty(), "Cannot find a proper kernel with this arguments");

        auto deconv = new deconvolution_gpu(arg, best_kernels[0]);

        return deconv;
    }
};

namespace{
    struct attach {
        attach() {
            implementation_map<deconvolution>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::yxfb), deconvolution_gpu::create);
            implementation_map<deconvolution>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::bfyx), deconvolution_gpu::create);
            implementation_map<deconvolution>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::yxfb), deconvolution_gpu::create);
            implementation_map<deconvolution>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::bfyx), deconvolution_gpu::create);
            implementation_map<deconvolution>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::byxf), deconvolution_gpu::create);
            implementation_map<deconvolution>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::byxf), deconvolution_gpu::create);
        }
        ~attach() {}
    };
    attach attach_impl;
}
} }
