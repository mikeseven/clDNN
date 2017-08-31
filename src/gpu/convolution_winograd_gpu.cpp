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

#include "convolution_inst.h"
#include "primitive_gpu_base.h"
#include "implementation_map.h"
#include "error_handler.h"
#include "kernel_selector_helper.h"
#include "kernel_runner.h"

namespace cldnn { namespace gpu {

struct convolution_winograd_gpu : typed_primitive_gpu_impl<convolution>
{
    using parent = typed_primitive_gpu_impl<convolution>;
    using parent::parent;

protected:
    kernel::kernel_arguments_data get_arguments(typed_primitive_inst<convolution>& instance, int32_t split) const override
    {
        kernel::kernel_arguments_data args = parent::get_arguments(instance, split);

        args.weights = &instance.weights_memory(split);
        args.bias = instance.bias_term() ? &instance.bias_memory(split) : nullptr;

        return args;
    }

    int32_t get_split() const override
    { 
        return _outer.get_split(); 
    }

public:
    static primitive_impl* create(const convolution_node &arg)
    {
        static int kern_idx = 0;
        ++kern_idx;

        const auto& primitive       = arg.get_primitive();
        const auto& input_layout    = arg.input().get_output_layout();
        const auto& weights_layout  = arg.weights(0).get_output_layout();
        const auto& output_layout   = arg.get_output_layout();

        const auto& input_size      = tensor{ input_layout.size.batch[0], input_layout.size.feature[0], align_to(input_layout.size.spatial[0], 4), align_to(input_layout.size.spatial[1] - 2, 8) + 2 };
        const auto& weights_size    = weights_layout.size;
        const auto& output_size     = tensor{ output_layout.size.batch[0], output_layout.size.feature[0], align_to(output_layout.size.spatial[0], 4), align_to(output_layout.size.spatial[1], 8) };

        const auto& split           = primitive->split();
        const auto& stride          = primitive->stride;
        const auto& dilation        = primitive->dilation;
        //const auto& input_offset    = primitive->input_offset;

        CLDNN_ERROR_NOT_EQUAL(arg.id(), "dilation", dilation, "expected value", tensor{ 1 }, "winograd convolution does not support dilation");
        CLDNN_ERROR_NOT_EQUAL(arg.id(), "filter dimesions", std::make_pair(weights_size.spatial[0], weights_size.spatial[1]), "expected values", std::make_pair(4, 3), "winograd convolution should get transformed filter with size 4x3");
        CLDNN_ERROR_NOT_EQUAL(arg.id(), "stride", std::make_pair(stride.spatial[0], stride.spatial[1]), "expected values", std::make_pair(1, 1), "winograd convolution is only supported for stride 1x1");
        CLDNN_ERROR_NOT_EQUAL(arg.id(), "split", split, "required value", 1, "Winograd conv does not support split");
        CLDNN_ERROR_NOT_EQUAL(arg.id(), "weights format", weights_layout.format, "winograd_2x3_s1_weights", format::winograd_2x3_s1_weights, "winograd convolution must have weights in winograd format");

        const auto depthwise_separable_opt = arg.get_depthwise_sep_opt();
        CLDNN_ERROR_NOT_EQUAL(arg.id(), "depth. sep. opt.", depthwise_separable_opt, "required value", false, "Winograd conv does not support depthwise separable run mode");

        const auto actual_split = depthwise_separable_opt ? (decltype(split))1 : split;

        assert(arg.get_output_layout().size.feature[0] / primitive->split() == weights_layout.size.batch[0]);

        constexpr tensor::value_type tile_n = 4; //goes in-depth
        constexpr tensor::value_type tile_m = 8; //goes over flattened x and y

        constexpr tensor::value_type filter_width = 3; //by definition of F(2,3)
        constexpr tensor::value_type filter_height = 3; //by definition of format::winograd_2x3_s1_data (our assumption)
        constexpr tensor::value_type filter_stride = 1; //by definition of format::winograd_2x3_s1_data (our assumption)

        constexpr tensor::value_type input_tile_width = 4;
        constexpr tensor::value_type input_tile_height = 1;
        constexpr tensor::value_type winograd_filter_width = input_tile_width; //by definition of the winograd algorithm
        constexpr tensor::value_type winograd_filter_height = filter_height; //for this format, winograd filter is considered to be a set of 1d filters so its height should remain the same as original filter's

        tensor::value_type nr_tiles_x = output_size.spatial[0] / input_tile_width; //input is already in winograd domain, so simply divide its width by tile's width to get tiles count
        tensor::value_type nr_tiles_y = output_size.spatial[1] / input_tile_height;
        tensor::value_type total_tiles_count = nr_tiles_x * nr_tiles_y;

        kernel_selector::cl_kernel_data kernel_data;

        kernel_data.kernelString = std::make_shared<kernel_selector::kernel_string>();
        {
            auto&& output_pitches = output_layout.get_pitches();
            std::string jits = "";
            jits += "#define INPUT_SIZE_X " + std::to_string(input_size.spatial[0]) + "\n";
            jits += "#define INPUT_SIZE_Y " + std::to_string(input_size.spatial[1]) + "\n";
            jits += "#define INPUT_FEATURE_NUM " + std::to_string(input_size.feature[0]) + "\n";
            jits += "#define OUTPUT_FEATURE_NUM " + std::to_string(output_size.feature[0]) + "\n";
            jits += "#define OUTPUT_PITCH_BATCH " + std::to_string(output_pitches.batch[0]) + "\n";
            jits += "#define OUTPUT_PITCH_FEATURE " + std::to_string(output_pitches.feature[0]) + "\n";
            jits += "#define OUTPUT_PITCH_SIZE_Y " + std::to_string(output_pitches.spatial[1]) + "\n";
            jits += "#define OUTPUT_PITCH_SIZE_X " + std::to_string(output_pitches.spatial[0]) + "\n";
            jits += "#define N " + std::to_string(output_layout.size.feature[0]) + "\n";
            jits += "#define M " + std::to_string(total_tiles_count) + "\n";
            jits += "#define K " + std::to_string(input_size.feature[0] * winograd_filter_height) + "\n";
            jits += "#define UNIT_TYPE " + std::string(input_layout.data_type == data_types::f16 ? "half" : "float") + "\n";
            if (arg.bias_term())
            {
                jits += "#define BIAS_TERM\n";
                //it looks like BIAS_PER_OUTPUT is unsed and incompatible with cldnn's core
                /*if (arg.bias(0).get_output_layout().size.spatial[0] != 1 || arg.bias(0).get_output_layout().size.spatial[1] != 1)
                    jits += "#define BIAS_PER_OUTPUT\n";*/

                auto&& bias_pitches = arg.bias().get_output_layout().get_pitches();
                jits += "#define BIAS_PITCH_SIZE_X " + std::to_string(bias_pitches.spatial[0]) + "\n";
                jits += "#define BIAS_PITCH_SIZE_Y " + std::to_string(bias_pitches.spatial[1]) + "\n";
                jits += "#define BIAS_PITCH_FEATURE " + std::to_string(bias_pitches.feature[0]) + "\n";
            }
            jits += "#define KERNEL(n) n##_" + std::to_string(kern_idx) + "\n";
            kernel_data.kernelString->jit = jits;
        }

        kernel_data.kernelString->str = KernelSelector::KernelBase::get_db().get("convolution_gpu_winograd_2x3_s1")[0];
        kernel_data.kernelString->options = "-cl-mad-enable -cl-std=CL2.0 -cl-fast-relaxed-math";
        kernel_data.kernelString->entry_point = "convolution_gpu_winograd_2x3_s1_" + std::to_string(kern_idx);
        kernel_data.kernelString->batch_compilation = true;

        kernel_data.arguments.push_back({ kernel_selector::kernel_argument_types::INPUT, 0 });
        kernel_data.arguments.push_back({ kernel_selector::kernel_argument_types::WEIGHTS, 0 });
        kernel_data.arguments.push_back({ kernel_selector::kernel_argument_types::OUTPUT, 0 });
        if (arg.bias_term())
            kernel_data.arguments.push_back({ kernel_selector::kernel_argument_types::BIAS, 0 });

        kernel_data.workGroups.global = {
            static_cast<size_t>(output_layout.size.feature[0] / tile_n),
            static_cast<size_t>(nr_tiles_x * nr_tiles_y / tile_m),
            static_cast<size_t>(input_tile_width * input_tile_height * input_size.batch[0])
        };

        kernel_data.workGroups.local = { 8, 1, 1 };
        kernel_data.layerID = "convolution_gpu_winograd_2x3_s1";

        kernel_selector::kernel_data data;
        data.kernelName = "convolution_gpu_winograd_2x3_s1_" + std::to_string(kern_idx);
        data.kernels.push_back(kernel_data);

        return new convolution_winograd_gpu(arg, data);
    }
};

namespace{
    struct attach {
        attach() {
            implementation_map<convolution>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::winograd_2x3_s1_data), convolution_winograd_gpu::create);
            implementation_map<convolution>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::winograd_2x3_s1_data), convolution_winograd_gpu::create);
        }
        ~attach() {}
    };
    attach attach_impl;
}
} }
