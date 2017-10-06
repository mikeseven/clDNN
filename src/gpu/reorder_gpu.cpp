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

#include "reorder_inst.h"
#include "primitive_gpu_base.h"
#include "implementation_map.h"
#include "kernel_selector_helper.h"
#include "error_handler.h"

namespace cldnn { namespace gpu {

struct reorder_gpu : typed_primitive_gpu_impl<reorder>
{
    using parent = typed_primitive_gpu_impl<reorder>;
    using parent::parent;

protected:

    virtual bool optimized_out(reorder_inst& instance) const override
    {
        return
            parent::optimized_out(instance) || _outer.can_be_optimized();
    }

    virtual kernel::kernel_arguments_data get_arguments(reorder_inst& instance, int32_t split) const override
    {
        kernel::kernel_arguments_data args = parent::get_arguments(instance, split);

        if (_outer.has_mean())
        {
            args.bias = &instance.mean_memory();
        }

        return args;
    }

public:

    static primitive_impl* create(const reorder_node& arg)
    {
        //TODO: move this logic to kernel selector
        auto&& input_layout = arg.input().get_output_layout();
        auto&& input_size = input_layout.size;
        auto&& output_layout = arg.get_output_layout();

        static int kern_idx = 0;
        ++kern_idx;

        if (output_layout.format == format::winograd_2x3_s1_data)
        {
            CLDNN_ERROR_NOT_EQUAL(arg.id(), "input format", input_layout.format, "bfyx", format::bfyx, "Conversion to winograd domain is only supported for bfyx input");

            constexpr tensor::value_type filter_width = 3; //by definition of F(2,3)
            constexpr tensor::value_type filter_height = 3; //by definition of format::winograd_2x3_s1_data (our assumption)
            constexpr tensor::value_type filter_stride = 1; //by definition of format::winograd_2x3_s1_data (our assumption)

            constexpr tensor::value_type input_tile_width = 4;
            constexpr tensor::value_type input_tile_height = 1;
            constexpr tensor::value_type winograd_filter_width = input_tile_width; //by definition of the winograd algorithm
            constexpr tensor::value_type winograd_filter_height = filter_height; //for this format, winograd filter is considered to be a set of 1d filters so its height should remain the same as original filter's

            tensor::value_type nr_tiles_x = ceil_div(output_layout.size.spatial[0], winograd_filter_width); //output is already in winograd domain, so simply divide its width by tile's width to get tiles count

            kernel_selector::cl_kernel_data kernel_data;

            kernel_data.kernelString = std::make_shared<kernel_selector::kernel_string>();
            {
                auto&& input_pitches = input_layout.get_pitches();
                auto&& output_pitches = output_layout.get_pitches();
                auto&& input_lower = input_layout.data_padding.lower_size();

                std::string jits = "";
                jits += "#define INPUT_FEATURE_NUM " + std::to_string(input_size.feature[0]) + "\n";
                jits += "#define INPUT_OFFSET_SIZE_X " + std::to_string(arg.get_input_offset().spatial[0]) + "\n";
                jits += "#define INPUT_OFFSET_SIZE_Y " + std::to_string(arg.get_input_offset().spatial[1]) + "\n";
                jits += "#define INPUT_PITCH_SIZE_X " + std::to_string(input_pitches.spatial[0]) + "\n";
                jits += "#define INPUT_PITCH_SIZE_Y " + std::to_string(input_pitches.spatial[1]) + "\n";
                jits += "#define INPUT_PITCH_FEATURE " + std::to_string(input_pitches.feature[0]) + "\n";
                jits += "#define INPUT_PITCH_BATCH " + std::to_string(input_pitches.batch[0]) + "\n";
                jits += "#define INPUT_PADDING_LOWER_SIZE_X " + std::to_string(input_lower.spatial[0]) + "\n";
                jits += "#define INPUT_PADDING_LOWER_SIZE_Y " + std::to_string(input_lower.spatial[1]) + "\n";
                jits += "#define INPUT_PADDING_LOWER_FEATURE " + std::to_string(input_lower.feature[0]) + "\n";
                jits += "#define INPUT_PADDING_LOWER_BATCH " + std::to_string(input_lower.batch[0]) + "\n";

                jits += "#define OUTPUT_PITCH_SIZE_X " + std::to_string(output_pitches.spatial[0]) + "\n";
                jits += "#define OUTPUT_PITCH_SIZE_Y " + std::to_string(output_pitches.spatial[1]) + "\n";
                jits += "#define OUTPUT_PITCH_FEATURE " + std::to_string(output_pitches.feature[0]) + "\n";
                jits += "#define OUTPUT_PITCH_BATCH " + std::to_string(output_pitches.batch[0]) + "\n";

                jits += "#define UNIT_TYPE " + std::string(input_layout.data_type == data_types::f16 ? "half" : "float") + "\n";
                jits += "#define KERNEL(n) n##_" + std::to_string(kern_idx) + "\n";
                kernel_data.kernelString->jit = jits;
            }

            kernel_data.kernelString->str = KernelSelector::KernelBase::get_db().get("reorder_to_winograd_2x3_s1")[0];
            kernel_data.kernelString->options = "-cl-mad-enable -cl-std=CL2.0 -cl-fast-relaxed-math";
            kernel_data.kernelString->entry_point = "reorder_to_winograd_2x3_s1_" + std::to_string(kern_idx);
            kernel_data.kernelString->batch_compilation = true;

            kernel_data.arguments.push_back({ kernel_selector::kernel_argument_types::INPUT, 0 });
            kernel_data.arguments.push_back({ kernel_selector::kernel_argument_types::OUTPUT, 0 });

            kernel_data.workGroups.global = {
                static_cast<size_t>(input_layout.size.feature[0] * input_layout.size.batch[0]),
                static_cast<size_t>(nr_tiles_x),
                static_cast<size_t>(output_layout.size.spatial[1])
            };
            kernel_data.workGroups.local = { input_size.feature[0] > 32 ? 32 : static_cast<size_t>(input_size.feature[0]), 1, 1 };
            kernel_data.layerID = "reorder_to_winograd_2x3_s1";

            kernel_selector::kernel_data data;
            data.kernelName = "reorder_to_winograd_2x3_s1_" + std::to_string(kern_idx);
            data.kernels.push_back(kernel_data);

            return new reorder_gpu(arg, data);
        }

        if (input_layout.format == format::winograd_2x3_s1_data)
        {
            constexpr tensor::value_type filter_width = 3; //by definition of F(2,3)
            constexpr tensor::value_type filter_height = 3; //by definition of format::winograd_2x3_s1_data (our assumption)
            constexpr tensor::value_type filter_stride = 1; //by definition of format::winograd_2x3_s1_data (our assumption)

            constexpr tensor::value_type input_tile_width = 4;
            constexpr tensor::value_type input_tile_height = 1;
            constexpr tensor::value_type winograd_filter_width = input_tile_width; //by definition of the winograd algorithm
            constexpr tensor::value_type winograd_filter_height = filter_height; //for this format, winograd filter is considered to be a set of 1d filters so its height should remain the same as original filter's
            constexpr tensor::value_type output_tile_width = 2; //by definition of F(2,3)

            kernel_selector::cl_kernel_data kernel_data;

            kernel_data.kernelString = std::make_shared<kernel_selector::kernel_string>();
            {
                auto&& input_pitches = input_layout.get_pitches();
                auto&& output_pitches = output_layout.get_pitches();
                auto&& output_lower = output_layout.data_padding.lower_size();

                std::string jits = "";
                jits += "#define INPUT_FEATURE_NUM " + std::to_string(input_size.feature[0]) + "\n";
                jits += "#define INPUT_PITCH_SIZE_X " + std::to_string(input_pitches.spatial[0]) + "\n";
                jits += "#define INPUT_PITCH_SIZE_Y " + std::to_string(input_pitches.spatial[1]) + "\n";
                jits += "#define INPUT_PITCH_FEATURE " + std::to_string(input_pitches.feature[0]) + "\n";
                jits += "#define INPUT_PITCH_BATCH " + std::to_string(input_pitches.batch[0]) + "\n";

                jits += "#define OUTPUT_SIZE_X " + std::to_string(output_layout.size.spatial[0]) + "\n";
                jits += "#define OUTPUT_PITCH_SIZE_X " + std::to_string(output_pitches.spatial[0]) + "\n";
                jits += "#define OUTPUT_PITCH_SIZE_Y " + std::to_string(output_pitches.spatial[1]) + "\n";
                jits += "#define OUTPUT_PITCH_FEATURE " + std::to_string(output_pitches.feature[0]) + "\n";
                jits += "#define OUTPUT_PITCH_BATCH " + std::to_string(output_pitches.batch[0]) + "\n";
                jits += "#define OUTPUT_PADDING_LOWER_SIZE_X " + std::to_string(output_lower.spatial[0]) + "\n";
                jits += "#define OUTPUT_PADDING_LOWER_SIZE_Y " + std::to_string(output_lower.spatial[1]) + "\n";
                jits += "#define OUTPUT_PADDING_LOWER_FEATURE " + std::to_string(output_lower.feature[0]) + "\n";
                jits += "#define OUTPUT_PADDING_LOWER_BATCH " + std::to_string(output_lower.batch[0]) + "\n";

                jits += "#define UNIT_TYPE " + std::string(input_layout.data_type == data_types::f16 ? "half" : "float") + "\n";
                if (output_layout.size.spatial[0] % output_tile_width != 0)
                    jits += "#define LEFTOVERS\n";

                if (arg.get_fused_activation_func() == activation_relu_negative_slope)
                    jits += "#define ACTIVATION(x) ((x) >= 0 ? (x) : ((x)*" + std::to_string(arg.get_fused_activation_params().a) + "))\n";

                jits += "#define KERNEL(n) n##_" + std::to_string(kern_idx) + "\n";
                kernel_data.kernelString->jit = jits;
            }

            kernel_data.kernelString->str = KernelSelector::KernelBase::get_db().get("reorder_from_winograd_2x3_s1")[0];
            kernel_data.kernelString->options = "-cl-mad-enable -cl-std=CL2.0 -cl-fast-relaxed-math";
            kernel_data.kernelString->entry_point = "reorder_from_winograd_2x3_s1_" + std::to_string(kern_idx);
            kernel_data.kernelString->batch_compilation = true;

            kernel_data.arguments.push_back({ kernel_selector::kernel_argument_types::INPUT, 0 });
            kernel_data.arguments.push_back({ kernel_selector::kernel_argument_types::OUTPUT, 0 });

            kernel_data.workGroups.global = {
                static_cast<size_t>(output_layout.size.feature[0] * output_layout.size.batch[0]),
                static_cast<size_t>(output_layout.size.spatial[0] / output_tile_width),
                static_cast<size_t>(output_layout.size.spatial[1])
            };
            kernel_data.workGroups.local = { input_size.feature[0] > 32 ? 32 : static_cast<size_t>(input_size.feature[0]), 1, 1 };
            kernel_data.layerID = "reorder_from_winograd_2x3_s1";

            kernel_selector::kernel_data data;
            data.kernelName = "reorder_from_winograd_2x3_s1_" + std::to_string(kern_idx);
            data.kernels.push_back(kernel_data);

            return new reorder_gpu(arg, data);
        }

        if (output_layout.format == format::winograd_2x3_s1_weights)
        {
            kernel_selector::cl_kernel_data kernel_data;

            kernel_data.kernelString = std::make_shared<kernel_selector::kernel_string>();
            {
                auto&& input_pitches = input_layout.get_pitches();
                auto&& output_pitches = output_layout.get_pitches();

                std::string jits = "";
                jits += "#define INPUT_FEATURE_NUM " + std::to_string(input_size.feature[0]) + "\n";
                jits += "#define INPUT_PITCH_SIZE_X " + std::to_string(input_pitches.spatial[0]) + "\n";
                jits += "#define INPUT_PITCH_SIZE_Y " + std::to_string(input_pitches.spatial[1]) + "\n";
                jits += "#define INPUT_PITCH_FEATURE " + std::to_string(input_pitches.feature[0]) + "\n";
                jits += "#define INPUT_PITCH_BATCH " + std::to_string(input_pitches.batch[0]) + "\n";

                jits += "#define OUTPUT_PITCH_SIZE_X " + std::to_string(output_pitches.spatial[0]) + "\n";
                jits += "#define OUTPUT_PITCH_SIZE_Y " + std::to_string(output_pitches.spatial[1]) + "\n";
                jits += "#define OUTPUT_PITCH_FEATURE " + std::to_string(output_pitches.feature[0]) + "\n";
                jits += "#define OUTPUT_PITCH_BATCH " + std::to_string(output_pitches.batch[0]) + "\n";

                jits += "#define INPUT_TYPE " + std::string(input_layout.data_type == data_types::f16 ? "half" : "float") + "\n";
                jits += "#define OUTPUT_TYPE " + std::string(output_layout.data_type == data_types::f16 ? "half" : "float") + "\n";
                jits += "#define KERNEL(n) n##_" + std::to_string(kern_idx) + "\n";
                kernel_data.kernelString->jit = jits;
            }

            kernel_data.kernelString->str = KernelSelector::KernelBase::get_db().get("reorder_weights_winograd_2x3_s1")[0];
            kernel_data.kernelString->options = "-cl-mad-enable -cl-std=CL2.0 -cl-fast-relaxed-math";
            kernel_data.kernelString->entry_point = "reorder_weights_winograd_2x3_s1_" + std::to_string(kern_idx);
            kernel_data.kernelString->batch_compilation = true;

            kernel_data.arguments.push_back({ kernel_selector::kernel_argument_types::INPUT, 0 });
            kernel_data.arguments.push_back({ kernel_selector::kernel_argument_types::OUTPUT, 0 });

            kernel_data.workGroups.global = {
                1,
                3,
                static_cast<size_t>(input_layout.size.feature[0] * input_layout.size.batch[0])
            };
            kernel_data.workGroups.local = { 1, 1, 32 };
            kernel_data.layerID = "reorder_weights_winograd_2x3_s1";

            kernel_selector::kernel_data data;
            data.kernelName = "reorder_weights_winograd_2x3_s1_" + std::to_string(kern_idx);
            data.kernels.push_back(kernel_data);

            return new reorder_gpu(arg, data);
        }

        auto reorder_params = get_default_params<kernel_selector::reorder_params>(arg);
        auto reorder_optional_params = get_default_optional_params<kernel_selector::reorder_optional_params>(arg.get_program());

        if (arg.has_mean())
        {
            const auto& mean_layout = arg.mean().get_output_layout();
            reorder_params.reorderParams.mean = convert_data_tensor(mean_layout);
            reorder_params.reorderParams.mode = kernel_selector::mean_subtruct_mode::IN_BUFFER;
        }
        else if (arg.get_primitive()->subtract_per_feature.empty() == false)
        {
            reorder_params.reorderParams.mode = kernel_selector::mean_subtruct_mode::INSIDE_PARAMS;
            reorder_params.reorderParams.meanValues = arg.get_primitive()->subtract_per_feature;
        }
        else
        {
            reorder_params.reorderParams.mode = kernel_selector::mean_subtruct_mode::NONE;
        }

        auto& kernel_selector = kernel_selector::reorder_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(reorder_params, reorder_optional_params);

        CLDNN_ERROR_BOOL(arg.id(), "Best_kernel.empty()", best_kernels.empty(), "Cannot find a proper kernel with this arguments");

        auto reorder = new reorder_gpu(arg, best_kernels[0]);

        return reorder;
    }
};

namespace {
    struct attach {
        attach() {
            implementation_map<reorder>::add({
                { engine_types::ocl, reorder_gpu::create }
            });
        }
        ~attach() {}
    };
    attach attach_impl;
}
} }