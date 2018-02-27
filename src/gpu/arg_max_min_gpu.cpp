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

#include "arg_max_min_inst.h"
#include "primitive_gpu_base.h"
#include "implementation_map.h"
#include "error_handler.h"
#include "kernel_selector_helper.h"
#include "kernel_runner.h"

namespace cldnn {
	namespace gpu {

		struct arg_max_min_gpu : typed_primitive_gpu_impl<arg_max_min>
		{
			using parent = typed_primitive_gpu_impl<arg_max_min>;
			using parent::parent;

		protected:

			virtual bool validate(typed_primitive_inst<arg_max_min>& instance) const override
			{
				bool res = parent::validate(instance);

				// Check whether all memory elements use the same unit type (FP16 or FP32).
				CLDNN_ERROR_DATA_TYPES_MISMATCH(_outer.id(), "Input memory", instance.input_memory().get_layout().data_type, "output memory", instance.output_memory().get_layout().data_type, "");

				return res;
			}

			virtual kernel::kernel_arguments_data get_arguments(typed_primitive_inst<arg_max_min>& instance, int32_t) const override
			{
				kernel::kernel_arguments_data args = parent::get_arguments(instance, 0);

				return args;
			}

		public:

			static primitive_impl* create(const arg_max_min_node &arg)
			{
				const auto& primitive = arg.get_primitive();
				//const auto& input_layout = arg.input().get_output_layout();

				//const auto& input_size = input_layout.size;

				const auto& axis = primitive->axis;
				const auto& top_k = primitive->top_k;
				const auto& out_type = primitive->output_type;
				const auto& with_axis = primitive->with_axis;

				auto argm_params = get_default_params<kernel_selector::arg_max_min_params>(arg);
				auto argm_optional_params = get_default_optional_params<kernel_selector::arg_max_min_optional_params>(arg.get_program());

				if (primitive->with_activation)
					convert_activation_func_params(primitive, argm_params);

				argm_params.argMaxParams.topK = top_k;
				if (with_axis) {
					switch (axis)
					{
					case 0:
						argm_params.argMaxParams.argMaxMinAxis = kernel_selector::argm_axis::BATCH;
					case 1:
						argm_params.argMaxParams.argMaxMinAxis = kernel_selector::argm_axis::FEATURE;
					case 2:
						argm_params.argMaxParams.argMaxMinAxis = kernel_selector::argm_axis::X;
					case 3:
						argm_params.argMaxParams.argMaxMinAxis = kernel_selector::argm_axis::Y;
					default:
						break;
					}
				}

				if (out_type == primitive->max)
					argm_params.argMaxParams.argMaxMinOut = kernel_selector::argm_output::MAX;
				else
					argm_params.argMaxParams.argMaxMinOut = kernel_selector::argm_output::MIN;
				auto& kernel_selector = kernel_selector::arg_max_min_kernel_selector::Instance();

				KernelSelector::KernelsData best_kernels = kernel_selector.GetBestKernels(argm_params, argm_optional_params);

				CLDNN_ERROR_BOOL(arg.id(), "Best_kernel.empty()", best_kernels.empty(), "Cannot find a proper kernel with this arguments");

				auto conv = new arg_max_min_gpu(arg, best_kernels[0]);

				return conv;
			}
		};

		namespace {
			struct attach {
				attach() {
					implementation_map<arg_max_min>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::yxfb), arg_max_min_gpu::create);
					implementation_map<arg_max_min>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::yxfb), arg_max_min_gpu::create);
					implementation_map<arg_max_min>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::bfyx), arg_max_min_gpu::create);
					implementation_map<arg_max_min>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::bfyx), arg_max_min_gpu::create);
					implementation_map<arg_max_min>::add(std::make_tuple(engine_types::ocl, data_types::i8, format::bfyx), arg_max_min_gpu::create);
					implementation_map<arg_max_min>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::byxf), arg_max_min_gpu::create);
					implementation_map<arg_max_min>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::byxf), arg_max_min_gpu::create);
				}
				~attach() {}
			};
			attach attach_impl;
		}
	}
}
