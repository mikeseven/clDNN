/*
// Copyright (c) 2018 Intel Corporation
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

#include "lookup_table_inst.h"
#include "primitive_gpu_base.h"
#include "implementation_map.h"
#include "error_handler.h"
#include "kernel_selector_helper.h"
#include "kernel_runner.h"

namespace cldnn {
    namespace gpu {

        struct lookup_table_gpu : typed_primitive_gpu_impl<lookup_table>
        {
            using parent = typed_primitive_gpu_impl<lookup_table>;
            using parent::parent;

        protected:

            virtual bool validate(typed_primitive_inst<lookup_table>& instance) const override
            {
                bool res = parent::validate(instance);

                // Check whether all memory elements use the same unit type (FP16 or FP32).
                CLDNN_ERROR_DATA_TYPES_MISMATCH(_outer.id(), "Input memory", instance.input_memory(1).get_layout().data_type, "output memory", instance.output_memory().get_layout().data_type, "");

                return res;
            }

            virtual kernel::kernel_arguments_data get_arguments(typed_primitive_inst<lookup_table>& instance, int32_t) const override
            {
                kernel::kernel_arguments_data args = parent::get_arguments(instance, 0);

                return args;
            }

        public:

            static primitive_impl* create(const lookup_table_node &arg)
            {
                const auto& primitive = arg.get_primitive();
                //const auto& input_layout = arg.input().get_output_layout();

                //const auto& input_size = input_layout.size;

                const auto& axis = primitive->axis;
                const auto& number_of_values = primitive->number_of_values;
                const auto& with_axis = primitive->with_axis;

                auto lookt_params = get_default_params<kernel_selector::lookup_table_params>(arg);
                auto lookt_optional_params = get_default_optional_params<kernel_selector::lookup_table_optional_params>(arg.get_program());

                lookt_params.lookUpTableParams.numberOfValues = number_of_values;
                if (with_axis) {
                    switch (axis)
                    {
                    case 0:
                        lookt_params.lookUpTableParams.lookUpTableAxis = kernel_selector::lookt_axis::BATCH;
                        break;
                    case 1:
                        lookt_params.lookUpTableParams.lookUpTableAxis = kernel_selector::lookt_axis::FEATURE;
                        break;
                    case 2:
                        lookt_params.lookUpTableParams.lookUpTableAxis = kernel_selector::lookt_axis::X;
                        break;
                    case 3:
                        lookt_params.lookUpTableParams.lookUpTableAxis = kernel_selector::lookt_axis::Y;
                        break;
                    default:
                        break;
                    }
                }
                lookt_params.lookUpTableParams.inputIndexes = convert_data_tensor(arg.input2().get_output_layout());
                auto& kernel_selector = kernel_selector::lookup_table_kernel_selector::Instance();

                KernelSelector::KernelsData best_kernels = kernel_selector.GetBestKernels(lookt_params, lookt_optional_params);

                CLDNN_ERROR_BOOL(arg.id(), "Best_kernel.empty()", best_kernels.empty(), "Cannot find a proper kernel with this arguments");

                auto conv = new lookup_table_gpu(arg, best_kernels[0]);

                return conv;
            }
        };

        namespace {
            struct attach {
                attach() {
                    implementation_map<lookup_table>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::yxfb), lookup_table_gpu::create);
                    implementation_map<lookup_table>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::yxfb), lookup_table_gpu::create);
                    implementation_map<lookup_table>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::bfyx), lookup_table_gpu::create);
                    implementation_map<lookup_table>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::bfyx), lookup_table_gpu::create);
                    implementation_map<lookup_table>::add(std::make_tuple(engine_types::ocl, data_types::i8, format::bfyx), lookup_table_gpu::create);
                    implementation_map<lookup_table>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::byxf), lookup_table_gpu::create);
                    implementation_map<lookup_table>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::byxf), lookup_table_gpu::create);
                }
                ~attach() {}
            };
            attach attach_impl;
        }
    }
}
