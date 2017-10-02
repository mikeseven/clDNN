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

#include "assign_patch_inst.h"
#include "primitive_gpu_base.h"
#include "implementation_map.h"
#include "error_handler.h"
#include "kernel_selector_helper.h"

namespace cldnn { namespace gpu {

struct assign_patch_gpu : typed_primitive_gpu_impl<assign_patch>
{
    using parent = typed_primitive_gpu_impl<assign_patch>;
    using parent::parent;

    static primitive_impl* create(const assign_patch_node& arg) 
    { 
        auto as_params = get_default_params<kernel_selector::assign_patch_params>(arg);
        auto as_optional_params = get_default_optional_params<kernel_selector::assign_patch_optional_params>(arg.get_program());

        const auto& primitive = arg.get_primitive();
        if (primitive->with_activation)
            convert_activation_func_params(primitive, as_params);

        as_params.inputs.push_back(convert_data_tensor(arg.nn().get_output_layout()));

        auto& kernel_selector = kernel_selector::assign_patch_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(as_params, as_optional_params);

        CLDNN_ERROR_BOOL(arg.id(), "Best_kernel.empty()", best_kernels.empty(), "Cannot find a proper kernel with this arguments");

        auto assign_patch = new assign_patch_gpu(arg, best_kernels[0]);

        return assign_patch;
    }
};

namespace {
    struct attach {
        attach() {
            implementation_map<assign_patch>::add({
                { std::make_tuple(engine_types::ocl, data_types::f32, format::bfyx), assign_patch_gpu::create },
                { std::make_tuple(engine_types::ocl, data_types::f16, format::bfyx), assign_patch_gpu::create }
            });
        }
        ~attach() {}
    };
    attach attach_impl;
}
} }
