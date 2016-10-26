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
#include "ocl_toolkit.h"
#include "kernels_cache.h"
#include "kernel.h"
#include "cache/primitive_db.h"

namespace neural {

const char warmup_kernel_name[] = "warm_up_gpu";

class program_builder : public gpu::context_holder {

public:
    program_builder() {
        gpu::kernel warmup_kernel(warmup_kernel_name);
        cl::Buffer out(context()->context(), CL_MEM_WRITE_ONLY, 1);
        warmup_kernel.run<cl_int, cl_int, cl_int, cl::Buffer>({ 1024, 8 }, 0, 111, 7, out);
        context()->queue().finish();
    }

    auto get_profiling_info() const -> decltype(context()->get_profiling_info()) { return context()->get_profiling_info(); }
};

worker_gpu::worker_gpu()
    : is_a_worker(type_id<neural::worker_gpu>())
    , builder(new program_builder())
{}

void worker_gpu::execute(const neural::task_group& requests) const {
    for (auto& task : requests.tasks) {
        task.callback(task.data);
    }
}

const std::vector<instrumentation::profiling_info>& worker_gpu::get_profiling_info() const {
    return builder->get_profiling_info();
}

worker worker_gpu::create() {
    return new worker_gpu();
}

namespace {
    struct attach {
        attach() {
			// cache implementation phase #1 that is a initial switch for using primitive database instead of string kernels
			// at later steps primitive database will be created only once per loading library but as for now it would require 
			// large refactor, so it will be done in smaller incremental steps. The same goes for picking first implementation
			// from the returned list.
			gpu::manager::primitive_db database;
            gpu::kernel_templates::add(warmup_kernel_name, database.get(warmup_kernel_name)[0]);
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
