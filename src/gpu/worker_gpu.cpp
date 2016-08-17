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

namespace neural {

class program_builder : public gpu::context_holder {
    gpu::kernels_cache::program_type _program;
public:
    program_builder(bool profiling_enabled) {
        context()->profiling_enabled(profiling_enabled);
        _program = gpu::kernels_cache::get().get_program(context());
    }

    auto get_profiling_info() const -> decltype(context()->get_profiling_info()) { return context()->get_profiling_info(); }
};

worker_gpu::arguments::arguments()
    : arguments(false)
    {}

worker_gpu::arguments::arguments(bool profiling)
    : profiling_enabled(profiling)
    {}

worker_gpu::worker_gpu(arguments arg)
    : is_a_worker(type_id<neural::worker_gpu>())
    , builder(new program_builder(arg.profiling_enabled))
{}

void worker_gpu::execute(const neural::task_group& requests) const {
    for (auto& task : requests.tasks) {
        task.callback(task.data);
    }
}

std::vector<std::pair<std::string, std::chrono::nanoseconds>> worker_gpu::get_profiling_info() const {
    return builder->get_profiling_info();
}

worker worker_gpu::create(arguments arg) {
    return new worker_gpu(arg);
}

}
