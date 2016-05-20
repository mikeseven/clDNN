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

#include "neural.h"
#include "thread_pool.h"

namespace neural {

void worker_cpu::execute(const std::vector<task>& requests) const {thread_pool->push_job(requests);}

worker_cpu::worker_cpu(neural::worker_cpu::arguments arg)
    : is_a_worker(type_id<neural::worker_cpu>())
    , argument(arg) 
    , thread_pool(new nn_thread_worker_pool(arg.thread_pool_size)) {};

worker_cpu::arguments::arguments(uint32_t arg_threadpool_size)
    : thread_pool_size(arg_threadpool_size) {}

worker_cpu::arguments::arguments()
    : thread_pool_size(0) {}

worker worker_cpu::create(worker_cpu::arguments arg) {
    try {
        auto result = std::unique_ptr<worker_cpu>(new worker_cpu(arg));
        return result.release();
    }
    catch(...) {
        throw std::runtime_error("worker_cpu::create: failed");
    }
}

} // namespace neural


