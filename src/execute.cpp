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
#include <thread>

namespace neural {

async_result execute(std::vector<primitive> list, worker arg_worker) 
{
    std::shared_ptr<volatile uint32_t> primitive_count(new uint32_t(static_cast<uint32_t>(list.size())));
    //auto thread_function = [](std::vector<primitive> list, worker arg_worker, std::shared_ptr<volatile uint32_t> primitive_count) {
    //    for(auto &item : list) {
    //        arg_worker.execute(item.work());
    //        --*primitive_count;
    //    }
    //};
	*primitive_count = 0;
	for (auto &item : list) 
		arg_worker.execute(item.work());

   // std::thread(thread_function, list, arg_worker, primitive_count).detach();
    return primitive_count;
}

async_result execute(std::vector<primitive> list) {
    static auto default_worker = worker_cpu::create({});
    return execute(list, default_worker);
}

} // namespace neural