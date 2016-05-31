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

void execute(std::vector<primitive> list, worker& worker) {
    for(auto &item : list)
        worker.execute(item.work());
}

void execute(std::vector<primitive> list)
{
    static nn_thread_worker_pool thread_pool_default;
    for(auto &item : list)
        thread_pool_default.push_job(item.work());
}

};