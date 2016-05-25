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
#include <atomic>

namespace neural {

class async_execution {
    std::atomic<size_t> _tasks_left;
    const std::vector<primitive> _primitives;
    const std::vector<worker> _workers;
    bool is_lazy() { return !!(_workers[0].engine()&engine::lazy); }
    void start() {
        _tasks_left = _primitives.size();
        auto thread_function = [&]() {
            for(auto &item : _primitives) {
                _workers[0].execute(item.work());
                --_tasks_left;
            }
        };
        std::thread(thread_function).detach();
    }
public:
    async_execution(std::vector<primitive> primitives, std::vector<worker> workers)
        : _tasks_left(0)
        , _primitives(primitives)
        , _workers(workers)
    {
        // validate than all engines are lazy or non-lazy
        for(auto worker : _workers)
            if(is_lazy()!=!!(worker.engine()&engine::lazy)) throw std::runtime_error("async_execution: lazy engines mixed with non-lazy one");

        // for non-lazy - start execution immediately
        if(!is_lazy()) start();
    }

    size_t tasks_left() { return _tasks_left; }

    void wait() {
        if(is_lazy()) {
            // all primitves are in; can do pattern match-replace here

            // start execution
            start();
        }
        // wait for completion
        while(_tasks_left); 
    }
};

async_result execute(std::vector<primitive> primitives, std::vector<worker> workers) {
    if(0==workers.size()) {
        static auto cpu_worker = worker_cpu::create({});
        workers.push_back(cpu_worker);
    }
    assert(1==workers.size()); // currently only one worker at time

    // all workers must lazy or all workers must be non-lazy
    return std::make_shared<async_execution>(primitives, workers);
}


size_t async_result::tasks_left() { return _execution->tasks_left(); }
void async_result::wait() { _execution->wait(); }

} // namespace neural