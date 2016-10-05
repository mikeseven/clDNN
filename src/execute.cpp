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

namespace neural 
{

class async_execution 
{
    std::atomic<size_t> _tasks_left;
    const std::vector<primitive> _primitives;
    const std::vector<worker> _workers;

    void start() {
        _tasks_left = _primitives.size();
        auto thread_function = [&]() {
            // here async_execution object exists
            const auto primitives_count = _primitives.size();

            // It is possible that async_execution will be deleted just after '--_task_left;'.
            // It imght occur after last iteration, but 'for()' loop will run again to find out that exit condition was met.
            // Because of this case we cannot use members from async_execution in 'for()' clause itself.
            for(size_t at=0; at<primitives_count; ++at) {
                _workers[0].execute(_primitives[at].work());
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
    }

    ~async_execution() { wait(); }

    size_t tasks_left() { return _tasks_left; }

    void wait() 
	{
        start();
        // wait for completion
        while(_tasks_left) std::this_thread::yield();
    }
};


async_result execute(std::vector<primitive> primitives, std::vector<worker> workers) 
{
    if(0==workers.size()) 
	{
        static auto gpu_worker = worker_gpu::create();
        workers.push_back(gpu_worker);
    }
    assert(1==workers.size()); // currently only one worker at time

    // all workers must lazy or all workers must be non-lazy
    return std::make_shared<async_execution>(primitives, workers);
}


size_t async_result::tasks_left() { return _execution->tasks_left(); }
void async_result::wait() { _execution->wait(); }

} // namespace neural