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

#include "thread_pool.h"
#include <algorithm>
#include <iostream>
#include <mutex>
#include <cassert>

namespace neural {

nn_thread_worker_pool::nn_thread_worker_pool(uint32_t arg_num_threads)
    : taskcount(0)
    , current_task_id(0) 
    , num_logical_per_physical_core(2)  // deafult number of logical core per physical.  Should be obtained by argument or through OS API
    , thread_batch_size(1) 
    , enable_thread_denom(1)
    , num_threads(arg_num_threads == 0 ? std::thread::hardware_concurrency() : arg_num_threads)
    , current_request(nullptr) 
    , stop(false)
{
    active_threads = num_threads;

    pthread_barrier_init(&br_wake, NULL, num_threads+1);
    pthread_barrier_init(&br_endtasks, NULL, num_threads + 1);

    // creating threads
    for (uint32_t thread_id = 0; thread_id < num_threads; ++thread_id) {
        threads.emplace_back(&nn_thread_worker_pool::process_task, this, thread_id);
    }
  //  std::this_thread::sleep_for(std::chrono::milliseconds(100));
}


nn_thread_worker_pool::~nn_thread_worker_pool() {
    {
        std::lock_guard<std::mutex> ul(mtx_wake);
        stop = true;

        // wake up all thread for terminating
#if !defined _WIN32
        cv_wake.notify_all();
#endif // !_WIN32
    }
    pthread_barrier_wait(&br_wake);
    for(auto &thread : threads) thread.join();

    pthread_barrier_destroy(&br_wake);
    pthread_barrier_destroy(&br_endtasks);
}


void nn_thread_worker_pool::push_job(const task_group& requests) 
{
    if (requests.tasks.size() == 0) return;

    std::unique_lock<std::mutex> ul(mtx_wake);
    current_request = &requests.tasks;
    taskcount  = static_cast<uint32_t>(requests.tasks.size());
    current_task_id = 0;
    enable_thread_denom = requests.schedule==schedule::single ? num_threads : 1;
    thread_batch_size = requests.schedule==schedule::unordered ? 1 : (taskcount+num_threads-1)/num_threads;
    //thread_batch_size = (thread_batch_size-1)/2 + 1;
    //cv_wake.notify_all();

    // waiting when all threads finish the job
    //cv_endtasks.wait(ul, [this] {return (current_task_id >= taskcount && active_threads == 0); });
    pthread_barrier_wait(&br_wake);
    pthread_barrier_wait(&br_endtasks);

    current_request = nullptr;
}



bool nn_thread_worker_pool::is_thread_enable(uint32_t threadId) { return (threadId % enable_thread_denom) == 0; };



// main loop of task processing
void nn_thread_worker_pool::process_task(uint32_t ) 
{
    for (;;) 
    {
//        std::unique_lock<std::mutex> ul(mtx_wake);
//        --active_threads;
//        // notify thread pool that thread, probably, is going to sleep
//        cv_endtasks.notify_one();
//        // thread waiting for events: new job (in that case current_task_id < taskcount) or  terminate (stop -> true) 
//        // is_thread_enable allow to enable only one thread per physical CPU if it's needed
//#ifndef _WIN32
//        cv_wake.wait(ul, [this, threadId] { return (is_thread_enable(threadId) && current_task_id < taskcount) || stop; });
//#else
//        do {
//            cv_wake.wait_for(ul, std::chrono::milliseconds(10), [this, threadId] { return (is_thread_enable(threadId) && current_task_id < taskcount) || stop; });
//        } while (!((is_thread_enable(threadId) && current_task_id < taskcount) || stop));
//#endif // _WIN32
//        ++active_threads;
//        ul.unlock();
        pthread_barrier_wait(&br_wake);

        if (stop) return;

        for (;;) 
        {
            const size_t nextTaskId = current_task_id.fetch_add(thread_batch_size); // get index of first element for processing by increasing atomic variable by value of task count per thread
            if(nextTaskId >= taskcount + thread_batch_size - 1) break;              // exit in case when all tasks done
            size_t endTaskId = nextTaskId + thread_batch_size;                      // calculate index last element for processing
            endTaskId = endTaskId > taskcount ? taskcount : endTaskId;              // check bound
            for(size_t i = nextTaskId; i < endTaskId; ++i)
                (*current_request)[i].callback((*current_request)[i].data);         // task processing 
        }
        pthread_barrier_wait(&br_endtasks);
    }
}

} // namespace neural
