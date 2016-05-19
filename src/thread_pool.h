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

#pragma once

#include <cstdint>
#include <algorithm>
#include <thread>
#include <atomic>
#include <mutex>
#include <queue>
#include <memory>
#include <condition_variable>
#include <cassert>

namespace neural {

/* This file contains internal device structure implementation as well as thread pool class.

Thread pool usage:
    0. Thread pool object is accessible inside internal device implementation.
    1. Create 'job' vector that will contain all requests for thread pool.
    2. Each request should contain callback function that is able to 'unpack' opaque
       request_handle that is also sent inside request.
    3. Specific implementation of opaque internal structure can be made if required.
    4. After 'job' is created, use push_job function.
        - This function blocks execution until all worker threads complete their work.
        - You can't send 'job' that have more requests than there are threads available.
          Such action will result in assertion.
        - This function does not clear nor deallocate job vector, you must do it by yourself.
*/

// Semaphore class used in thread pool implementation.
struct nn_semaphore {
    // Setup semaphore and clear its counter.
    nn_semaphore() : mtx(), mtx_lock(mtx, std::defer_lock), count(0) {}

    // Set semaphore specific value that will be later decremented.
    void set_semaphore(size_t _count) {
        // Thread pool queue should be cleared which is indicated by semaphore value (==0).
        mtx_lock.lock();
        assert(count == 0);
        count = _count;
    }

    // Notifies waiters, decrements semaphore value and atomically clears working state
    // under semaphore lock. This method is called under wake lock also - as should
    // be every place changing request value.
    void atomic_clear_state_and_notify(const task **request) {
        std::unique_lock<std::mutex> lck(mtx);
        *request = nullptr;
        --count;
        cv.notify_one();
    }

    // Waits until tasks count will be equal to zero.
    void wait_for_all_and_clear() {
        cv.wait(mtx_lock, [this]() { return count == 0; });
        mtx_lock.unlock();
    }

    // Wait until any thread is done
    void wait_change() {
        cv.wait(mtx_lock);
    }

private:
    // This mutex is locked so 'count' member can't be accessed in more than one thread.
    std::mutex mtx;
    std::unique_lock<std::mutex> mtx_lock;
    std::condition_variable cv;

    // Number of tasks in pool queue.
    size_t count;
};

// Basic thread worker class.
struct nn_thread_worker {
    nn_thread_worker(uint32_t id, nn_semaphore* semaphore)
        : hypervisor_semaphore(semaphore),
        current_request(nullptr),
        thread_id(id),
        worker_awaken(false),
        close_worker(false),
        worker_thread(&nn_thread_worker::task_loop, this)
    {
        // Wait until OS complete thread creation and call main thread function.
        while (!worker_awaken)
            ;
    }

    ~nn_thread_worker() {
        {
            // Set termination flag and wake up thread. Mutex will make
            // sure it wont happen during current task processing.
            std::lock_guard<std::mutex> wake_lock(wake_mutex);
            close_worker = true;
            wake_condition.notify_one();
        }

        // Wait for termination.
        worker_thread.join();
    }


    // Adds request to the thread.
    void add_request(const task *request) {
        if (request == nullptr) throw std::invalid_argument("null request sent");

        {
            // Locks wake mutex - it will get unlocked after worker
            // thread will cleanup itself after previous work.
            std::lock_guard<std::mutex> wake_lock(wake_mutex);
            current_request = request;
            wake_condition.notify_one();
        }
    }

    // Checks if there is any request processed.
    bool is_ready() const {
        return current_request == nullptr;
    }

private:
    // Main worker thread routine.
    void task_loop() 
    {
        // Aquire wake mutex so no other thread can incorrectly interfere in thread values.
        std::unique_lock<std::mutex> wake_lock(wake_mutex);

        // Let constructor know that thread is fully created and safely locked.
        worker_awaken = true;

        // Main loop.
        while (!close_worker) {
            // Waits for notification that can be caused by new task or destructor.
            // When thread is in waiting state, it removes lock so other threads can push
            // new jobs into it or check its state. But after thread is notified, it locks
            // it again, so no code under this mutex can interfere during task processing.
            // It also causes thread to wait until all other threads changing its state
            // complete their work before thread starts.
            wake_condition.wait(wake_lock);

            // Safety check for spurious wake up or other wake up that shouldnt cause thread
            // to work on data - e.g. it could be termination call.
            bool assign_check;
            {
                assign_check = current_request != nullptr;

                if (assign_check) {
                    // Call user callback with sent request handle.
                    current_request->callback(current_request->data);

                    // Safely clears request state and notifies pool semaphore.
                    hypervisor_semaphore->atomic_clear_state_and_notify(&current_request);
                }
            }
        }
    }

     // Semaphore of thread pool, shader by all threads.
    nn_semaphore* hypervisor_semaphore;

    // Value used both as request handle and indicator of thread state.
    const task* current_request;

    // ID of thread visible by thread pool.
    uint32_t thread_id;

    // Special controlling values.
    volatile bool worker_awaken;
    volatile bool close_worker;

    // Mutexes used by a thread.
    mutable std::mutex wake_mutex;
    std::condition_variable wake_condition;

public:
    // Main object of worker thread.
    std::thread worker_thread;
};

// Thread pool implementation.
struct nn_thread_worker_pool {
    // Basic constructor.
    nn_thread_worker_pool() 
    { 
        uint32_t num_threads = std::thread::hardware_concurrency();
        for (uint32_t thread_id = 0; thread_id < num_threads; ++thread_id) {
            auto thread = std::unique_ptr<nn_thread_worker>(new nn_thread_worker(thread_id, &semaphore));
            threads.push_back(std::move(thread));
        }
    }

    nn_thread_worker_pool(uint32_t arg_num_threads) 
    { 
        uint32_t num_threads = arg_num_threads;
        for (uint32_t thread_id = 0; thread_id < num_threads; ++thread_id) {
            auto thread = std::unique_ptr<nn_thread_worker>(new nn_thread_worker(thread_id, &semaphore));
            threads.push_back(std::move(thread));
        }
    }

    ~nn_thread_worker_pool() {}

    // Get number of worker threads available.
    size_t get_num_threads() {
        // If there are no worker threads, then one thread is available - the pool thread.
        return threads.size()==0 ? 1 : threads.size();
    }

    // Push job queue.
    void push_job(const std::vector<task>& requests) {
        // Sent requests to worker threads.
        if (threads.size() != 0) {
            // Setup semaphore and lock its mutex.
            semaphore.set_semaphore(requests.size());

            auto threads_begin = std::begin(threads);
            auto threads_end = std::end(threads);

            // Run tasks.
            for (auto& request : requests) {
                auto ready_thread = threads_end;

                // Find waiting thread.
                do{
                    ready_thread = std::find_if(threads_begin, threads_end,
                        [&](std::unique_ptr<nn_thread_worker> const& thread) { return thread->is_ready(); });
                    if (ready_thread != threads_end) break;
                    semaphore.wait_change();
                } while (ready_thread == threads_end);

                // It will hit if there are no ready threads.
                assert(ready_thread != threads_end);

                // Add request to ready thread found.
                (**ready_thread).add_request(&request);
            }

            // Wait for all threads and clear semaphore locks.
            semaphore.wait_for_all_and_clear();
        } else {
            // Singlethreaded pool... run tasks sequentially by itself.
            for (auto& request : requests) {
                request.callback(request.data);
            }
        }
    }

private:

    // Main semaphore, visible by all worker threads.
    nn_semaphore semaphore;

    // Vector of worker threads.
    std::vector<std::unique_ptr<nn_thread_worker>> threads;
};

}
