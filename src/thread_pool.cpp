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

namespace neural
{

	nn_thread_worker_pool::nn_thread_worker_pool(size_t arg_num_threads)
		:taskcount(0), 
		current_task_id(0), 
		num_logical_per_physical_core(2),			 // deafult number of logical core per physical.  Should be obtained by argument or through OS API
		thread_batch_size(1), 
		enable_thread_denom(1), 
		current_request(nullptr), 
		stop(false)
	{
		unsigned int num_logic_cores = std::thread::hardware_concurrency();
		unsigned int num_threads = arg_num_threads == 0 ? num_logic_cores : static_cast<unsigned int>(arg_num_threads);
		active_threads = num_threads;

		// creating threads
		for (unsigned int thread_id = 0; thread_id < num_threads; ++thread_id)
		{
			std::thread nthread(&nn_thread_worker_pool::process_task, this, thread_id);

			// Setting affinity mask for thread. Will be finished in future
			/*GROUP_AFFINITY ga;
			GROUP_AFFINITY pga;
			auto idx = thread_id;
			ga.Group = idx / 64;
			ga.Mask = 1i64 << (idx % 64);

			auto a = SetThreadGroupAffinity(thread->worker_thread.native_handle(), &ga, &pga);
			auto b = SetThreadAffinityMask(thread->worker_thread.native_handle(), ga.Mask);
			//auto err = GetLastError();
			*/

			threads.push_back(std::move(nthread));
		}
		/*std::unique_lock<std::mutex> ul(mtx_wake);
		cv_endtasks.wait(ul, [this] {return (active_threads == 0); });*/
	}

	nn_thread_worker_pool::~nn_thread_worker_pool()
	{
		{
			std::lock_guard<std::mutex> ul(mtx_wake);
			stop = true;
			// wake up all thread for terminating
			cv_wake.notify_all();
		}
		std::for_each(threads.begin(), threads.end(), std::mem_fn(&std::thread::join));
	}


	void nn_thread_worker_pool::push_job(const task_group& requests)
	{
		{
			std::lock_guard<std::mutex> ul(mtx_wake);
			current_request = &requests.tsk;
			taskcount  = requests.tsk.size();
			current_task_id = 0;
			enable_thread_denom = requests.use_hyper_threading ? 1 : num_logical_per_physical_core;
			thread_batch_size = requests.task_count_per_thread;
			cv_wake.notify_all();
		}

		std::unique_lock<std::mutex> ul(mtx_wake);
		// waiting when all threads finish the job
		cv_endtasks.wait(ul, [this] {return (current_task_id >= taskcount && active_threads == 0); });
		current_request = nullptr;
	}

	inline bool nn_thread_worker_pool::is_thread_enable(uint32_t threadId)
	{
		return (threadId % enable_thread_denom) == 0;
	};

	// main loop of task processing
	void nn_thread_worker_pool::process_task(uint32_t threadId)
	{
		while (true)
		{
			std::unique_lock<std::mutex> ul(mtx_wake);
			active_threads--;
			// notify thread pool that thread, probably, is going to sleep
			cv_endtasks.notify_one();
			// thread waiting for events: new job (in that case current_task_id < taskcount) or  terminate (stop -> true) 
			// is_thread_enable allow to enable only one thread per physical CPU if it's needed
			cv_wake.wait(ul, [this, threadId] { return (is_thread_enable(threadId) && current_task_id < taskcount) || stop; });
			active_threads++;
			ul.unlock();

			if (stop) 
				return;

			while (true)
			{
				size_t nextTaskId = current_task_id.fetch_add(thread_batch_size);   // get index of first element for processing by increasing atomic variable by value of task count per thread
				if (nextTaskId >= taskcount + thread_batch_size - 1)				// exit in case when all tasks done
					break;

				size_t endTaskId = nextTaskId + thread_batch_size;					// calculate index last element for processing
				endTaskId = endTaskId > taskcount ? taskcount : endTaskId;			// check bound
				for (size_t i = nextTaskId; i < endTaskId; ++i)
					(*current_request)[i].callback((*current_request)[i].data);		// task processing 
			}
		}
	}

}