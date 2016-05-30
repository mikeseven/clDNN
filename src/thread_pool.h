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

#include <thread>
#include <mutex>
#include <cassert>
#include <vector>
#include <atomic>
#include <condition_variable>
#include "neural.h"

namespace neural 
{
	class nn_thread_worker_pool
	{
		size_t taskcount;
		std::atomic<size_t> current_task_id;
		
		int active_threads;
		int num_logical_per_physical_core;
		
		std::mutex mtx_wake;
		std::condition_variable cv_wake;
		std::condition_variable cv_endtasks;

		int thread_batch_size;
		int enable_thread_denom;
		const std::vector<task>* current_request;
		volatile bool stop;

		std::vector<std::thread> threads;

		bool is_thread_enable(uint32_t threadId);
		void process_task(uint32_t threadId);

	public:
		nn_thread_worker_pool(size_t arg_num_threads = 0);
		~nn_thread_worker_pool();

		// push	tasks for processing
		void push_job(const task_group& requests);
	};
};



