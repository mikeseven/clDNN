#pragma once

#include <thread>
#include <mutex>
#include <cassert>
#include <vector>
#include <atomic>

namespace neural 
{
	// task to be performed in form of callback & data for it
	struct task {
		void(*callback)(const void *);
		const void *data;
	};

	struct task_group
	{
		bool use_hyper_threading = true;	
		int task_count_per_thread = 1;		// portion of tasks which get every thread for processing per iteration
		std::vector<task> tsk;

		task_group(const std::vector<task>& _tsk) : tsk(_tsk) {};	// workaround, could be remove in future
		task_group(const task _task) : tsk{ _task } {};				// workaround, could be remove in future

		task_group(){};
		task_group(const task_group& tp) : tsk(tp.tsk), use_hyper_threading(tp.use_hyper_threading), task_count_per_thread(tp.task_count_per_thread){};
		task_group& operator=(const task_group& tp)
		{
			if (this != &tp)
			{
				tsk = tp.tsk;
				use_hyper_threading = tp.use_hyper_threading;
				task_count_per_thread = tp.task_count_per_thread;
			}
			return *this;
		}
	};


	class nn_thread_worker_pool
	{
		size_t taskcount;
		std::atomic<size_t> current_task_id;
		
		int active_threads;
		int num_threads_per_core;
		
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



