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

	struct task_package
	{
		bool use_hyper_threading = true;
		int batch_size = 1;
		std::vector<task> tsk;

		task_package(const std::vector<task>& _tsk) : tsk(_tsk) {};  //!!!
		task_package(const task _task) : tsk{ _task } {};  //!!!

		task_package(){};
		task_package(const task_package& tp) : tsk(tp.tsk), use_hyper_threading(tp.use_hyper_threading), batch_size(tp.batch_size){};
		task_package& operator=(const task_package& tp)
		{
			if (this != &tp)
			{
				tsk = tp.tsk;
				use_hyper_threading = tp.use_hyper_threading;
				batch_size = tp.batch_size;
			}
			return *this;
		}
	};


	class nn_thread_worker_pool
	{
		size_t taskcount;
		std::atomic<size_t> current_task_id;
		std::atomic<size_t> remain_task_count;

		int active_threads;
		int num_threads_per_core;
		
		std::mutex mtx_wake;
		std::condition_variable cv_wake;
		std::condition_variable cv_endtasks;

		int window_size;
		int enable_thread_mod;
		const std::vector<task>* current_request;
		volatile bool stop;

		std::vector<std::thread> threads;


		bool is_thread_enable(uint32_t threadId)
		{
			return (threadId % enable_thread_mod) == 0;
		}

		void process_task(uint32_t threadId);

	public:
		nn_thread_worker_pool(size_t arg_num_threads = 0);
		~nn_thread_worker_pool();

		void push_job(const task_package& requests);
		//void push_job(const std::vector<task>& requests);
	};
};



