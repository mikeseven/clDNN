#include "thread_pool.h"
#include <algorithm>
#include <iostream>

namespace neural
{

	nn_thread_worker_pool::nn_thread_worker_pool(size_t arg_num_threads)
		:taskcount(0), current_task_id(0), stop(false), current_request(nullptr), num_threads_per_core(2), enable_thread_mod(1), window_size(1)
	{
		unsigned int num_logic_cores = std::thread::hardware_concurrency();
		unsigned int num_threads = arg_num_threads == 0 ? num_logic_cores : static_cast<unsigned int>(arg_num_threads);
		active_threads = num_threads;

		for (unsigned int thread_id = 0; thread_id < num_threads; ++thread_id)
		{
			std::thread nthread(&nn_thread_worker_pool::process_task, this, thread_id);

			/*GROUP_AFFINITY ga;
			GROUP_AFFINITY pga;
			auto idx = thread_id;
			ga.Group = idx / 64;
			ga.Mask = 1i64 << (idx % 64);

			auto a = SetThreadGroupAffinity(thread->worker_thread.native_handle(), &ga, &pga);
			auto b = SetThreadAffinityMask(thread->worker_thread.native_handle(), ga.Mask);
*/
			//auto err = GetLastError();
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
			cv_wake.notify_all();
		}
		std::for_each(threads.begin(), threads.end(), std::mem_fn(&std::thread::join));
	}


	void nn_thread_worker_pool::push_job(const task_package& requests)
	{
		{
			std::lock_guard<std::mutex> ul(mtx_wake);
			current_request = &requests.tsk;
			remain_task_count = requests.tsk.size();
			taskcount  = remain_task_count;
			current_task_id = 0;
			enable_thread_mod = requests.use_hyper_threading ? 1 : num_threads_per_core;
			window_size = requests.batch_size;
			cv_wake.notify_all();
		}

		std::unique_lock<std::mutex> ul(mtx_wake);
		cv_endtasks.wait(ul, [this] {return (remain_task_count == 0 && active_threads == 0); });
		current_request = nullptr;
	}

	//void nn_thread_worker_pool::push_job(const std::vector<task>& requests)
	//{
	//	std::cout << "------------" << requests.size() << std::endl;
	//	{
	//		std::lock_guard<std::mutex> ul(mtx_wake);
	//		current_request = &requests;
	//		remain_task_count = requests.size();
	//		taskcount = requests.size();
	//		current_task_id = 0;
	//		
	//		enable_thread_mod = 1; // !!!
	//		window_size = 1;       // !!!

	//		cv_wake.notify_all();
	//	}

	//	std::unique_lock<std::mutex> ul(mtx_wake);
	//	cv_endtasks.wait(ul, [this] {return (remain_task_count == 0 && active_threads == 0); });
	//	current_request = nullptr;
	//}



	void nn_thread_worker_pool::process_task(uint32_t threadId)
	{
		while (true)
		{
			std::unique_lock<std::mutex> ul(mtx_wake);
			active_threads--;
			cv_endtasks.notify_one();
			cv_wake.wait(ul, [this, threadId] { return (is_thread_enable(threadId) && current_task_id < taskcount) || stop; });
			active_threads++;
			ul.unlock();

			if (stop)
				return;

			while (true)
			{
				size_t nextTaskId = current_task_id.fetch_add(window_size);

				if (nextTaskId >= taskcount + window_size - 1)
					break;

				size_t endTaskId = nextTaskId + window_size;
				endTaskId = endTaskId > taskcount ? taskcount : endTaskId;
				for (size_t i = nextTaskId; i < endTaskId; ++i)
				{
					(*current_request)[i].callback((*current_request)[i].data);
					remain_task_count--;
				}
			}
		}
	}

}