#include "api/neural_base.h"
#include "thread_pool.h"

#include <iostream>

namespace neural {

nn_thread_worker_pool thread_pool;

void execute(std::vector<primitive> list) {
    try {
        for(auto &item : list)
            thread_pool.push_job(item.work());
    } catch (std::exception &e) {
        std::cerr << e.what();
    } catch (...) {
        std::cerr << "Unknown exception has been thrown.";
    }
}


};