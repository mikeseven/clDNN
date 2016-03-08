#include "api/neural_base.h"
#include "thread_pool.h"

namespace neural {

nn_thread_worker_pool thread_pool;

void execute(std::vector<primitive> list) {
    for(auto &item : list) {
        thread_pool.push_job(item.work());
    }

}


};