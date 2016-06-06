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

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "api/neural.h"
#include "tests/gtest/gtest.h"
#include "test_utils/test_utils.h"
#include "thread_pool.h"

using namespace neural;
using namespace tests;

void do_some (const void* a)
{
    long  sum = 0;
    int count = rand() % 10;
    for (long i = 0; i < count; i++)
    {
        sum += i*i;
    }
    void* a_temp = (void*)(a);
    int* b = static_cast<int*>(a_temp);
    *b = sum;
    //*b = *b+1;
    //std::cout << "number=" << *static_cast<const int*>(a) << "\n";
    //std::cout << "b=" << *b << "\n";
}

TEST(thread_pool, check_result) {

    task_group group;
    const int task_count = 10;
    int num[task_count];
    for (int i = 0; i < task_count; i++)
    {
        num[i] = i;
        group.tasks.push_back({ &do_some, &num[i] });
    }

    nn_thread_worker_pool wp;
    wp.push_job(group);

    for (int i = 0; i < task_count; i++)
    {
        std::cout << "num[i]=" << num[i] << "\n";
    }

}