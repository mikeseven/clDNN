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

#include "pooling.h"

namespace neural {

    struct parameters_pooling_f32
    {
        void**   input_ptr; 
        void**   output_ptr;
        uint64_t input_offset;    
        uint64_t output_offset;
        pooling* pooling_data;
    };

    class pooling_cpu_avx2_batch24 : is_an_implementation 
    {
        std::vector<task> tasks;
        std::vector<parameters_pooling_f32> tasks_parameters;
        const pooling &outer;

    public:
        pooling_cpu_avx2_batch24(pooling &arg);
        ~pooling_cpu_avx2_batch24();
        static void implementation(const void *ptr);

        static is_an_implementation *create(pooling &arg) { return new pooling_cpu_avx2_batch24(arg); };
        task_group work() { return this->tasks; };
    };
    //struct pooling_backward_cpu_reference : is_an_implementation {
    //    pooling_backward_cpu_reference(pooling_backward &arg);
    //    ~pooling_backward_cpu_reference();
    //    static void implementation(const void *ptr);

    //    static is_an_implementation *create(pooling_backward &arg) { return new pooling_backward_cpu_reference(arg); };
    //    std::vector<task> work() { return {task{implementation, &outer}}; };

    //    const pooling_backward &outer;
    //};
}