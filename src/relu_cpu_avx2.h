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

#include "relu.h"

namespace neural {
    struct relu_cpu_avx2 : is_an_implementation {
        relu_cpu_avx2(relu &arg);
        ~relu_cpu_avx2();

        static is_an_implementation *create(relu &arg) { return new relu_cpu_avx2(arg); };
        task_group work() { return relu_ptr->work(); };

        std::unique_ptr<is_an_implementation> relu_ptr;
        const relu &outer;
    };

    struct relu_backward_cpu_avx2 : is_an_implementation {
        relu_backward_cpu_avx2(relu_backward &arg);
        ~relu_backward_cpu_avx2();

        static is_an_implementation *create(relu_backward &arg) { return new relu_backward_cpu_avx2(arg); };
		task_group work() { return relu_ptr->work(); };

        std::unique_ptr<is_an_implementation> relu_ptr;
        const relu_backward &outer;
    };
}