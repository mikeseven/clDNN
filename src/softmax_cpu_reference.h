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

#include "softmax.h"

namespace neural {
namespace normalization {
    struct softmax_cpu_reference : is_an_implementation {
        softmax_cpu_reference(softmax &arg);
        ~softmax_cpu_reference();
        static void implementation(const void *ptr);

        static is_an_implementation *create(softmax &arg) { return new softmax_cpu_reference(arg); };
<<<<<<< HEAD
        task_group work() { return {task{implementation, &outer}}; };
=======
        std::vector<task> work() { return {task{implementation, &outer}}; };
>>>>>>> 820a3d28fd75b823e9e673f8e33a588d70695ead

        const softmax &outer;
    };
}
}
