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

#include "api/neural.h"

namespace neural {
    struct relu_gpu : is_an_implementation {
        relu_gpu(relu &arg);
        ~relu_gpu();
        static void implementation(const void *ptr);

        static is_an_implementation *create(relu &arg) { return new relu_gpu(arg); };
        task_group work() override { return {{task{implementation, &outer}}, schedule::unordered}; };

        const relu &outer;
    };
}