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

#include "convolution_relu.h"

namespace neural {
    struct convolution_relu_gpu : is_an_implementation {
        convolution_relu_gpu(convolution_relu &arg);
        ~convolution_relu_gpu();
        static void implementation(const void *ptr);
        const convolution_relu &outer;

        static is_an_implementation *create(convolution_relu &arg) { return new convolution_relu_gpu(arg); };
        task_group work() {
            return{ { task{ implementation, &outer } }, schedule::single };
        }
    };
}