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

#include "convolution.h"

namespace neural {
    struct convolution_cpu_jit_generic : is_an_implementation {
        convolution_cpu_jit_generic(convolution &arg);
        ~convolution_cpu_jit_generic();

        static is_an_implementation *create(convolution &arg) { return new convolution_cpu_jit_generic(arg); };
        task_group work() { return jit_conv_ptr->work(); };

        std::unique_ptr<is_an_implementation> jit_conv_ptr;
        const convolution &outer;
    };
}