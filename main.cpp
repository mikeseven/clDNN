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

#include "neural.h"
#include "thread_pool.h"


auto main(int, char *[]) -> int {
    extern void spatial_bn_trivial_example_forward_training_float();
    extern void spatial_bn_trivial_example_forward_training_double();
    extern void spatial_bn_trivial_example_backward_training_float();
    extern void spatial_bn_trivial_example_backward_training_double();
    extern void spatial_bn_trivial_example_inference_float();
    extern void spatial_bn_trivial_example_inference_double();
    extern void spatial_bn_complex_example_training_float();
    extern void spatial_bn_complex_example_training_double();

    spatial_bn_trivial_example_forward_training_float();
    spatial_bn_trivial_example_forward_training_double();
    spatial_bn_trivial_example_backward_training_float();
    spatial_bn_trivial_example_backward_training_double();
    spatial_bn_trivial_example_inference_float();
    spatial_bn_trivial_example_inference_double();
    spatial_bn_complex_example_training_float();
    spatial_bn_complex_example_training_double();
}