// Copyright (c) 2017 Intel Corporation
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

// --------------------------------------------------------------------------------------------------------------------------------
// L3_SIMD_4x8
// Input matrices dimensions: M x K x N
// Output matrix dimensions: M x N
// --------------------------------------------------------------------------------------------------------------------------------

__attribute__((reqd_work_group_size(8, 1, 1)))
void kernel KERNEL(convolution_gpu_winograd_2x3_s1)
(
    const __global UNIT_TYPE *signalw,
    const __global UNIT_TYPE *filterw,
          __global UNIT_TYPE *outputw
#ifdef BIAS_TERM
    , const __global UNIT_TYPE* bias
#endif
)
{

};
