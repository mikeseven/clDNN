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

__kernel void Fully_Connected_GPU(const __global float* input, uint input_size, const __global float* weights, uint4 weight_size, __global float* bias, __global float* pDst)
{
    const int x = get_global_id(0);
    
    pDst[x] = 0;
    for(uint i = 0; i < input_size; i++)
    {
        pDst[x] += input[i] * weights[(x * weight_size.x) + i];
    }
    pDst[x] += bias[x];
};