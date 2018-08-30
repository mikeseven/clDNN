// Copyright (c) 2016-2017 Intel Corporation
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


#include "include/include_all.cl"

#if MAX_POOLING
    #define INIT_VAL CHAR_MIN
#elif AVG_POOLING
    #define INIT_VAL 0
#else
#error
#endif


inline int FUNC(apply_pooling)(int tmp, int in)
{
#if MAX_POOLING
    return max(tmp, in);
#elif AVG_POOLING
    return tmp + in;
#endif
}

KERNEL(pooling_gpu_fs_bs_yx_bsv4_fsv32)(
    const __global UNIT_TYPE* input,
    __global UNIT_TYPE* output)
{

}

#undef INIT_VAL