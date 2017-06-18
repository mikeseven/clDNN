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


#if FP16_SUPPORTED
    #pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#if INPUT_DIMS == 2
#if   defined INPUT_LAYOUT_BF
inline unsigned int get_output_index(unsigned int f, unsigned int b, unsigned int, unsigned int, unsigned int)
#elif defined INPUT_LAYOUT_FB
inline unsigned int get_output_index(unsigned int b, unsigned int f, unsigned int, unsigned int, unsigned int)
#endif
{ 
   return OUTPUT_OFFSET + b*OUTPUT_BATCH_PITCH + f*OUTPUT_FEATURE_PITCH
}
#endif

#if INPUT_DIMS == 4
#if   defined INPUT_LAYOUT_BFYX
inline unsigned int get_output_index(unsigned int x, unsigned int y, unsigned int f, unsigned int b, unsigned int)
#elif defined INPUT_LAYOUT_YXFB
inline unsigned int get_output_index(unsigned int b, unsigned int f, unsigned int x, unsigned int y, unsigned int)
#elif defined INPUT_LAYOUT_BYXF
inline unsigned int get_output_index(unsigned int f, unsigned int x, unsigned int y, unsigned int b, unsigned int)
#elif defined INPUT_LAYOUT_FYXB
inline unsigned int get_output_index(unsigned int b, unsigned int x, unsigned int y, unsigned int f, unsigned int)
#endif
{ 
   return OUTPUT_OFFSET + b*OUTPUT_BATCH_PITCH + f*OUTPUT_FEATURE_PITCH + y*OUTPUT_Y_PITCH + x*OUTPUT_X_PITCH;
}
#endif

#if INPUT_DIMS == 5
#if   defined INPUT_LAYOUT_BRFYX
inline unsigned int get_output_index(unsigned int x, unsigned int y, unsigned int f, unsigned int r, unsigned int b)
#endif
{ 
   return OUTPUT_OFFSET + b*OUTPUT_BATCH_PITCH + r*OUTPUT_ROI_PITCH + f*OUTPUT_FEATURE_PITCH + y*OUTPUT_Y_PITCH + x*OUTPUT_X_PITCH;
}
#endif

#if   defined REORDER_OUTPUT_MODE_XYZW
inline unsigned int get_output_index(unsigned int x, unsigned int y, unsigned int z, unsigned int w)
#elif defined REORDER_OUTPUT_MODE_XYWZ
inline unsigned int get_output_index(unsigned int x, unsigned int y, unsigned int w, unsigned int z)
#elif defined REORDER_OUTPUT_MODE_XWYZ
inline unsigned int get_output_index(unsigned int x, unsigned int w, unsigned int y, unsigned int z)
#elif defined REORDER_OUTPUT_MODE_WXYZ
inline unsigned int get_output_index(unsigned int w, unsigned int x, unsigned int y, unsigned int z)
#elif defined REORDER_OUTPUT_MODE_XZYW
inline unsigned int get_output_index(unsigned int x, unsigned int z, unsigned int y, unsigned int w)
#elif defined REORDER_OUTPUT_MODE_ZYXW
inline unsigned int get_output_index(unsigned int z, unsigned int y, unsigned int x, unsigned int w)
#elif defined REORDER_OUTPUT_MODE_YXZW
inline unsigned int get_output_index(unsigned int y, unsigned int x, unsigned int z, unsigned int w)
#endif
{ 
   return OUTPUT_OFFSET + w*OUTPUT_PITCH_3 + z*OUTPUT_PITCH_2 + y*OUTPUT_PITCH_1 + x/*OUTPUT_PITCH_0*/;
}

uint FUNC(OUTPUT_FORMAT)(uint size[DIMENSIONS], uint pos[DIMENSIONS], uint lpad[DIMENSIONS], uint upad[DIMENSIONS]) 
{
    OUTPUT_FORMAT_IMPLEMENTATION
}

KERNEL (reorder)(const __global SRC_TYPE* input, __global DEST_TYPE* output)
{
    const unsigned x = get_global_id(0);
    const unsigned y = get_global_id(1);
#if INPUT_DIMS == 3
    const unsigned z = get_global_id(2);
    const unsigned w = 0;
#elif INPUT_DIMS == 4
    const unsigned z = get_global_id(2) % INPUT_DIM_3;
    const unsigned w = get_global_id(2) / INPUT_DIM_3;
#endif

    uint pos[DIMENSIONS]; // position in each of dimensions
    pos[CALCULATION_ORDER[DIMENSIONS-1]] = global_id_2;
    pos[CALCULATION_ORDER[DIMENSIONS-2]] = global_id_1;
    uint pos1D = global_id_0;
    for(uint i = 0; i < DIMENSIONS-2; i++)
    {
        uint order_idx = CALCULATION_ORDER[i];
        pos[order_idx] = pos1D % SIZE[order_idx];
        pos1D /= SIZE[order_idx];
    }

    uint output_pos = FUNC_CALL(OUTPUT_FORMAT)(SIZE, pos, LOWER_PADDING, UPPER_PADDING);
    uint input_idx = (global_id_2 * global_size_1 + global_id_1) * global_size_0 + global_id_0;
    output[output_pos] = SRC_DEST_TYPE_CVT_FUNC(input[input_idx]);
}

#undef SRC_DEST_TYPE_CVT_FUNC
#undef TYPE_CVT_FUNC2
#undef TYPE_CVT_FUNC3