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

///////////////////////// Input Index /////////////////////////
#if SIMPLE_INPUT
inline uint FUNC(get_input_index)(uint o, uint i, uint y, uint x)
{ 
   return INPUT_OFFSET + o*INPUT_OFM_PITCH + i*INPUT_IFM_PITCH + y*INPUT_Y_PITCH + x*INPUT_X_PITCH;
}
#else
#error - not supported
#endif

///////////////////////// Output Index /////////////////////////

#if SIMPLE_OUTPUT
inline uint FUNC(get_output_index)(uint o, uint i, uint y, uint x)
{ 
    return OUT_OFFSET + o*OUT_OFM_PITCH + i*OUT_IFM_PITCH + y*OUT_Y_PITCH + x*OUT_X_PITCH;
}
#else

#if   defined OUTPUT_LAYOUT_OS_IYX_OSV16
inline uint FUNC(get_output_index)(uint o, uint i, uint y, uint x)
{ 
    const uint slice_id = o / SUB_GROUP_SIZE;
    const uint id_in_slice = o % SUB_GROUP_SIZE;
    const size_t output_idx = OUT_OFFSET + id_in_slice + SUB_GROUP_SIZE * (i*OUT_IFM_PITCH + y*OUT_Y_PITCH + x*OUT_X_PITCH + slice_id*OUT_OFM_PITCH);
    return output_idx;
}
#elif defined OUTPUT_LAYOUT_OS_I_OSV16
inline uint FUNC(get_output_index)(uint o, uint i, uint, uint)
{ 
    const uint slice_id = o / SUB_GROUP_SIZE;
    const uint id_in_slice = o % SUB_GROUP_SIZE;
    const size_t output_idx = OUT_OFFSET + id_in_slice + SUB_GROUP_SIZE * (i*OUT_IFM_PITCH + slice_id*OUT_OFM_PITCH);
    return output_idx;
}
#elif defined OUTPUT_LAYOUT_OS_IS_ISV8_OSV8
inline uint FUNC(get_output_index)(uint o, uint i, uint, uint)
{ 
    return 0; // TODO
}
#elif defined OUTPUT_LAYOUT_IYXO_OM16X2_AXY || defined OUTPUT_LAYOUT_IYXO_OM16X2_AX_G32 || defined OUTPUT_LAYOUT_IYXO_OM8X2_AX_G32
inline uint FUNC(get_output_index)(uint o, uint i, uint y, uint x)
{
    const uint aligned_ofm_line = OUT_X_PITCH;
    const uint ifm_height_pitch = (OUT_IFM_PITCH/aligned_ofm_line);
    
#if defined OUTPUT_LAYOUT_IYXO_OM16X2_AXY
    const uint dst_height = i*ifm_height_pitch + y*OUTPUT_X + x;
    const uint base_filter_index = y*OUTPUT_X + x;
#elif defined OUTPUT_LAYOUT_IYXO_OM16X2_AX_G32 || defined OUTPUT_LAYOUT_IYXO_OM8X2_AX_G32
    const uint aligned_x_line = OUT_Y_PITCH / OUT_X_PITCH;
    const uint dst_height = i*ifm_height_pitch + y*aligned_x_line + x;
    const uint base_filter_index = x;
#endif

    const uint aligned_height = dst_height & 0xfffffffe;
    const uint base_filter_odd = (base_filter_index & 0x1);

    uint slice_id = o / SUB_GROUP_SIZE;
    uint id_in_slice = o % SUB_GROUP_SIZE;
    uint slice_pitch = 2*SUB_GROUP_SIZE;
    uint offset_in_slice = (int)(SUB_GROUP_SIZE*base_filter_odd);
#if defined OUTPUT_LAYOUT_IYXO_OM16X2_AX_G32 || defined OUTPUT_LAYOUT_IYXO_OM8X2_AX_G32
    const bool last_line_in_base_filter = (x == (OUTPUT_X - 1));
    if (last_line_in_base_filter && base_filter_odd == 0)
    {
        const uint element_in_slice = 32;
        slice_id = o / element_in_slice;
        id_in_slice = o % element_in_slice;
        slice_pitch = 2*element_in_slice;
        offset_in_slice = 0;
    }
#endif
    
    const uint in_line = (slice_pitch*slice_id + offset_in_slice + id_in_slice);
    
    const size_t output_idx = OUT_OFFSET + aligned_height*aligned_ofm_line + in_line;

    return output_idx;
}
#endif

#endif

inline uint4 FUNC(reshape)(uint o, uint i, uint y, uint x)
{
#if (INPUT_DIMS == OUT_DIMS)
    return (uint4)(o,i,y,x);
#elif (INPUT_DIMS == 2 && (OUT_DIMS == 4)
    uint _i  = i / (INPUT_Y*INPUT_X);
    uint _xy = i % (INPUT_Y*INPUT_X);
    uint _y = _xy / INPUT_X;
    uint _x = _xy % INPUT_X;
    return (uint4)(o,_i,_y,_x);
#elif (INPUT_DIMS == 4 && (OUT_DIMS == 2)
    uint _i = i*INPUT_Y*INPUT_X + y*INPUT_X + x;
    return (uint4)(o,_i,0,0);
#else
#error
#endif
}


KERNEL (reorder_weights)(const __global SRC_TYPE* input, __global DEST_TYPE* output)
{
    const unsigned o = get_global_id(0);
    const unsigned i = get_global_id(1);
#if   INPUT_DIMS == 2
    const unsigned y = 0;
    const unsigned x = 0;
#elif INPUT_DIMS == 4
    const unsigned y = get_global_id(2) / INPUT_X;
    const unsigned x = get_global_id(2) % INPUT_X;
#endif
    uint4 ir = FUNC_CALL(reshape)(o,i,y,x);
    output[FUNC_CALL(get_output_index)(o, i, y, x)] = input[FUNC_CALL(get_input_index)(ir[0],ir[1],ir[2],ir[3])];
}
