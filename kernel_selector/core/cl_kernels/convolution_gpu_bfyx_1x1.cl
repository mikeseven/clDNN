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

// TODO: use CAT
#define CONCAT_TOKEN_HANDLER1(prefix, suffix) prefix##suffix

// Expands and concatenates two tokens into one.
#define CONCAT_TOKEN(prefix, suffix) CONCAT_TOKEN_HANDLER1(prefix, suffix)

// Creates vector type.
#define MAKE_VECTOR_TYPE(elem_type, size) CONCAT_TOKEN(elem_type, size)


#if FP16_UNIT_USED
    #define MULTIPLY_BLOCKS_8x8(_result, _blockA, _blockB)  \
    {   \
        const half8 acol0 = TRANSPOSE_BLOCK_8_FP16( _blockA.s0 ); \
        const half8 acol1 = TRANSPOSE_BLOCK_8_FP16( _blockA.s1 ); \
        const half8 acol2 = TRANSPOSE_BLOCK_8_FP16( _blockA.s2 ); \
        const half8 acol3 = TRANSPOSE_BLOCK_8_FP16( _blockA.s3 ); \
        const half8 acol4 = TRANSPOSE_BLOCK_8_FP16( _blockA.s4 ); \
        const half8 acol5 = TRANSPOSE_BLOCK_8_FP16( _blockA.s5 ); \
        const half8 acol6 = TRANSPOSE_BLOCK_8_FP16( _blockA.s6 ); \
        const half8 acol7 = TRANSPOSE_BLOCK_8_FP16( _blockA.s7 ); \
        _result = fma( _blockB.s0, acol0, _result ); \
        _result = fma( _blockB.s1, acol1, _result ); \
        _result = fma( _blockB.s2, acol2, _result ); \
        _result = fma( _blockB.s3, acol3, _result ); \
        _result = fma( _blockB.s4, acol4, _result ); \
        _result = fma( _blockB.s5, acol5, _result ); \
        _result = fma( _blockB.s6, acol6, _result ); \
        _result = fma( _blockB.s7, acol7, _result ); \
    }
#else
    // Block read - currently block is 4 bytes aligned.
    #define ALIGNED_BLOCK_READ8(ptr, byte_offset) as_float8(intel_sub_group_block_read8((const __global uint*)(ptr) + (byte_offset)))

    #define MULTIPLY_BLOCKS_8x8(_result, _blockA, _blockB)  \
    {   \
        const float8 acol0 = TRANSPOSE_BLOCK_8( _blockA.s0 ); \
        const float8 acol1 = TRANSPOSE_BLOCK_8( _blockA.s1 ); \
        const float8 acol2 = TRANSPOSE_BLOCK_8( _blockA.s2 ); \
        const float8 acol3 = TRANSPOSE_BLOCK_8( _blockA.s3 ); \
        const float8 acol4 = TRANSPOSE_BLOCK_8( _blockA.s4 ); \
        const float8 acol5 = TRANSPOSE_BLOCK_8( _blockA.s5 ); \
        const float8 acol6 = TRANSPOSE_BLOCK_8( _blockA.s6 ); \
        const float8 acol7 = TRANSPOSE_BLOCK_8( _blockA.s7 ); \
        _result = mad( _blockB.s0, acol0, _result ); \
        _result = mad( _blockB.s1, acol1, _result ); \
        _result = mad( _blockB.s2, acol2, _result ); \
        _result = mad( _blockB.s3, acol3, _result ); \
        _result = mad( _blockB.s4, acol4, _result ); \
        _result = mad( _blockB.s5, acol5, _result ); \
        _result = mad( _blockB.s6, acol6, _result ); \
        _result = mad( _blockB.s7, acol7, _result ); \
    }
#endif

__attribute__((intel_reqd_sub_group_size(8)))
KERNEL(convolution_bfyx_1x1)(
    __global INPUT0_TYPE* input, 
    __global OUTPUT_TYPE* output, 
    __global FILTER_TYPE* weights, 
#if BIAS_TERM
    __global BIAS_TYPE* biases,
#endif
    uint split_idx)
{
    const uint xy = get_group_id(0) * 16 + get_sub_group_local_id();
    const uint xy2 = xy + 8;

    const uint x = xy % OUTPUT_SIZE_X;
    const uint y = xy / OUTPUT_SIZE_X;
    const uint x2 = xy2 % OUTPUT_SIZE_X;
    const uint y2 = xy2 / OUTPUT_SIZE_X;
    const uint f = get_group_id(1) * 16 + get_sub_group_local_id();//get_global_id(1);
    const uint b = get_global_id(2);
    const uint group_f = get_group_id(1) * 16;

    MAKE_VECTOR_TYPE(UNIT_TYPE, 8) blockC00;
    MAKE_VECTOR_TYPE(UNIT_TYPE, 8) blockC01;
    MAKE_VECTOR_TYPE(UNIT_TYPE, 8) blockC10;
    MAKE_VECTOR_TYPE(UNIT_TYPE, 8) blockC11;

#if BIAS_TERM
    #if   BIAS_PER_OUTPUT
        const uint bias_index = GET_DATA_INDEX(BIAS, b, f, y, x);
    #elif BIAS_PER_OFM
        const uint bias_index = f;
    #endif
    for(uint i = 0; i < 8; i++)
    {
        blockC00[i] = intel_sub_group_shuffle(biases[bias_index], i);
        blockC01[i] = intel_sub_group_shuffle(biases[bias_index], i);
        blockC10[i] = intel_sub_group_shuffle(biases[bias_index+8], i);
        blockC11[i] = intel_sub_group_shuffle(biases[bias_index+8], i);
    }
#endif

#if DEPTHWISE_SEPARABLE_OPT
    const uint in_split_offset = (f / FILTER_OFM_NUM) * INPUT0_FEATURE_PITCH * FILTER_IFM_NUM;
#else
    const uint in_split_offset = split_idx * INPUT0_FEATURE_PITCH * FILTER_IFM_NUM;
#endif
    const uint filter_offset = f*FILTER_OFM_PITCH;
    const uint input_offset = b*INPUT0_BATCH_PITCH + INPUT0_OFFSET + in_split_offset;
    const uint filter_offset2 = (f+8)*FILTER_OFM_PITCH;

    for (uint k = 0; k < FILTER_IFM_NUM / 8; ++k)
    {
        MAKE_VECTOR_TYPE(UNIT_TYPE, 8) blockA00;
        MAKE_VECTOR_TYPE(UNIT_TYPE, 8) blockA01;

        MAKE_VECTOR_TYPE(UNIT_TYPE, 8) blockB00;
        MAKE_VECTOR_TYPE(UNIT_TYPE, 8) blockB01;

        uint input_idx = input_offset + xy + k*8*INPUT0_FEATURE_PITCH;
        uint filter_idx = filter_offset + k*8*FILTER_IFM_PITCH;

        uint input_idx2 = input_offset + xy2 + k*8*INPUT0_FEATURE_PITCH;
        uint filter_idx2 = filter_offset2 + k*8*FILTER_IFM_PITCH;

        for(uint i = 0; i < 8; i++)
        {
            blockA00[i] = input[input_idx];
            input_idx += INPUT0_FEATURE_PITCH;
            blockB00[i] = weights[filter_idx];
            filter_idx += FILTER_IFM_PITCH;

            blockA01[i] = input[input_idx2];
            input_idx2 += INPUT0_FEATURE_PITCH;
            blockB01[i] = weights[filter_idx2];
            filter_idx2 += FILTER_IFM_PITCH;
        }

         MULTIPLY_BLOCKS_8x8(blockC00, blockB00, blockA00);
         MULTIPLY_BLOCKS_8x8(blockC01, blockB00, blockA01);

         MULTIPLY_BLOCKS_8x8(blockC10, blockB01, blockA00);
         MULTIPLY_BLOCKS_8x8(blockC11, blockB01, blockA01);

    }

    if(xy >= INPUT0_SIZE_X * INPUT0_SIZE_Y)
        return;
    
    const uint out_split_offset = split_idx * OUTPUT_FEATURE_PITCH * OUTPUT_FEATURE_NUM;


    for(uint i = 0; i < 8; i++)
    {
        const uint dst_index = GET_DATA_INDEX(OUTPUT, b, group_f+i, y, x) + out_split_offset;     
        output[dst_index] = ACTIVATION(blockC00[i], NL_M, NL_N);
    }

    for(uint i = 0; i < 8; i++)
    {
        const uint dst_index = GET_DATA_INDEX(OUTPUT, b, group_f+i+8, y, x) + out_split_offset;     
        output[dst_index] = ACTIVATION(blockC10[i], NL_M, NL_N);
    }

    if(xy2 >= INPUT0_SIZE_X * INPUT0_SIZE_Y)
        return;

    for(uint i = 0; i < 8; i++)
    {
        const uint dst_index = GET_DATA_INDEX(OUTPUT, b, group_f+i, y2, x2) + out_split_offset;     
        output[dst_index] = ACTIVATION(blockC01[i], NL_M, NL_N);
    }

    for(uint i = 0; i < 8; i++)
    {
        const uint dst_index = GET_DATA_INDEX(OUTPUT, b, group_f+i+8, y2, x2) + out_split_offset;     
        output[dst_index] = ACTIVATION(blockC11[i], NL_M, NL_N);
    }

}
