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

#include "include/cnn_common.cl"

#if defined(cl_intel_subgroups_short)
#define TILE_M          1
#define TILE_K          KERNEL_WIDTH
#define TILE_N          32

__attribute__((intel_reqd_sub_group_size(16)))
KERNEL(convolution_f16)(
    const __global half *src0,
    __global half *dst,
    const __global half *src1,
#ifdef OUTPUT_BIASED
    const __global half *bias,
#endif
    uint split_idx)
{
#include "include/cnn_common_data_types.cl"

    const unsigned group_x = get_group_id(0);
    const unsigned group_y = get_group_id(1);
    const unsigned global_x = get_global_id(0);
    const unsigned global_y = get_global_id(1);
    const unsigned global_z = get_global_id(2);

    unsigned interleaved_y;
    unsigned kernel_y;
    unsigned kernel_idx;

    // Result ctile (*dst) is M rows x N columns
    // LWG size is 1x16.  Thus each thread calculates 16*M rows x N cols of ctile.
    half16  blockC00 = 0.f;
    half16  blockC10 = 0.f;

    const uint in_split_offset = split_idx * INPUT_FEATURE_PITCH * INPUT_FEATURE_NUM;
    // Src0 (patch input) is directly used as atile.
    // Each work item points to the start of a different patch.
    // atile is M rows x K columns.
#if defined(INPUT_BUFFER_WIDTH_PADDED) && defined(INPUT_BUFFER_HEIGHT_PADDED)
    uint src0_read_offset = INPUT_VIEW_OFFSET + in_split_offset
     + INPUT_BATCH_PITCH * global_z                                   // batch offset
     + ( ( global_y / OUTPUT_SIZE_X ) * STRIDE_Y * INPUT_Y_PITCH )      // y offset
     + ( ( global_y % OUTPUT_SIZE_X ) * STRIDE_X );                 // x offset
#elif !defined(INPUT_BUFFER_WIDTH_PADDED) && !defined(INPUT_BUFFER_HEIGHT_PADDED)
    #pragma error - fix this path
    const int y_offset = ( global_y / OUTPUT_SIZE_X ) * STRIDE_Y - INPUT_PADDING_Y;
    const int x_offset = ( global_y % OUTPUT_SIZE_X ) * STRIDE_X - INPUT_PADDING_X;
    uint src0_read_offset = INPUT_OFFSET + in_split_offset + INPUT_BATCH_PITCH * global_z
                            + y_offset * INPUT_Y_PITCH;

    int partial_left = 0, partial_right = 0;
    if (x_offset < 0)
    {
        partial_left = min((int) KERNEL_WIDTH, (int) abs(x_offset));
        src0_read_offset -= partial_left;
    }
    else
    {
        partial_left = 0;
        src0_read_offset +=  x_offset;
    }
    if ((x_offset + KERNEL_WIDTH) >= INPUT_SIZE_X)
        partial_right = min(KERNEL_WIDTH, INPUT_SIZE_X - x_offset);
    else
        partial_right = KERNEL_WIDTH;

#elif defined(INPUT_BUFFER_WIDTH_PADDED)
    #pragma error - fix this path
    // TODO: Handle offset
    const int y_offset = ( global_y / OUTPUT_SIZE_X ) * STRIDE_Y -INPUT_PADDING_Y;
    int src0_read_offset = in_split_offset + INPUT_BATCH_PITCH * global_z        // batch offset
     + y_offset * INPUT_Y_PITCH                              // y offset
     + ( ( global_y % OUTPUT_SIZE_X ) * STRIDE_X );                // x offset
#endif

    const __global half *src0_read = src0 + src0_read_offset;

    // Src1 (filter) is directly used as btile.
    // It starts at the top of src1 and walks down.
    // btile is K rows x N columns.
    const __global half *src1_read = src1 + ( global_x * TILE_N * 2 );

#define DOT_PRODUCT_16( _result, _rowA, colB )    \
    {   \
        _result.s0 = mad( _rowA, sub_group_broadcast( colB,  0 ), _result.s0 );  \
        _result.s1 = mad( _rowA, sub_group_broadcast( colB,  1 ), _result.s1 );  \
        _result.s2 = mad( _rowA, sub_group_broadcast( colB,  2 ), _result.s2 );  \
        _result.s3 = mad( _rowA, sub_group_broadcast( colB,  3 ), _result.s3 );  \
        _result.s4 = mad( _rowA, sub_group_broadcast( colB,  4 ), _result.s4 );  \
        _result.s5 = mad( _rowA, sub_group_broadcast( colB,  5 ), _result.s5 );  \
        _result.s6 = mad( _rowA, sub_group_broadcast( colB,  6 ), _result.s6 );  \
        _result.s7 = mad( _rowA, sub_group_broadcast( colB,  7 ), _result.s7 );  \
        _result.s8 = mad( _rowA, sub_group_broadcast( colB,  8 ), _result.s8 );  \
        _result.s9 = mad( _rowA, sub_group_broadcast( colB,  9 ), _result.s9 );  \
        _result.sa = mad( _rowA, sub_group_broadcast( colB, 10 ), _result.sa );  \
        _result.sb = mad( _rowA, sub_group_broadcast( colB, 11 ), _result.sb );  \
        _result.sc = mad( _rowA, sub_group_broadcast( colB, 12 ), _result.sc );  \
        _result.sd = mad( _rowA, sub_group_broadcast( colB, 13 ), _result.sd );  \
        _result.se = mad( _rowA, sub_group_broadcast( colB, 14 ), _result.se );  \
        _result.sf = mad( _rowA, sub_group_broadcast( colB, 15 ), _result.sf );  \
    }
    typedef CAT( half, KERNEL_WIDTH ) half_t;
    // Walk DOWN src0 (patch 0, 1, 2, ...) and DOWN src1.
    // Inner loop loads and FMADs one row (KERNEL_WIDTH) of each input patch
    // and KERNEL_WIDTH/2 rows of interleaved filter.
    unsigned patch_depth = 0;
    __attribute__((opencl_unroll_hint(1)))
    do
    {
        int patch_row = 0;
        __attribute__((opencl_unroll_hint(1)))
        do
        {
            // Load atile and btile.
            // Kernel data is partially interleaved.  Every 2 rows are interleaved at half16 granularity.
            // The exception is that if KERNEL_WIDTH is odd the last row is not interleaved.  The non
            // interleaved row is padded with zero to ensure same size as interleaved rows. This
            // interleaving is done to ensure 0% GDR bank conflicts.  For example, this is how the
            // kernel data would be arranged before/after interleaving for KERNEL_WIDTH=3.
            // (0, 0) (16, 0) (32, 0) (48, 0) ...     (0, 0) ( 0, 1) (16, 0) ( 0, 1) (32, 0) (0, 1) (48, 0) ...
            // (0, 1) (16, 1) (32, 1) (48, 1) ... =>  (0, 2) (16, 2) (32, 2) (48, 2) ...
            // (0, 2) (16, 2) (32, 2) (48, 2) ...     ...
            // ...
            const bool kernel_width_is_odd = KERNEL_WIDTH % 2 == 1;
            #if defined(INPUT_BUFFER_WIDTH_PADDED) && defined(INPUT_BUFFER_HEIGHT_PADDED)
            
            // in case the data is not aligned to sizeof(T)*KERNEL_WIDTH we need to use vload or set the data in a loop
            half blockA00[KERNEL_WIDTH];
            {
                unsigned i = 0;
                LOOP(KERNEL_WIDTH, i, 
                {
                    blockA00[i] = src0_read[i];
                } )
            }
            
            half*  pblockA00 = (half*)(&blockA00);

            #elif !defined(INPUT_BUFFER_WIDTH_PADDED) && !defined(INPUT_BUFFER_HEIGHT_PADDED)
            // TODO: Fixed vload issue in this path.
            #pragma error
            half_t blockA00;
            half*  pblockA00 = (half*)(&blockA00);
            #if (INPUT_PADDING_X == 1) && (INPPUT_PADDING_Y == 1) && (KERNEL_WIDTH == 3) && (KERNEL_HEIGHT == 3)
            if ((y_offset +  patch_row < 0) || ((y_offset + patch_row) >= INPUT_SIZE_Y))
            {
                blockA00 = half_zeros;
            }
            else
            {
                 blockA00 = ( (const __global half_t*)(src0_read - partial_left) )[0];
                 if (partial_left) pblockA00[0] = 0;
                 if (partial_right != KERNEL_WIDTH) pblockA00[KERNEL_WIDTH - 1] = 0;
            }
            #else
            if ((y_offset +  patch_row < 0) || ((y_offset + patch_row) >= INPUT_SIZE_Y))
            {
                blockA00 = half_zeros;
            }
            else
            {
                 blockA00 = ( (const __global half_t*)(src0_read - partial_left) )[0];
                 for (unsigned i = 0; i < partial_left; ++i) pblockA00[i] = 0;
                 for (unsigned i = partial_right; i < KERNEL_WIDTH; ++i) pblockA00[i] = 0;

            }
            #endif
            #elif defined(INPUT_BUFFER_WIDTH_PADDED)
            // TODO: Fixed vload issue in this path.
            #pragma error
            if ((y_offset +  patch_row < 0) || ((y_offset + patch_row) >= INPUT_SIZE_Y))
            {
                blockA00 = half_zeros;
            }
            else
            {
                blockA00 = ( (const __global half_t*)(src0_read) )[0];
            }
            #endif
            src0_read += INPUT_Y_PITCH;

            ushort blockB00[KERNEL_WIDTH * 2];
            ushort4* p4BlockB00 = (ushort4*)blockB00;
            ushort2* p2BlockB00 = (ushort2*)blockB00;
            half* pBlockB00  = (half*)blockB00;

            interleaved_y = 0;
            LOOP(KERNEL_WIDTH_DIV2, interleaved_y,
            {
                p4BlockB00[interleaved_y] = intel_sub_group_block_read_us4( (const __global ushort*)src1_read );
                src1_read += ALIGNED_OFM * 2;
            } )
            if ( kernel_width_is_odd )
            {
                p2BlockB00[KERNEL_WIDTH - 1] = intel_sub_group_block_read_us2( (const __global ushort*)src1_read );
                src1_read += ALIGNED_OFM * 2;
            }

            // Perform MADs
            kernel_idx = 0;
            interleaved_y = 0;
            LOOP(KERNEL_WIDTH_DIV2, interleaved_y,
            {
                kernel_y = interleaved_y * 2;
                DOT_PRODUCT_16( blockC00, pblockA00[kernel_y    ], pBlockB00[kernel_idx] ); kernel_idx++;
                DOT_PRODUCT_16( blockC00, pblockA00[kernel_y + 1], pBlockB00[kernel_idx] ); kernel_idx++;
                DOT_PRODUCT_16( blockC10, pblockA00[kernel_y    ], pBlockB00[kernel_idx] ); kernel_idx++;
                DOT_PRODUCT_16( blockC10, pblockA00[kernel_y + 1], pBlockB00[kernel_idx] ); kernel_idx++;
            } )
            if ( kernel_width_is_odd )
            {
                kernel_y = interleaved_y * 2;
                DOT_PRODUCT_16( blockC00, pblockA00[kernel_y], pBlockB00[kernel_idx] ); kernel_idx++;
                DOT_PRODUCT_16( blockC10, pblockA00[kernel_y], pBlockB00[kernel_idx] ); kernel_idx++;
            }
        }
        while( ++patch_row < KERNEL_HEIGHT );

        src0_read += INPUT_FEATURE_PITCH - ( KERNEL_HEIGHT * INPUT_Y_PITCH ); // reset to start of next slice of patch
    }
    while ( ++patch_depth < INPUT_FEATURE_NUM );

    #undef DOT_PRODUCT_16

    const uint out_split_offset = split_idx * OUTPUT_FEATURE_PITCH * OUTPUT_FEATURE_NUM;
    // Dst resembles a cube of width x height x (output channel * batches).  Each tile writes:
    // (SIMD * TILE_M) x 1 x TILE_N.  Partial writes most likely generated if padding used.
    __global half *out = dst + OUTPUT_OFFSET + out_split_offset
     + global_z * OUTPUT_BATCH_PITCH                                                   // batch offset
     + ( group_x * TILE_N ) * OUTPUT_FEATURE_PITCH                                       // channel offset
     + ( ( global_y * TILE_M ) / OUTPUT_SIZE_X ) * OUTPUT_Y_PITCH  // y offset
     + ( ( global_y * TILE_M ) % OUTPUT_SIZE_X );               // x offset


    if (global_y * TILE_M < OUTPUT_SIZE_X * OUTPUT_SIZE_Y )
    {
         #ifdef OUTPUT_BIASED
         __global half16* biasPtr = (__global half16*) (bias + group_x * TILE_N);
         #endif

#if ( ( OUTPUT_FEATURE_NUM % TILE_N ) == 0 )

        #ifdef OUTPUT_BIASED
        blockC00 += *biasPtr;
        blockC10 += *(biasPtr + 1);
        #endif

        blockC00 = FUNC_CALL(activation_function_half16)(blockC00, NL_M, NL_N);
        blockC10 = FUNC_CALL(activation_function_half16)(blockC10, NL_M, NL_N);

        for (unsigned i = 0; i < 16; i++)
        {
            out[( 0+i) * OUTPUT_FEATURE_PITCH] = blockC00[i];
            out[(16+i) * OUTPUT_FEATURE_PITCH] = blockC10[i];
        }

#elif ( ( OUTPUT_FEATURE_NUM % 16 ) == 0 )
        if ( ( global_x + 1 ) < get_global_size(0) )
        {
            #ifdef OUTPUT_BIASED
            blockC00 += *biasPtr;
            blockC10 += *(biasPtr + 1);
            #endif

            blockC00 = FUNC_CALL(activation_function_half16)(blockC00, NL_M, NL_N);
            blockC10 = FUNC_CALL(activation_function_half16)(blockC10, NL_M, NL_N);

            for ( unsigned i = 0; i < 16; i++ )
            {
                out[( 0+i) * OUTPUT_FEATURE_PITCH] = blockC00[i];
                out[(16+i) * OUTPUT_FEATURE_PITCH] = blockC10[i];
            }
        }
        else
        {
            #ifdef OUTPUT_BIASED
            blockC00 += *biasPtr;
            #endif

            blockC00 = FUNC_CALL(activation_function_half16)(blockC00, NL_M, NL_N);

            for (unsigned i = 0; i < 16; i++)
            {
                out[( 0+i) * OUTPUT_FEATURE_PITCH] = blockC00[i];
            }
        }
#else
        if ( ( global_x + 1 ) < get_global_size(0) )
        {
            #ifdef OUTPUT_BIASED
            blockC00 += *biasPtr;
            blockC10 += *(biasPtr + 1);
            #endif

            blockC00 = FUNC_CALL(activation_function_half16)(blockC00, NL_M, NL_N);
            blockC10 = FUNC_CALL(activation_function_half16)(blockC10, NL_M, NL_N);

            for ( unsigned i = 0; i < 16; i++ )
            {
                out[( 0+i) * OUTPUT_FEATURE_PITCH] = blockC00[i];
                out[(16+i) * OUTPUT_FEATURE_PITCH] = blockC10[i];
            }
        }
        else
        {
#if ( (OUTPUT_FEATURE_NUM % TILE_N) > 16 )

            #ifdef OUTPUT_BIASED
            blockC00 += *biasPtr;
            blockC10 += *(biasPtr + 1);
            #endif

            blockC00 = FUNC_CALL(activation_function_half16)(blockC00, NL_M, NL_N);
            blockC10 = FUNC_CALL(activation_function_half16)(blockC10, NL_M, NL_N);

            for (unsigned i = 0; i < 16 ; i++)
            {
                out[( 0+i) * OUTPUT_FEATURE_PITCH] = blockC00[i];
            }
            for (unsigned i = 0; i < OUTPUT_FEATURE_NUM % 16 ; i++)
            {
                out[(16+i) * OUTPUT_FEATURE_PITCH] = blockC10[i];
            }
#else
            #ifdef OUTPUT_BIASED
            blockC00 += *biasPtr;
            #endif

            blockC00 = FUNC_CALL(activation_function_half16)(blockC00, NL_M, NL_N);

            for (unsigned i = 0; i < OUTPUT_FEATURE_NUM % 16 ; i++)
            {
                out[( 0+i) * OUTPUT_FEATURE_PITCH] = blockC00[i];
            }
#endif
        }
#endif
    }

}
#endif
