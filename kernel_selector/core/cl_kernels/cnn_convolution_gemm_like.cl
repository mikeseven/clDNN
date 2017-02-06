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

#if defined(__convolution_f16) && defined(cl_intel_subgroups_short)
#define TILE_M          1
#define TILE_K          KERNEL_WIDTH
#define TILE_N          32

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void convolution_f16(
    const __global half *src0,
    __global half *dst,
    const __global half *src1,
    const __global half *bias)
{
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

    // Src0 (patch input) is directly used as atile.
    // Each work item points to the start of a different patch.
    // atile is M rows x K columns.
#if defined(INPUT_BUFFER_WIDTH_PADDED) && defined(INPUT_BUFFER_HEIGHT_PADDED)
    uint src0_read_offset = //INPUT_OFFSET - TODO: make sure that it handles correctly
       INPUT_BATCH_PITCH * global_z                                   // batch offset
     + ( ( global_y / OUT_WIDTH ) * STRIDE_Y * INPUT_ROW_PITCH )      // y offset
     + ( ( global_y % OUT_WIDTH ) * STRIDE_X );                 // x offset
#elif !defined(INPUT_BUFFER_WIDTH_PADDED) && !defined(INPUT_BUFFER_HEIGHT_PADDED)
    const int y_offset = ( global_y / OUT_WIDTH ) * STRIDE_Y - INPUT_PADDING_Y;
    const int x_offset = ( global_y % OUT_WIDTH ) * STRIDE_X - INPUT_PADDING_X;
    uint src0_read_offset = INPUT_OFFSET + INPUT_BATCH_PITCH * global_z
                            + y_offset * INPUT_ROW_PITCH;

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
    if ((x_offset + KERNEL_WIDTH) >= INPUT_WIDTH)
        partial_right = min(KERNEL_WIDTH, INPUT_WIDTH - x_offset);
    else
        partial_right = KERNEL_WIDTH;

#elif defined(INPUT_BUFFER_WIDTH_PADDED)
    // TODO: Handle offset
    const int y_offset = ( global_y / OUT_WIDTH ) * STRIDE_Y -INPUT_PADDING_Y;
    int src0_read_offset = INPUT_BATCH_PITCH * global_z        // batch offset
     + y_offset * INPUT_ROW_PITCH                              // y offset
     + ( ( global_y % OUT_WIDTH ) * STRIDE_X );                // x offset
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
        unsigned patch_row = 0;
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
            if ((y_offset +  patch_row < 0) || ((y_offset + patch_row) >= INPUT_HEIGHT))
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
            if ((y_offset +  patch_row < 0) || ((y_offset + patch_row) >= INPUT_HEIGHT))
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
            if ((y_offset +  patch_row < 0) || ((y_offset + patch_row) >= INPUT_HEIGHT))
            {
                blockA00 = half_zeros;
            }
            else
            {
                blockA00 = ( (const __global half_t*)(src0_read) )[0];
            }
            #endif
            src0_read += INPUT_ROW_PITCH;

            ushort blockB00[KERNEL_WIDTH * 2];
            ushort4* p4BlockB00 = (ushort4*)blockB00;
            ushort2* p2BlockB00 = (ushort2*)blockB00;
            half* pBlockB00  = (half*)blockB00;

            interleaved_y = 0;
            LOOP(KERNEL_WIDTH_DIV2, interleaved_y,
            {
                p4BlockB00[interleaved_y] = intel_sub_group_block_read_us4( (const __global ushort*)src1_read );
                src1_read += WIDTH1 * 2;
            } )
            if ( kernel_width_is_odd )
            {
                p2BlockB00[KERNEL_WIDTH - 1] = intel_sub_group_block_read_us2( (const __global ushort*)src1_read );
                src1_read += WIDTH1 * 2;
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

        src0_read += INPUT_SLICE_PITCH - ( KERNEL_HEIGHT * INPUT_ROW_PITCH ); // reset to start of next slice of patch
    }
    while ( ++patch_depth < INPUT_DEPTH );

    #undef DOT_PRODUCT_16

    // Dst resembles a cube of width x height x (output channel * batches).  Each tile writes:
    // (SIMD * TILE_M) x 1 x TILE_N.  Partial writes most likely generated if padding used.
    __global half *out = dst + OUT_OFFSET
     + global_z * OUT_BATCH_PITCH                                                   // batch offset
     + ( group_x * TILE_N ) * OUT_SLICE_PITCH                                       // channel offset
     + ( ( global_y * TILE_M ) / OUT_WIDTH ) * OUT_ROW_PITCH  // y offset
     + ( ( global_y * TILE_M ) % OUT_WIDTH );               // x offset

    unsigned m = NL_M;
    unsigned n = NL_N;

    const half in_m = *(half*)(&m);
    const half in_n = *(half*)(&n);

    if (global_y * TILE_M < OUT_WIDTH * OUT_HEIGHT )
    {
         #ifdef OUTPUT_BIASED
         __global half16* biasPtr = (__global half16*) (bias + global_z * OUT_DEPTH + group_x * TILE_N);
         #endif

#if ( ( OUT_DEPTH % TILE_N ) == 0 )

        #ifdef OUTPUT_BIASED
        blockC00 += *biasPtr;
        blockC10 += *(biasPtr + 1);
        #endif

        blockC00 = activation_function_half16(blockC00, in_m, in_n);
        blockC10 = activation_function_half16(blockC10, in_m, in_n);

        for (unsigned i = 0; i < 16; i++)
        {
            out[( 0+i) * OUT_SLICE_PITCH] = blockC00[i];
            out[(16+i) * OUT_SLICE_PITCH] = blockC10[i];
        }

#elif ( ( OUT_DEPTH % 16 ) == 0 )
        if ( ( global_x + 1 ) < get_global_size(0) )
        {
            #ifdef OUTPUT_BIASED
            blockC00 += *biasPtr;
            blockC10 += *(biasPtr + 1);
            #endif

            blockC00 = activation_function_half16(blockC00, in_m, in_n);
            blockC10 = activation_function_half16(blockC10, in_m, in_n);

            for ( unsigned i = 0; i < 16; i++ )
            {
                out[( 0+i) * OUT_SLICE_PITCH] = blockC00[i];
                out[(16+i) * OUT_SLICE_PITCH] = blockC10[i];
            }
        }
        else
        {
            #ifdef OUTPUT_BIASED
            blockC00 += *biasPtr;
            #endif

            blockC00 = activation_function_half16(blockC00, in_m, in_n);

            for (unsigned i = 0; i < 16; i++)
            {
                out[( 0+i) * OUT_SLICE_PITCH] = blockC00[i];
            }
        }
#else
        if ( ( global_x + 1 ) < get_global_size(0) )
        {
            #ifdef OUTPUT_BIASED
            blockC00 += *biasPtr;
            blockC10 += *(biasPtr + 1);
            #endif

            blockC00 = activation_function_half16(blockC00, in_m, in_n);
            blockC10 = activation_function_half16(blockC10, in_m, in_n);

            for ( unsigned i = 0; i < 16; i++ )
            {
                out[( 0+i) * OUT_SLICE_PITCH] = blockC00[i];
                out[(16+i) * OUT_SLICE_PITCH] = blockC10[i];
            }
        }
        else
        {
#if ( (OUT_DEPTH % TILE_N) > 16 )

            #ifdef OUTPUT_BIASED
            blockC00 += *biasPtr;
            blockC10 += *(biasPtr + 1);
            #endif

            blockC00 = activation_function_half16(blockC00, in_m, in_n);
            blockC10 = activation_function_half16(blockC10, in_m, in_n);

            for (unsigned i = 0; i < 16 ; i++)
            {
                out[( 0+i) * OUT_SLICE_PITCH] = blockC00[i];
            }
            for (unsigned i = 0; i < OUT_DEPTH % 16 ; i++)
            {
                out[(16+i) * OUT_SLICE_PITCH] = blockC10[i];
            }
#else
            #ifdef OUTPUT_BIASED
            blockC00 += *biasPtr;
            #endif

            blockC00 = activation_function_half16(blockC00, in_m, in_n);

            for (unsigned i = 0; i < OUT_DEPTH % 16 ; i++)
            {
                out[( 0+i) * OUT_SLICE_PITCH] = blockC00[i];
            }
#endif
        }
#endif
    }

}
#endif

#ifdef __convolution_f32

#define TILE_M          2
#define TILE_K          KERNEL_WIDTH
#define TILE_N          32

__attribute__((intel_reqd_sub_group_size(8)))
__kernel void convolution_f32(
    const __global float *src0,
    __global float *dst,
    const __global float *src1,
    const __global float *bias)
{
    const unsigned group_x = get_group_id(0);
    const unsigned group_y = get_group_id(1);
    const unsigned global_x = get_global_id(0);
    const unsigned global_y = get_global_id(1);
    const unsigned global_z = get_global_id(2);
    unsigned interleaved_y;
    unsigned kernel_y;
    unsigned kernel_idx;

    // Result ctile (*dst) is M rows x N columns
    // LWG size is 1x8.  Thus each thread calculates 8*M rows x N cols of ctile.
    float8  blockC00 = 0.f;
    float8  blockC10 = 0.f;
    float8  blockC20 = 0.f;
    float8  blockC30 = 0.f;
    float8  blockC01 = 0.f;
    float8  blockC11 = 0.f;
    float8  blockC21 = 0.f;
    float8  blockC31 = 0.f;

    // Src0 (patch input) is directly used as atile.
    // Each work item points to the start of a different patch.
    // atile is M rows x K columns.
    int src0_read_offset0 = INPUT_BATCH_PITCH * global_z                            // batch offset
     + ( ( ( global_y * TILE_M + 0 ) / OUT_WIDTH ) * STRIDE_Y * INPUT_ROW_PITCH )   // y offset
     + ( ( ( global_y * TILE_M + 0 ) % OUT_WIDTH ) * STRIDE_X );                    // x offset
    int src0_read_offset1 = INPUT_BATCH_PITCH * global_z                            // batch offset
     + ( ( ( global_y * TILE_M + 1 ) / OUT_WIDTH ) * STRIDE_Y * INPUT_ROW_PITCH )   // y offset
     + ( ( ( global_y * TILE_M + 1 ) % OUT_WIDTH ) * STRIDE_X );                    // x offset

    // Src1 (filter) is directly used as btile.
    // It starts at the top of src1 and walks down.
    // btile is K rows x N columns.
    const __global float *src1_read = src1 + ( global_x * TILE_N * 2);

#define DOT_PRODUCT_8( _result, _rowA, colB )    \
    {   \
        _result.s0 = mad( _rowA, sub_group_broadcast( colB, 0 ), _result.s0 );  \
        _result.s1 = mad( _rowA, sub_group_broadcast( colB, 1 ), _result.s1 );  \
        _result.s2 = mad( _rowA, sub_group_broadcast( colB, 2 ), _result.s2 );  \
        _result.s3 = mad( _rowA, sub_group_broadcast( colB, 3 ), _result.s3 );  \
        _result.s4 = mad( _rowA, sub_group_broadcast( colB, 4 ), _result.s4 );  \
        _result.s5 = mad( _rowA, sub_group_broadcast( colB, 5 ), _result.s5 );  \
        _result.s6 = mad( _rowA, sub_group_broadcast( colB, 6 ), _result.s6 );  \
        _result.s7 = mad( _rowA, sub_group_broadcast( colB, 7 ), _result.s7 );  \
    }
    typedef CAT( float, KERNEL_WIDTH ) float_t;

    // Walk DOWN src0 (patch 0, 1, 2, ...) and DOWN src1.
    // Inner loop loads and FMADs one row (KERNEL_WIDTH) of each input patch
    // and KERNEL_WIDTH/2 rows of interleaved filter.
    unsigned patch_depth = 0;
    do
    {
        unsigned patch_row = 0;
        do
        {
            // Load atile and btile.
            // Kernel data is partially interleaved.  Every 2 rows are interleaved at float8 granularity.
            // The exception is that if KERNEL_WIDTH is odd the last row is not interleaved.  The non
            // interleaved row is padded with zero to ensure same size as interleaved rows. This
            // interleaving is done to ensure 0% GDR bank conflicts.  For example, this is how the
            // kernel data would be arranged before/after interleaving for KERNEL_WIDTH=3.
            // (0, 0) (8, 0) (16, 0) (24, 0) ...       (0, 0) (0, 1) (8, 0) (0, 1) (16, 0) (0, 1) (24, 0) ..
            // (0, 1) (8, 1) (16, 1) (24, 1) ... =>    (0, 2) (8, 2) (16, 2) (24, 2) ...
            // (0, 2) (8, 2) (16, 2) (24, 2) ...       ...
            // ...
            const bool kernel_width_is_odd = KERNEL_WIDTH % 2 == 1;

            const __global float *src0_read0 = src0 + src0_read_offset0;
            const __global float *src0_read1 = src0 + src0_read_offset1;
            
            float blockA00[KERNEL_WIDTH];
            float blockA01[KERNEL_WIDTH];
            
            // in case the data is not aligned to sizeof(T)*KERNEL_WIDTH we need to use vload or set the data in a loop
            {
                unsigned i = 0;
                LOOP(KERNEL_WIDTH, i, 
                {
                    blockA00[i] = src0_read0[i];
                    blockA01[i] = src0_read1[i];
                } )
            }

            float*  pblockA00 = (float*)(&blockA00);
            float*  pblockA01 = (float*)(&blockA01);

            src0_read_offset0 += INPUT_ROW_PITCH;
            src0_read_offset1 += INPUT_ROW_PITCH;


            float blockB00[KERNEL_WIDTH*4];
            float8* p8BlockB00 = (float8*)blockB00;
            float4* p4BlockB00 = (float4*)blockB00;
            float*  pBlockB00 =  (float* )blockB00;

            interleaved_y = 0;
            LOOP(KERNEL_WIDTH_DIV2, interleaved_y,
            {
                p8BlockB00[interleaved_y] = as_float8( intel_sub_group_block_read8( (const __global uint*)src1_read ) );
                src1_read += WIDTH1 * 2;
            } )
            if ( kernel_width_is_odd )
            {
                p4BlockB00[KERNEL_WIDTH - 1] = as_float4( intel_sub_group_block_read4( (const __global uint*)src1_read ) );
                src1_read += WIDTH1 * 2;
            }

            // Perform MADs
            kernel_idx = 0;
            interleaved_y = 0;
            LOOP(KERNEL_WIDTH_DIV2, interleaved_y,
            {
                kernel_y = interleaved_y * 2;
                DOT_PRODUCT_8( blockC00, pblockA00[kernel_y    ], pBlockB00[kernel_idx] );
                DOT_PRODUCT_8( blockC01, pblockA01[kernel_y    ], pBlockB00[kernel_idx] ); kernel_idx++;
                DOT_PRODUCT_8( blockC00, pblockA00[kernel_y + 1], pBlockB00[kernel_idx] );
                DOT_PRODUCT_8( blockC01, pblockA01[kernel_y + 1], pBlockB00[kernel_idx] ); kernel_idx++;
                DOT_PRODUCT_8( blockC10, pblockA00[kernel_y    ], pBlockB00[kernel_idx] );
                DOT_PRODUCT_8( blockC11, pblockA01[kernel_y    ], pBlockB00[kernel_idx] ); kernel_idx++;
                DOT_PRODUCT_8( blockC10, pblockA00[kernel_y + 1], pBlockB00[kernel_idx] );
                DOT_PRODUCT_8( blockC11, pblockA01[kernel_y + 1], pBlockB00[kernel_idx] ); kernel_idx++;
                DOT_PRODUCT_8( blockC20, pblockA00[kernel_y    ], pBlockB00[kernel_idx] );
                DOT_PRODUCT_8( blockC21, pblockA01[kernel_y    ], pBlockB00[kernel_idx] ); kernel_idx++;
                DOT_PRODUCT_8( blockC20, pblockA00[kernel_y + 1], pBlockB00[kernel_idx] );
                DOT_PRODUCT_8( blockC21, pblockA01[kernel_y + 1], pBlockB00[kernel_idx] ); kernel_idx++;
                DOT_PRODUCT_8( blockC30, pblockA00[kernel_y    ], pBlockB00[kernel_idx] );
                DOT_PRODUCT_8( blockC31, pblockA01[kernel_y    ], pBlockB00[kernel_idx] ); kernel_idx++;
                DOT_PRODUCT_8( blockC30, pblockA00[kernel_y + 1], pBlockB00[kernel_idx] );
                DOT_PRODUCT_8( blockC31, pblockA01[kernel_y + 1], pBlockB00[kernel_idx] ); kernel_idx++;
            } )
            if ( kernel_width_is_odd )
            {
                kernel_y = interleaved_y * 2;
                DOT_PRODUCT_8( blockC00, pblockA00[kernel_y], pBlockB00[kernel_idx] );
                DOT_PRODUCT_8( blockC01, pblockA01[kernel_y], pBlockB00[kernel_idx] ); kernel_idx++;
                DOT_PRODUCT_8( blockC10, pblockA00[kernel_y], pBlockB00[kernel_idx] );
                DOT_PRODUCT_8( blockC11, pblockA01[kernel_y], pBlockB00[kernel_idx] ); kernel_idx++;
                DOT_PRODUCT_8( blockC20, pblockA00[kernel_y], pBlockB00[kernel_idx] );
                DOT_PRODUCT_8( blockC21, pblockA01[kernel_y], pBlockB00[kernel_idx] ); kernel_idx++;
                DOT_PRODUCT_8( blockC30, pblockA00[kernel_y], pBlockB00[kernel_idx] );
                DOT_PRODUCT_8( blockC31, pblockA01[kernel_y], pBlockB00[kernel_idx] ); kernel_idx++;
            }
        }

        //while( ++patch_row < 1 ); //debug
        while( ++patch_row < KERNEL_HEIGHT );

        src0_read_offset0 += INPUT_SLICE_PITCH - ( KERNEL_HEIGHT * INPUT_ROW_PITCH ); // reset to start of next slice of patch
        src0_read_offset1 += INPUT_SLICE_PITCH - ( KERNEL_HEIGHT * INPUT_ROW_PITCH ); // reset to start of next slice of patch
    }
    //while ( ++patch_depth < 1 );  //debug
    while ( ++patch_depth < INPUT_DEPTH );

    // Dst resembles a cube of width x height x (output channel * batches).  Each tile writes:
    // (SIMD * TILE_M) x 1 x TILE_N.  Partial writes most likely generated if padding used.
    __global float *out0 = dst + OUT_OFFSET
     + global_z * OUT_BATCH_PITCH                                                       // batch offset
     + ( group_x * TILE_N ) * OUT_SLICE_PITCH                                           // channel offset
     + ( ( global_y * TILE_M + 0 ) / OUT_WIDTH ) * OUT_ROW_PITCH // y offset
     + ( ( global_y * TILE_M + 0 ) % OUT_WIDTH );               // x offset
    __global float *out1 = dst + OUT_OFFSET
     + global_z * OUT_BATCH_PITCH                                                       // batch offset
     + ( group_x * TILE_N ) * OUT_SLICE_PITCH                                           // channel offset
     + ( ( global_y * TILE_M + 1 ) / OUT_WIDTH ) * OUT_ROW_PITCH // y offset
     + ( ( global_y * TILE_M + 1 ) % OUT_WIDTH );               // x offset

    #ifdef OUTPUT_BIASED
    __global float8* biasPtr = (__global float8*) (bias + global_z * OUT_DEPTH + group_x * TILE_N);
    #endif
    unsigned m = NL_M;
    unsigned n = NL_N;

    float in_m = *(float*)(&m);
    float in_n = *(float*)(&n);
    if( global_y * TILE_M < OUT_WIDTH * OUT_HEIGHT )
    {
        if ( ( OUT_DEPTH % TILE_N ) == 0 )
        {
            #ifdef OUTPUT_BIASED
            blockC00 += *biasPtr;
            blockC10 += *(biasPtr + 1);
            blockC20 += *(biasPtr + 2);
            blockC30 += *(biasPtr + 3);
            #endif

            blockC00 = activation_function_float8(blockC00, in_m, in_n);
            blockC10 = activation_function_float8(blockC10, in_m, in_n);
            blockC20 = activation_function_float8(blockC20, in_m, in_n);
            blockC30 = activation_function_float8(blockC30, in_m, in_n);

            for( unsigned i = 0; i < 8; i++ )
            {
                out0[( 0+i) * OUT_SLICE_PITCH] = blockC00[i];
                out0[( 8+i) * OUT_SLICE_PITCH] = blockC10[i];
                out0[(16+i) * OUT_SLICE_PITCH] = blockC20[i];
                out0[(24+i) * OUT_SLICE_PITCH] = blockC30[i];
            }
        }
        else
        {
            if ( ( global_x + 1 ) < get_global_size(0) )
            {
                #ifdef OUTPUT_BIASED
                blockC00 += *biasPtr;
                blockC10 += *(biasPtr + 1);
                blockC20 += *(biasPtr + 2);
                blockC30 += *(biasPtr + 3);
                #endif

                blockC00 = activation_function_float8(blockC00, in_m, in_n);
                blockC10 = activation_function_float8(blockC10, in_m, in_n);
                blockC20 = activation_function_float8(blockC20, in_m, in_n);
                blockC30 = activation_function_float8(blockC30, in_m, in_n);

                for ( unsigned i = 0; i < 8; i++ )
                {
                    out0[( 0+i) * OUT_SLICE_PITCH] = blockC00[i];
                    out0[( 8+i) * OUT_SLICE_PITCH] = blockC10[i];
                    out0[(16+i) * OUT_SLICE_PITCH] = blockC20[i];
                    out0[(24+i) * OUT_SLICE_PITCH] = blockC30[i];
                }
            }
            else
            {
                if ( ( OUT_DEPTH % TILE_N ) >= 24 )
                {
                    #ifdef OUTPUT_BIASED
                    blockC00 += *biasPtr;
                    blockC10 += *(biasPtr + 1);
                    blockC20 += *(biasPtr + 2);
                    if (( OUT_DEPTH % TILE_N) > 24 ) blockC30 += *(biasPtr + 3);
                    #endif

                    blockC00 = activation_function_float8(blockC00, in_m, in_n);
                    blockC10 = activation_function_float8(blockC10, in_m, in_n);
                    blockC20 = activation_function_float8(blockC20, in_m, in_n);

                    for (unsigned i = 0; i < 8; i++)
                    {
                        out0[( 0+i) * OUT_SLICE_PITCH] = blockC00[i];
                        out0[( 8+i) * OUT_SLICE_PITCH] = blockC10[i];
                        out0[(16+i) * OUT_SLICE_PITCH] = blockC20[i];
                    }

                    // remaining output channels
                    for (unsigned i = 0; i < OUT_DEPTH % 8; i++)
                    {
                        out0[(24+i) * OUT_SLICE_PITCH] = activation_function(blockC30[i], NL_M, NL_N);
                    }
                }
                else if ( ( OUT_DEPTH % TILE_N ) >= 16 )
                {
                    #ifdef OUTPUT_BIASED
                    blockC00 += *biasPtr;
                    blockC10 += *(biasPtr + 1);
                    if (( OUT_DEPTH % TILE_N) > 16 )
                        blockC20 += *(biasPtr + 2);
                    #endif

                    blockC00 = activation_function_float8(blockC00, in_m, in_n);
                    blockC10 = activation_function_float8(blockC10, in_m, in_n);

                    for (unsigned i = 0; i < 8; i++)
                    {
                        out0[( 0+i) * OUT_SLICE_PITCH] = blockC00[i];
                        out0[( 8+i) * OUT_SLICE_PITCH] = blockC10[i];
                    }

                    for (unsigned i = 0; i < OUT_DEPTH % 8; i++)
                    {
                        out0[(16+i) * OUT_SLICE_PITCH] = activation_function(blockC20[i], NL_M, NL_N);

                    }
                }
                else if ( ( OUT_DEPTH % TILE_N ) >= 8 )
                {
                    #ifdef OUTPUT_BIASED
                    blockC00 += *biasPtr;
                    if (( OUT_DEPTH % TILE_N) > 8 )
                        blockC10 += *(biasPtr + 1);
                    #endif

                    blockC00 = activation_function_float8(blockC00, in_m, in_n);

                    for (unsigned i = 0; i < 8; i++)
                    {
                        out0[( 0+i) * OUT_SLICE_PITCH] = blockC00[i];
                    }

                    for (unsigned i = 0; i < OUT_DEPTH % 8; i++)
                    {
                        out0[(8+i) * OUT_SLICE_PITCH] = activation_function(blockC10[i], NL_M, NL_N);
                    }
                }
                else
                {
                    #ifdef OUTPUT_BIASED
                    blockC00 += *biasPtr;
                    #endif
                    for (unsigned i = 0; i < OUT_DEPTH % 8; i++)
                    {
                        out0[( 0+i) * OUT_SLICE_PITCH] = activation_function(blockC00[i], NL_M, NL_N);
                    }
                }
            }
        }
    }

    if ((global_y * TILE_M + 1) < OUT_WIDTH * OUT_HEIGHT )
    {
        if ( ( OUT_DEPTH % TILE_N ) == 0 )
        {
            #ifdef OUTPUT_BIASED
            blockC01 += *biasPtr;
            blockC11 += *(biasPtr + 1);
            blockC21 += *(biasPtr + 2);
            blockC31 += *(biasPtr + 3);
            #endif

            blockC01 = activation_function_float8(blockC01, in_m, in_n);
            blockC11 = activation_function_float8(blockC11, in_m, in_n);
            blockC21 = activation_function_float8(blockC21, in_m, in_n);
            blockC31 = activation_function_float8(blockC31, in_m, in_n);

            for( unsigned i = 0; i < 8; i++ )
            {
                out1[( 0+i) * OUT_SLICE_PITCH] = blockC01[i];
                out1[( 8+i) * OUT_SLICE_PITCH] = blockC11[i];
                out1[(16+i) * OUT_SLICE_PITCH] = blockC21[i];
                out1[(24+i) * OUT_SLICE_PITCH] = blockC31[i];
            }
        }
        else
        {
            if ( ( global_x + 1 ) < get_global_size(0) )
            {
                #ifdef OUTPUT_BIASED
                blockC01 += *biasPtr;
                blockC11 += *(biasPtr + 1);
                blockC21 += *(biasPtr + 2);
                blockC31 += *(biasPtr + 3);
                #endif

                blockC01 = activation_function_float8(blockC01, in_m, in_n);
                blockC11 = activation_function_float8(blockC11, in_m, in_n);
                blockC21 = activation_function_float8(blockC21, in_m, in_n);
                blockC31 = activation_function_float8(blockC31, in_m, in_n);

                for ( unsigned i = 0; i < 8; i++ )
                {
                    out1[( 0+i) * OUT_SLICE_PITCH] = blockC01[i];
                    out1[( 8+i) * OUT_SLICE_PITCH] = blockC11[i];
                    out1[(16+i) * OUT_SLICE_PITCH] = blockC21[i];
                    out1[(24+i) * OUT_SLICE_PITCH] = blockC31[i];
                }
            }
            else
            {
                if ( ( OUT_DEPTH % TILE_N ) >= 24 )
                {
                    #ifdef OUTPUT_BIASED
                    blockC01 += *biasPtr;
                    blockC11 += *(biasPtr + 1);
                    blockC21 += *(biasPtr + 2);
                    if ( ( OUT_DEPTH % TILE_N ) > 24 ) blockC31 += *(biasPtr + 3);
                    #endif

                    blockC01 = activation_function_float8(blockC01, in_m, in_n);
                    blockC11 = activation_function_float8(blockC11, in_m, in_n);
                    blockC21 = activation_function_float8(blockC21, in_m, in_n);

                    for (unsigned i = 0; i < 8; i++)
                    {
                        out1[( 0+i) * OUT_SLICE_PITCH] = blockC01[i];
                        out1[( 8+i) * OUT_SLICE_PITCH] = blockC11[i];
                        out1[(16+i) * OUT_SLICE_PITCH] = blockC21[i];
                    }

                    // Remaining channels
                    for (unsigned i = 0; i < OUT_DEPTH % 8; i++)
                    {
                        out1[(24+i) * OUT_SLICE_PITCH] = activation_function(blockC31[i], NL_M, NL_N);
                    }
                }
                else if ( ( OUT_DEPTH % TILE_N ) >= 16 )
                {
                    #ifdef OUTPUT_BIASED
                    blockC01 += *biasPtr;
                    blockC11 += *(biasPtr + 1);
                    if ( ( OUT_DEPTH % TILE_N ) > 16 ) blockC21 += *(biasPtr + 2);
                    #endif

                    blockC01 = activation_function_float8(blockC01, in_m, in_n);
                    blockC11 = activation_function_float8(blockC11, in_m, in_n);

                    for (unsigned i = 0; i < 8; i++)
                    {
                        out1[( 0+i) * OUT_SLICE_PITCH] = blockC01[i];
                        out1[( 8+i) * OUT_SLICE_PITCH] = blockC11[i];
                    }

                    for (unsigned i = 0; i < OUT_DEPTH % 8; i++)
                    {
                        out1[(16+i) * OUT_SLICE_PITCH] = activation_function(blockC21[i], NL_M, NL_N);
                    }
                }
                else if ( ( OUT_DEPTH % TILE_N ) >= 8 )
                {
                    #ifdef OUTPUT_BIASED
                    blockC01 += *biasPtr;
                    if ( ( OUT_DEPTH % TILE_N ) > 8 ) blockC11 += *(biasPtr + 1);
                    #endif

                    blockC01 = activation_function_float8(blockC01, in_m, in_n);

                    for (unsigned i = 0; i < 8; i++)
                    {
                        out1[( 0+i) * OUT_SLICE_PITCH] = blockC01[i];
                    }

                    for (unsigned i = 0; i < OUT_DEPTH % 8; i++)
                    {
                        out1[(8+i) * OUT_SLICE_PITCH] = activation_function(blockC11[i], NL_M, NL_N);
                    }
                }
                else
                {
                    #ifdef OUTPUT_BIASED
                    blockC01 += *biasPtr;
                    #endif

                    for (unsigned i = 0; i < OUT_DEPTH % 8; i++)
                    {
                        out1[( 0+i) * OUT_SLICE_PITCH] = activation_function(blockC01[i], NL_M, NL_N);
                    }
                }
            }
        }
    }
}
#endif
