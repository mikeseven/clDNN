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

#include "include/common.cl"
#include "include/data_types.cl"
#include "include/fetch.cl"
#include "include/mmad.cl"

#define WG_TILE_M 128  // Work-Group tile size M, Must be mutliple of 32
#define WG_TILE_N 128  // Work-Group tile size N, Must be mutliple of 32

#define MATRIX_K K
#define MATRIX_M M
#define MATRIX_N N
#define DIM_X 0
#define DIM_Y 1
#define MATRIX_SMALL_K 32
#define SG_TILE_M 32
#define SG_TILE_N 32
#define SG_SIZE 8
#define SIMD_LANE_M SG_TILE_M
#define SIMD_LANE_N (SG_TILE_N / SG_SIZE)
#define WG_SIZE (SG_SIZE * WG_TILE_N / SG_TILE_N) * (WG_TILE_M / SG_TILE_M)

__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE)))
KERNEL(Kernel_GEMM_MMAD8_32x32SG_128x128WG_SLM_INT8)(
    __global int8* const g_matrixA,
    __global char8* g_matrixC,
    __global int8* const g_matrixB,
#if BIAS_TERM
    __global BIAS_TYPE* biases,
#endif
#if QUANTIZATION_TERM
    __global float* quantizations,
#endif
#if CALIBRATION_TERM
    __global float* calibrations,
#endif
    uint split_idx)
{
    // Each work-group works to compute 128x128 tile.
    // Each work-group contains 16 sub-groups.
    // Each sub-group within the work-group works to compute a 32x32 tile.
    // 1) All work-items in WG fill SLM with tileA (128x32) and tileB (32x128).
    // 2) Each sub-group works to compute 32x32 tileC (stored in regC).
    //    Note that each work-item in the sub-group computes a 32x4 chunk of tileC.
    // 3) Repeat until tileC is fully computed (while moving tileA and tileB "windows")
    __local int8 l_workGroupTileA[(WG_TILE_M * MATRIX_SMALL_K) / sizeof(int8)];
    __local int8 l_workGroupTileB[(WG_TILE_N * MATRIX_SMALL_K) / sizeof(int8)];
    __local uint* l_workGroupTileA_uint = (__local uint*)l_workGroupTileA;

    const uint l_groupSize = get_local_size(DIM_X) * get_local_size(DIM_Y);

    // Thread IDs
    const uint g_tidY = get_global_id(DIM_Y);
    const uint g_tidX = get_global_id(DIM_X);
    const uint l_tidX = get_local_id(DIM_X);
    const uint l_tidY = get_local_id(DIM_Y);
    const uint l_tid = l_tidY * get_local_size(DIM_X) + l_tidX;

    // SubGroup IDs
    const uint sg_tid = get_sub_group_local_id();
    const uint sg_global_idX = (uint)(g_tidX / SG_SIZE);
    const uint sg_global_idY = g_tidY;
    const uint sg_local_idX = (uint)(l_tidX / SG_SIZE);
    const uint sg_local_idY = l_tidY;
    const uint sg_local_id = sg_local_idY * get_local_size(DIM_X) / SG_SIZE + sg_local_idX;

    // Registers
    // int8 x 4 x 32 = 32x32 ints for accumulators 
    int8 regC[(SIMD_LANE_M / 8) * SIMD_LANE_N] = {0}; // Each work-item responsible for 32x4 ints elts
    int8 rowA[SG_TILE_M / 8]; // each work-item will hold 1/8 of matrixA
    int8 colB; // each lane will store 32x4 piece of matrixB

    // SLM indices
    const uint l_offsetTileA = SG_TILE_M * (MATRIX_SMALL_K / sizeof(uint)) * sg_local_idY;
    const uint numElements32x32TileB = (SG_TILE_N * SG_TILE_M) / sizeof(int8);
    const uint numElements32x8TileB = numElements32x32TileB / 4;
    const uint l_offsetTileB = numElements32x32TileB * sg_local_idX;
    const uint l_offsetTileB_col0 = l_offsetTileB + sg_tid;
    const uint l_offsetTileB_col1 = l_offsetTileB + 1 * numElements32x8TileB + sg_tid;
    const uint l_offsetTileB_col2 = l_offsetTileB + 2 * numElements32x8TileB + sg_tid;
    const uint l_offsetTileB_col3 = l_offsetTileB + 3 * numElements32x8TileB + sg_tid;

    // Global indices
    uint g_offsetTileA = WG_TILE_M * (MATRIX_K / sizeof(int8)) * get_group_id(DIM_Y);
    uint g_offsetTileB = WG_TILE_N * (MATRIX_K / sizeof(int8)) * get_group_id(DIM_X);
    uint g_idxA = g_offsetTileA + l_tid * (MATRIX_K / sizeof(int8));
    uint g_idxB = g_offsetTileB + l_tid * (MATRIX_K / sizeof(int8));

    // Initial SLM setup
    {
        l_workGroupTileA[l_tid] = g_matrixA[g_idxA];
        l_workGroupTileB[l_tid] = g_matrixB[g_idxB];

        g_idxA += MATRIX_SMALL_K / sizeof(int8);
        g_idxB += MATRIX_SMALL_K / sizeof(int8);

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    __attribute__((opencl_unroll_hint(1)))
    for (uint i = 0; i < MATRIX_K / MATRIX_SMALL_K; i++)
    {
        /*
         * SLM setup - HDC read only
         */

        // Overlap HDC reads with MMAD compute
        int8 hdcReadValueA = g_matrixA[g_idxA];
        int8 hdcReadValueB = g_matrixB[g_idxB];

        g_idxA += MATRIX_SMALL_K / sizeof(int8);
        g_idxB += MATRIX_SMALL_K / sizeof(int8);

        /*
         * MMAD compute
         */

        // Read tile A from SLM to regA
        uint l_offsetTileATemp = l_offsetTileA;
        __attribute__((opencl_unroll_hint(SG_TILE_M / 8)))
        for (uint j = 0; j < (SG_TILE_M / 8); ++j)
        {
            rowA[j] = as_int8(SLM_BLOCK_READ_8(&l_workGroupTileA_uint[l_offsetTileATemp]));
            l_offsetTileATemp += 8 * (MATRIX_SMALL_K / sizeof(uint));
        }

        // Read tile B from SLM to regB and compute MMAD
        colB = l_workGroupTileB[l_offsetTileB_col0];
        __attribute__((opencl_unroll_hint(SG_TILE_M / 8)))
        for (uint j = 0; j < (SG_TILE_M / 8); ++j)
        {
            // Compute partial C
            regC[0*(SIMD_LANE_M / 8) + j] = MMAD_8x8(regC[0*(SIMD_LANE_M / 8) + j], rowA[j], colB);
        }

        colB = l_workGroupTileB[l_offsetTileB_col1];
        __attribute__((opencl_unroll_hint(SG_TILE_M / 8)))
        for (uint j = 0; j < (SG_TILE_M / 8); ++j)
        {
            // Compute partial C
            regC[1*(SIMD_LANE_M / 8) + j] = MMAD_8x8(regC[1*(SIMD_LANE_M / 8) + j], rowA[j], colB);
        }

        colB = l_workGroupTileB[l_offsetTileB_col2];
        __attribute__((opencl_unroll_hint(SG_TILE_M / 8)))
        for (uint j = 0; j < (SG_TILE_M / 8); ++j)
        {
            // Compute partial C
            regC[2*(SIMD_LANE_M / 8) + j] = MMAD_8x8(regC[2*(SIMD_LANE_M / 8) + j], rowA[j], colB);
        }

        colB = l_workGroupTileB[l_offsetTileB_col3];
        __attribute__((opencl_unroll_hint(SG_TILE_M / 8)))
        for (uint j = 0; j < (SG_TILE_M / 8); ++j)
        {
            // Compute partial C
            regC[3*(SIMD_LANE_M / 8) + j] = MMAD_8x8(regC[3*(SIMD_LANE_M / 8) + j], rowA[j], colB);
        }

        /*
         * SLM setup - SLM write only
         */

        barrier(CLK_LOCAL_MEM_FENCE);

        l_workGroupTileA[l_tid] = hdcReadValueA;
        l_workGroupTileB[l_tid] = hdcReadValueB;

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if(get_global_id(0) == 0 && get_global_id(1) == 0 && get_global_id(2) == 0)
    {
        printf("accumulator: %d\n", (int)(regC[0][0]));
    }

    // Write final accumulated values
    uint cOffset = sg_global_idX * ((MATRIX_M / 8) * SG_TILE_N) + sg_global_idY * (SG_TILE_M / 8) +
                   sg_tid * (MATRIX_M / 8);
    __attribute__((opencl_unroll_hint(SIMD_LANE_N)))
    for (uint i = 0; i < (SIMD_LANE_N); ++i)
    {
        __attribute__((opencl_unroll_hint(SIMD_LANE_M / 8)))
        for (uint j = 0; j < (SIMD_LANE_M / 8); ++j)
        {
            char8 to_return;
            for(uint z = 0; z < 8; z++)
            {
                int accumulator = regC[i*(SIMD_LANE_M / 8) + j][z];
                int byte_offset = cOffset * 8 + j * 8 + z;
                int f = (byte_offset % 32) + ((byte_offset / OUT_F_BLOCK_PITCH) * 32);
#if BIAS_TERM
                const unsigned bias_index = f;
#if CALIBRATION_TERM
                accumulator = (UNIT_TYPE)round(((float)accumulator * quantizations[f] * I_QF + biases[bias_index]) * calibrations[f]);
#else  // CALIBRATION_TERM
                accumulator = (UNIT_TYPE)round(((float)accumulator * quantizations[f] * I_QF + biases[bias_index]) * O_QF);
#endif // CALIBRATION_TERM

#endif // BIAS_TERM

                to_return[z] = ACTIVATION(convert_char(accumulator), NL_M, NL_N);
            }
//            if( (cOffset + j) * 8 < OUTPUT_LENGTH)
//            {
                g_matrixC[cOffset + j] = to_return;
//            }

        }
        cOffset += SG_SIZE * (MATRIX_M / 8);
    }
}