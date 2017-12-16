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

// --------------------------------------------------------------------------------------------------------------------------------
// L3_SIMD_4x8
// Input matrices dimensions: M x K x N
// Output matrix dimensions: M x N
// --------------------------------------------------------------------------------------------------------------------------------

#include "include/data_types.cl"
#include "include/activation_functions.cl"

#define DOT4i_sep( _result, _A, _B, i)					\
    {	\
	if (!upperHalf) {							\
	_result = mad(_A.s0, intel_sub_group_shuffle( _B.s0, i), _result);	\
	_result = mad(_A.s1, intel_sub_group_shuffle( _B.s1, i), _result);	\
	_result = mad(_A.s2, intel_sub_group_shuffle( _B.s2, i), _result);	\
	_result = mad(_A.s3, intel_sub_group_shuffle( _B.s3, i), _result);	\
	} else {  \
	_result = mad(_A.s0, intel_sub_group_shuffle( _B.s0, i+8), _result);	\
	_result = mad(_A.s1, intel_sub_group_shuffle( _B.s1, i+8), _result);	\
	_result = mad(_A.s2, intel_sub_group_shuffle( _B.s2, i+8), _result);	\
	_result = mad(_A.s3, intel_sub_group_shuffle( _B.s3, i+8), _result);	\
	}   \
    }

#define DOT4i( _result, _A, _B, i)					\
    {	\
	_result = mad(_A.s0, intel_sub_group_shuffle( _B.s0, (upperHalf?i+8:i)), _result);	\
	_result = mad(_A.s1, intel_sub_group_shuffle( _B.s1, (upperHalf?i+8:i)), _result);	\
	_result = mad(_A.s2, intel_sub_group_shuffle( _B.s2, (upperHalf?i+8:i)), _result);	\
	_result = mad(_A.s3, intel_sub_group_shuffle( _B.s3, (upperHalf?i+8:i)), _result);	\
    }

#define UNIT_TYPE_2 CAT(UNIT_TYPE, 2)
#define UNIT_TYPE_4 CAT(UNIT_TYPE, 4)
#define UNIT_TYPE_8 CAT(UNIT_TYPE, 8)

__attribute__((reqd_work_group_size(8, 2, 8)))
__attribute__((intel_reqd_sub_group_size(16)))
KERNEL(convolution_gpu_winograd_2x3_s1_fused)
(
    __global INPUT0_TYPE* I, 
    __global OUTPUT_TYPE* O, 
    __global FILTER_TYPE* U, 
#if BIAS_TERM
    const __global UNIT_TYPE * bias,
#endif 
    uint split_idx)
{
    //               (DxC2)x(UxWx8c)
	const uint slmSize = (4*2)*(2*16*2);
    __local UNIT_TYPE_4 V[slmSize*2]; // 8 KB

    /* These constants are defined as precompiler macros during compilation. */
     const uint WC = W*INPUT0_FEATURE_NUM; 
	 const uint HW = H*W; 
     const uint HWC = H*WC; 
     const uint WC4 = WC >> 2; 
     const uint K16 = FILTER_OFM_NUM >> 4; 
     const uint C4 = INPUT0_FEATURE_NUM >> 2; 
     const uint K2 = FILTER_OFM_NUM >> 1; 
     const uint QK2 = Q*K2; 
     const uint QK = Q*FILTER_OFM_NUM; 
     const uint PQK = P*QK; 
    
	const uint upperHalf = get_local_id(1);
    uint gx = get_group_id(0);
    uint gy = get_group_id(1)*2+upperHalf;
    uint gz = get_group_id(2);
    uint gk = gz % K16;
    uint gn = gz / K16;

	uint glbx = get_global_id(0);
	#define lx get_local_id(0)
	#define lz get_local_id(2)

    uint lxd4 = lx >> 2;
    uint lxm4 = lx % 4;

    uint lzd4 = lz >> 2;
    uint lzm4 = lz % 4;

    // Load 16x6 input tile, with 2 pixel overlap in X and y.
    // Compute 14x4 output tile.
    // Load 8 filters / thread.
    // 8 threads total: 2 filters x 4 winograd components. 16 filters total.
    int x = gx*14 + lz*2 + lxd4 - px;
    int y = gy*4 - py;
    uint k = gk*16 + lzd4*8;
    
    // #                                  x->
    // #     M0    M1    M2    M3    M4    M5    M6
    // #   +------------------------------------------
    // # u | 0 1 | 0 1 | 0 1 | 0 1 | 0 1 | 0 1 | 0 1 |
    // # | | 2 3 | 2 3 | 2 3 | 2 3 | 2 3 | 2 3 | 2 3 |
    // # v
    // #

    UNIT_TYPE_4 M0 = (UNIT_TYPE_4)(UNIT_VAL_ZERO, UNIT_VAL_ZERO, UNIT_VAL_ZERO, UNIT_VAL_ZERO);
    UNIT_TYPE_4 M1 = (UNIT_TYPE_4)(UNIT_VAL_ZERO, UNIT_VAL_ZERO, UNIT_VAL_ZERO, UNIT_VAL_ZERO);
    UNIT_TYPE_4 M2 = (UNIT_TYPE_4)(UNIT_VAL_ZERO, UNIT_VAL_ZERO, UNIT_VAL_ZERO, UNIT_VAL_ZERO);
    UNIT_TYPE_4 M3 = (UNIT_TYPE_4)(UNIT_VAL_ZERO, UNIT_VAL_ZERO, UNIT_VAL_ZERO, UNIT_VAL_ZERO);
    UNIT_TYPE_4 M4 = (UNIT_TYPE_4)(UNIT_VAL_ZERO, UNIT_VAL_ZERO, UNIT_VAL_ZERO, UNIT_VAL_ZERO);
    UNIT_TYPE_4 M5 = (UNIT_TYPE_4)(UNIT_VAL_ZERO, UNIT_VAL_ZERO, UNIT_VAL_ZERO, UNIT_VAL_ZERO);
    UNIT_TYPE_4 M6 = (UNIT_TYPE_4)(UNIT_VAL_ZERO, UNIT_VAL_ZERO, UNIT_VAL_ZERO, UNIT_VAL_ZERO);

#if INPUT0_LAYOUT_BYXF
    const __global UNIT_TYPE_4 *I_load = (const __global UNIT_TYPE_4*)&I[gn*HWC + ((uint)y)*WC + ((uint)x)*INPUT0_FEATURE_NUM];
#else
	const __global UNIT_TYPE *I_load = (const __global UNIT_TYPE*)&I[gn*HWC + ((uint)y)*W + ((uint)x)];
#endif


    uint lxm2 = lx % 2;
    uint lxb1 = (lx & 2)/2;
                                     
    __local UNIT_TYPE_4 *V_write = &V[lxb1*256 + lz*4 + lxd4*2 + lxm2 + slmSize*upperHalf];
    __local const UNIT_TYPE_4 *V_read = &V[lzm4*64 + lx + slmSize*upperHalf];

    uint2 coordU0;
    coordU0.x = (lzm4*24 + k*12);
    coordU0.y = 0;

	
    __attribute__((opencl_unroll_hint(1)))
    for (uint c = lxm4; c < C4_up16; c += 4) {

        // 2*14 * 3 * 16 = 1344 MADs

        // Transform HxW x C        -> DxUxW x C
        //           6x16x16 inputs -> 4x2x16x16 winograd components.
        {
			bool x_in =  0 <= x && x < W;
			bool y0_in = 0 <= (y + 0) && (y + 0) < H && x_in;
			bool y1_in = 0 <= (y + 1) && (y + 1) < H && x_in;
			bool y2_in = 0 <= (y + 2) && (y + 2) < H && x_in;
			bool y3_in = 0 <= (y + 3) && (y + 3) < H && x_in;
			bool y4_in = 0 <= (y + 4) && (y + 4) < H && x_in;
			bool y5_in = 0 <= (y + 5) && (y + 5) < H && x_in;

			#if INPUT0_LAYOUT_BYXF
    const __global UNIT_TYPE_4 *I_load_0 = &I_load[0*WC4];
    const __global UNIT_TYPE_4 *I_load_1 = &I_load[1*WC4];
    const __global UNIT_TYPE_4 *I_load_2 = &I_load[2*WC4];
    const __global UNIT_TYPE_4 *I_load_3 = &I_load[3*WC4];
    const __global UNIT_TYPE_4 *I_load_4 = &I_load[4*WC4];
    const __global UNIT_TYPE_4 *I_load_5 = &I_load[5*WC4];
#else
    const __global UNIT_TYPE *I_load_0 = &I_load[0*W]; //y0_in ? &I_load[0*W] : zeros4;
    const __global UNIT_TYPE *I_load_1 = &I_load[1*W]; //y1_in ? &I_load[1*W] : zeros4;
    const __global UNIT_TYPE *I_load_2 = &I_load[2*W]; //y2_in ? &I_load[2*W] : zeros4;
    const __global UNIT_TYPE *I_load_3 = &I_load[3*W]; //y3_in ? &I_load[3*W] : zeros4;
    const __global UNIT_TYPE *I_load_4 = &I_load[4*W]; //y4_in ? &I_load[4*W] : zeros4;
    const __global UNIT_TYPE *I_load_5 = &I_load[5*W]; //y5_in ? &I_load[5*W] : zeros4;
#endif


            // Workgroup loads 6x16x16 inputs.
#if INPUT0_LAYOUT_BYXF
            UNIT_TYPE_4 I0 =  y0_in ? I_load_0[c]:(UNIT_TYPE_4)(UNIT_VAL_ZERO,UNIT_VAL_ZERO,UNIT_VAL_ZERO,UNIT_VAL_ZERO);
            UNIT_TYPE_4 I1 =  y1_in ? I_load_1[c]:(UNIT_TYPE_4)(UNIT_VAL_ZERO,UNIT_VAL_ZERO,UNIT_VAL_ZERO,UNIT_VAL_ZERO);
            UNIT_TYPE_4 I2 =  y2_in ? I_load_2[c]:(UNIT_TYPE_4)(UNIT_VAL_ZERO,UNIT_VAL_ZERO,UNIT_VAL_ZERO,UNIT_VAL_ZERO);
            UNIT_TYPE_4 I3 =  y3_in ? I_load_3[c]:(UNIT_TYPE_4)(UNIT_VAL_ZERO,UNIT_VAL_ZERO,UNIT_VAL_ZERO,UNIT_VAL_ZERO);
            UNIT_TYPE_4 I4 =  y4_in ? I_load_4[c]:(UNIT_TYPE_4)(UNIT_VAL_ZERO,UNIT_VAL_ZERO,UNIT_VAL_ZERO,UNIT_VAL_ZERO);
            UNIT_TYPE_4 I5 =  y5_in ? I_load_5[c]:(UNIT_TYPE_4)(UNIT_VAL_ZERO,UNIT_VAL_ZERO,UNIT_VAL_ZERO,UNIT_VAL_ZERO);
#else
            UNIT_TYPE_4 I0 = y0_in ? (UNIT_TYPE_4)(I_load_0[c*HW*4], I_load_0[c*HW*4+HW], I_load_0[c*HW*4+HW*2], I_load_0[c*HW*4+HW*3]):(UNIT_TYPE_4)(UNIT_VAL_ZERO,UNIT_VAL_ZERO,UNIT_VAL_ZERO,UNIT_VAL_ZERO);
            UNIT_TYPE_4 I1 = y1_in ? (UNIT_TYPE_4)(I_load_1[c*HW*4], I_load_1[c*HW*4+HW], I_load_1[c*HW*4+HW*2], I_load_1[c*HW*4+HW*3]):(UNIT_TYPE_4)(UNIT_VAL_ZERO,UNIT_VAL_ZERO,UNIT_VAL_ZERO,UNIT_VAL_ZERO);
            UNIT_TYPE_4 I2 = y2_in ? (UNIT_TYPE_4)(I_load_2[c*HW*4], I_load_2[c*HW*4+HW], I_load_2[c*HW*4+HW*2], I_load_2[c*HW*4+HW*3]):(UNIT_TYPE_4)(UNIT_VAL_ZERO,UNIT_VAL_ZERO,UNIT_VAL_ZERO,UNIT_VAL_ZERO);
            UNIT_TYPE_4 I3 = y3_in ? (UNIT_TYPE_4)(I_load_3[c*HW*4], I_load_3[c*HW*4+HW], I_load_3[c*HW*4+HW*2], I_load_3[c*HW*4+HW*3]):(UNIT_TYPE_4)(UNIT_VAL_ZERO,UNIT_VAL_ZERO,UNIT_VAL_ZERO,UNIT_VAL_ZERO);
            UNIT_TYPE_4 I4 = y4_in ? (UNIT_TYPE_4)(I_load_4[c*HW*4], I_load_4[c*HW*4+HW], I_load_4[c*HW*4+HW*2], I_load_4[c*HW*4+HW*3]):(UNIT_TYPE_4)(UNIT_VAL_ZERO,UNIT_VAL_ZERO,UNIT_VAL_ZERO,UNIT_VAL_ZERO);
            UNIT_TYPE_4 I5 = y5_in ? (UNIT_TYPE_4)(I_load_5[c*HW*4], I_load_5[c*HW*4+HW], I_load_5[c*HW*4+HW*2], I_load_5[c*HW*4+HW*3]):(UNIT_TYPE_4)(UNIT_VAL_ZERO,UNIT_VAL_ZERO,UNIT_VAL_ZERO,UNIT_VAL_ZERO);
#endif

            // Compute Winograd f2x3 data transform and store components in SLM.
            V_write[0*64] = I0 - I2;
            V_write[1*64] = I1 + I2;
            V_write[2*64] = -I1 + I2;
            V_write[3*64] = I1 - I3;

            V_write[0*64 + 32] = I2 - I4;
            V_write[1*64 + 32] = I3 + I4;
            V_write[2*64 + 32] = -I3 + I4;
            V_write[3*64 + 32] = I3 - I5;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        __local const UNIT_TYPE_4 *V_read_c8 = V_read;
		uint RndCntr = 0;

        __attribute__((opencl_unroll_hint(1)))
        for (uint c8 = 0; c8 < 2; ++c8) {

            // 2*14 * 3 * 8 = 672 MADs

            // Fetch 8 channels of Winograd input components, spread across subgroup.
            // row 0

            __attribute__((opencl_unroll_hint(2)))
            for (uint c5 = 0; c5 < 2; ++c5) {
			    const uint c4 = c5; +8*upperHalf;
                // 2*14 * 3 * 4 = 336 MADs

                // Fetch 4 channels of Winograd filter components.
                //uint2 coordU = coordU0;
				//uint coordU_x = coordU0.x + get_sub_group_local_id()%8;
				const uint flatA = coordU0.y*FILTER_OFM_NUM*KCOLSW*KROWSW + coordU0.x + get_sub_group_local_id()%8;
                UNIT_TYPE_4 f0 = (UNIT_TYPE_4)(
				*(__global UNIT_TYPE *)(&U[flatA+0*FILTER_OFM_NUM*KCOLSW*KROWSW]), // as_UNIT_TYPE_4(intel_sub_group_block_read4(U, coordU));
				*(__global UNIT_TYPE *)(&U[flatA+1*FILTER_OFM_NUM*KCOLSW*KROWSW]), // as_UNIT_TYPE_4(intel_sub_group_block_read4(U, coordU));
				*(__global UNIT_TYPE *)(&U[flatA+2*FILTER_OFM_NUM*KCOLSW*KROWSW]), // as_UNIT_TYPE_4(intel_sub_group_block_read4(U, coordU));
				*(__global UNIT_TYPE *)(&U[flatA+3*FILTER_OFM_NUM*KCOLSW*KROWSW])); // as_UNIT_TYPE_4(intel_sub_group_block_read4(U, coordU));

				//flatA += 8;
                UNIT_TYPE_4 f1 = (UNIT_TYPE_4)(
				*(__global UNIT_TYPE *)(&U[flatA+8+0*FILTER_OFM_NUM*KCOLSW*KROWSW]),
				*(__global UNIT_TYPE *)(&U[flatA+8+1*FILTER_OFM_NUM*KCOLSW*KROWSW]),
				*(__global UNIT_TYPE *)(&U[flatA+8+2*FILTER_OFM_NUM*KCOLSW*KROWSW]),
				*(__global UNIT_TYPE *)(&U[flatA+8+3*FILTER_OFM_NUM*KCOLSW*KROWSW]));

				//flatA += 8;
                UNIT_TYPE_4 f2 = (UNIT_TYPE_4)(
				*(__global UNIT_TYPE *)(&U[flatA+16+0*FILTER_OFM_NUM*KCOLSW*KROWSW]),
				*(__global UNIT_TYPE *)(&U[flatA+16+1*FILTER_OFM_NUM*KCOLSW*KROWSW]),
				*(__global UNIT_TYPE *)(&U[flatA+16+2*FILTER_OFM_NUM*KCOLSW*KROWSW]),
				*(__global UNIT_TYPE *)(&U[flatA+16+3*FILTER_OFM_NUM*KCOLSW*KROWSW]));
                coordU0.y += 4;

				// row 0
				UNIT_TYPE_4 V00 = V_read_c8[0*8 + 32*0 + RndCntr*256];
				DOT4i(M0.s0, f0, V00, 0 + c4);
                DOT4i(M0.s1, f0, V00, 2 + c4);
				DOT4i(M0.s0, f1, V00, 2 + c4);
                DOT4i(M1.s0, f0, V00, 4 + c4);
                DOT4i(M0.s1, f1, V00, 4 + c4);
				DOT4i(M0.s0, f2, V00, 4 + c4);
                DOT4i(M1.s1, f0, V00, 6 + c4);
                DOT4i(M1.s0, f1, V00, 6 + c4);
                DOT4i(M0.s1, f2, V00, 6 + c4);

                UNIT_TYPE_4 V01 = V_read_c8[1*8 + 32*0 + RndCntr*256];
			    DOT4i(M2.s0, f0, V01, 0 + c4);
                DOT4i(M1.s0, f2, V01, 0 + c4);
				DOT4i(M1.s1, f1, V01, 0 + c4);
                DOT4i(M2.s0, f1, V01, 2 + c4);
                DOT4i(M2.s1, f0, V01, 2 + c4);
                DOT4i(M1.s1, f2, V01, 2 + c4);
                DOT4i(M2.s0, f2, V01, 4 + c4);
                DOT4i(M2.s1, f1, V01, 4 + c4);
                DOT4i(M3.s0, f0, V01, 4 + c4);
                DOT4i(M2.s1, f2, V01, 6 + c4);
                DOT4i(M3.s0, f1, V01, 6 + c4);
                DOT4i(M3.s1, f0, V01, 6 + c4);

                UNIT_TYPE_4 V02 = V_read_c8[2*8 + 32*0 + RndCntr*256];
                DOT4i(M4.s0, f0, V02, 0 + c4);
				DOT4i(M3.s0, f2, V02, 0 + c4);
				DOT4i(M3.s1, f1, V02, 0 + c4);
                DOT4i(M4.s0, f1, V02, 2 + c4);
                DOT4i(M4.s1, f0, V02, 2 + c4);
                DOT4i(M3.s1, f2, V02, 2 + c4);
                DOT4i(M4.s0, f2, V02, 4 + c4);
                DOT4i(M4.s1, f1, V02, 4 + c4);
                DOT4i(M5.s0, f0, V02, 4 + c4);
                DOT4i(M4.s1, f2, V02, 6 + c4);
                DOT4i(M5.s0, f1, V02, 6 + c4);
                DOT4i(M5.s1, f0, V02, 6 + c4);

				UNIT_TYPE_4 V03 = V_read_c8[3*8 + 32*0 + RndCntr*256];
				DOT4i(M6.s0, f0, V03, 0 + c4);
                DOT4i(M5.s1, f1, V03, 0 + c4);
				DOT4i(M5.s0, f2, V03, 0 + c4);
                DOT4i(M6.s0, f1, V03, 2 + c4);
                DOT4i(M6.s1, f0, V03, 2 + c4);
                DOT4i(M5.s1, f2, V03, 2 + c4);
                DOT4i(M6.s0, f2, V03, 4 + c4);
                DOT4i(M6.s1, f1, V03, 4 + c4);
                DOT4i(M6.s1, f2, V03, 6 + c4);

                // row 1
				UNIT_TYPE_4 V10 = V_read_c8[0*8 + 32*1 + RndCntr*256];
                DOT4i(M0.s2, f0, V10, 0 + c4);
                DOT4i(M0.s3, f0, V10, 2 + c4);
				DOT4i(M0.s2, f1, V10, 2 + c4);
                DOT4i(M0.s3, f1, V10, 4 + c4);
                DOT4i(M1.s2, f0, V10, 4 + c4);
				DOT4i(M0.s2, f2, V10, 4 + c4);
                DOT4i(M1.s2, f1, V10, 6 + c4);
                DOT4i(M1.s3, f0, V10, 6 + c4);
                DOT4i(M0.s3, f2, V10, 6 + c4);

				UNIT_TYPE_4 V11 = V_read_c8[1*8 + 32*1 + RndCntr*256];
				DOT4i(M2.s2, f0, V11, 0 + c4);
				DOT4i(M1.s3, f1, V11, 0 + c4);
				DOT4i(M1.s2, f2, V11, 0 + c4);
                DOT4i(M2.s2, f1, V11, 2 + c4);
                DOT4i(M2.s3, f0, V11, 2 + c4);
                DOT4i(M1.s3, f2, V11, 2 + c4);
                DOT4i(M3.s2, f0, V11, 4 + c4);
                DOT4i(M2.s3, f1, V11, 4 + c4);
                DOT4i(M2.s2, f2, V11, 4 + c4);
                DOT4i(M3.s2, f1, V11, 6 + c4);
                DOT4i(M3.s3, f0, V11, 6 + c4);
                DOT4i(M2.s3, f2, V11, 6 + c4);

				UNIT_TYPE_4 V12 = V_read_c8[2*8 + 32*1 + RndCntr*256];
                DOT4i(M4.s2, f0, V12, 0 + c4);
				DOT4i(M3.s2, f2, V12, 0 + c4);
				DOT4i(M3.s3, f1, V12, 0 + c4);
                DOT4i(M4.s3, f0, V12, 2 + c4);
                DOT4i(M4.s2, f1, V12, 2 + c4);
                DOT4i(M3.s3, f2, V12, 2 + c4);
                DOT4i(M5.s2, f0, V12, 4 + c4);
                DOT4i(M4.s3, f1, V12, 4 + c4);
                DOT4i(M4.s2, f2, V12, 4 + c4);
                DOT4i(M5.s3, f0, V12, 6 + c4);
                DOT4i(M5.s2, f1, V12, 6 + c4);
                DOT4i(M4.s3, f2, V12, 6 + c4);

				UNIT_TYPE_4 V13 = V_read_c8[3*8 + 32*1 + RndCntr*256];
                DOT4i(M6.s2, f0, V13, 0 + c4);
                DOT4i(M5.s2, f2, V13, 0 + c4);
                DOT4i(M5.s3, f1, V13, 0 + c4);
                DOT4i(M6.s3, f0, V13, 2 + c4);
                DOT4i(M6.s2, f1, V13, 2 + c4);
                DOT4i(M5.s3, f2, V13, 2 + c4);
                DOT4i(M6.s3, f1, V13, 4 + c4);
                DOT4i(M6.s2, f2, V13, 4 + c4);
                DOT4i(M6.s3, f2, V13, 6 + c4);
            }
			RndCntr++;
			//V_read_c8 += 256;

        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store multiplies in SLM.
    {
        __local UNIT_TYPE_4 *M_write = &V[lz*7*8 + lx + slmSize*upperHalf];

        M_write[0*8] = M0;
        M_write[1*8] = M1;
        M_write[2*8] = M2;
        M_write[3*8] = M3;
        M_write[4*8] = M4;
        M_write[5*8] = M5;
        M_write[6*8] = M6;

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lz < 7) 
	{
        // Load multiplies from SLM.
        __local const UNIT_TYPE_8 *M_read = (__local UNIT_TYPE_8*)&V[lz*8 + lxd4*224 + lxm4*2 + slmSize*upperHalf];
        
        UNIT_TYPE_8 M0 = M_read[0*28];
        UNIT_TYPE_8 M1 = M_read[1*28];
        UNIT_TYPE_8 M2 = M_read[2*28];
        UNIT_TYPE_8 M3 = M_read[3*28];

        // Inverse Transform.
        UNIT_TYPE_8 S0 = M0 + M1 + M2;
        UNIT_TYPE_8 S1 = M1 - M2 - M3;

        // Store output to global memory.
        uint p = gy*4 + OUTPUT_PAD_BEFORE_SIZE_Y;
        uint q = gx*14 + lz*2 + OUTPUT_PAD_BEFORE_SIZE_X;
        uint k = gk*16 + lx*2;

		// bias and activation
		#if BIAS_TERM
		#if BIAS_PER_OUTPUT
            const unsigned bias_index0 = k*OUTPUT_SIZE_X*OUTPUT_SIZE_Y + trow*OUTPUT_SIZE_X + q;
			const unsigned bias_index1 = bias_index0 + 1;
		#else
            const unsigned bias_index0 = k;
			const unsigned bias_index1 = bias_index0 + 1;
		#endif
		#endif

#if OUTPUT_LAYOUT_BYXF
		uint outindex = gn*PQK + p*Q*FILTER_OFM_NUM + q*FILTER_OFM_NUM + k;
        __global UNIT_TYPE_2 *O_write = (__global UNIT_TYPE_2 *)&O[outindex];
#else
        __global UNIT_TYPE *O_write_0 = (__global UNIT_TYPE *)(&O[gn*PQK + k*Q*P + (p+0)*Q + q]);
        __global UNIT_TYPE *O_write_1 = (__global UNIT_TYPE *)(&O[gn*PQK + k*Q*P + (p+1)*Q + q]);
        __global UNIT_TYPE *O_write_2 = (__global UNIT_TYPE *)(&O[gn*PQK + k*Q*P + (p+2)*Q + q]);
        __global UNIT_TYPE *O_write_3 = (__global UNIT_TYPE *)(&O[gn*PQK + k*Q*P + (p+3)*Q + q]);
#endif

        // TODO: clip output by P, Q
        bool q0_in = q < Q - OUTPUT_PAD_AFTER_SIZE_X;
        bool q1_in = q + 1 < Q - OUTPUT_PAD_AFTER_SIZE_X;

		if (k < FILTER_OFM_NUM) {
            if (p < P - OUTPUT_PAD_AFTER_SIZE_Y) {
                if (q0_in) {

#if OUTPUT_LAYOUT_BYXF
#if BIAS_TERM
                    O_write[0*QK2 + 0*K2] = (UNIT_TYPE_2)(ACTIVATION(S0.s0 + bias[bias_index0], NL_M, NL_N), ACTIVATION(S0.s4 + bias[bias_index1], NL_M, NL_N));
#else
                    O_write[0*QK2 + 0*K2] = (UNIT_TYPE_2)(ACTIVATION(S0.s0, NL_M, NL_N), ACTIVATION(S0.s4, NL_M, NL_N));
#endif
#else
#if BIAS_TERM
                    O_write_0[0] = ACTIVATION(S0.s0 + bias[bias_index0], NL_M, NL_N);
                    O_write_0[0+Q*P] = ACTIVATION(S0.s4 + bias[bias_index1], NL_M, NL_N);
#else
                    O_write_0[0] = ACTIVATION(S0.s0, NL_M, NL_N);
                    O_write_0[0+Q*P] = ACTIVATION(S0.s4, NL_M, NL_N);
#endif
#endif 
                }
                if (q1_in) {
#if OUTPUT_LAYOUT_BYXF
#if BIAS_TERM
                    O_write[0*QK2 + 1*K2] = (UNIT_TYPE_2)(ACTIVATION(S0.s1 + bias[bias_index0], NL_M, NL_N), ACTIVATION(S0.s5 + bias[bias_index1], NL_M, NL_N));
#else
                    O_write[0*QK2 + 1*K2] = (UNIT_TYPE_2)(ACTIVATION(S0.s1, NL_M, NL_N), ACTIVATION(S0.s5, NL_M, NL_N));
#endif 
#else
#if BIAS_TERM
                    O_write_0[1] = ACTIVATION(S0.s1 + bias[bias_index0], NL_M, NL_N);
                    O_write_0[1+Q*P] = ACTIVATION(S0.s5 + bias[bias_index1], NL_M, NL_N);
#else
                    O_write_0[1] = ACTIVATION(S0.s1, NL_M, NL_N);
                    O_write_0[1+Q*P] = ACTIVATION(S0.s5, NL_M, NL_N);
#endif 
#endif 
                }
            }

            // row 1
            if (p + 1 < P - OUTPUT_PAD_AFTER_SIZE_Y) {
                if (q0_in) {
#if OUTPUT_LAYOUT_BYXF
#if BIAS_TERM
                    O_write[1*QK2 + 0*K2] = (UNIT_TYPE_2)(ACTIVATION(S1.s0 + bias[bias_index0], NL_M, NL_N), ACTIVATION(S1.s4 + bias[bias_index1], NL_M, NL_N));
#else
                    O_write[1*QK2 + 0*K2] = (UNIT_TYPE_2)(ACTIVATION(S1.s0, NL_M, NL_N), ACTIVATION(S1.s4, NL_M, NL_N));
#endif 
#else
#if BIAS_TERM
                    O_write_1[0] = ACTIVATION(S1.s0 + bias[bias_index0], NL_M, NL_N);
                    O_write_1[0+Q*P] = ACTIVATION(S1.s4 + bias[bias_index1], NL_M, NL_N);
#else
                    O_write_1[0] = ACTIVATION(S1.s0, NL_M, NL_N);
                    O_write_1[0+Q*P] = ACTIVATION(S1.s4, NL_M, NL_N);
#endif 
#endif 
                }
                if (q1_in) {
#if OUTPUT_LAYOUT_BYXF
#if BIAS_TERM
                    O_write[1*QK2 + 1*K2] = (UNIT_TYPE_2)(ACTIVATION(S1.s1 + bias[bias_index0], NL_M, NL_N), ACTIVATION(S1.s5 + bias[bias_index1], NL_M, NL_N));
#else
                    O_write[1*QK2 + 1*K2] = (UNIT_TYPE_2)(ACTIVATION(S1.s1, NL_M, NL_N), ACTIVATION(S1.s5, NL_M, NL_N));
#endif 
#else
#if BIAS_TERM
                    O_write_1[1] = ACTIVATION(S1.s1 + bias[bias_index0], NL_M, NL_N);
                    O_write_1[1+Q*P] = ACTIVATION(S1.s5 + bias[bias_index1], NL_M, NL_N);
#else
                    O_write_1[1] = ACTIVATION(S1.s1, NL_M, NL_N);
                    O_write_1[1+Q*P] = ACTIVATION(S1.s5, NL_M, NL_N);
#endif 
#endif 
                }
            }

            // row 2
            if (p + 2 < P - OUTPUT_PAD_AFTER_SIZE_Y) {
                if (q0_in) {
#if OUTPUT_LAYOUT_BYXF
#if BIAS_TERM
                    O_write[2*QK2 + 0*K2] = (UNIT_TYPE_2)(ACTIVATION(S0.s2 + bias[bias_index0], NL_M, NL_N), ACTIVATION(S0.s6 + bias[bias_index1], NL_M, NL_N));
#else
                    O_write[2*QK2 + 0*K2] = (UNIT_TYPE_2)(ACTIVATION(S0.s2, NL_M, NL_N), ACTIVATION(S0.s6, NL_M, NL_N));
#endif 
#else
#if BIAS_TERM
                    O_write_2[0] = ACTIVATION(S0.s2 + bias[bias_index0], NL_M, NL_N);
                    O_write_2[0+Q*P] = ACTIVATION(S0.s6 + bias[bias_index1], NL_M, NL_N);
#else
                    O_write_2[0] = ACTIVATION(S0.s2, NL_M, NL_N);
                    O_write_2[0+Q*P] = ACTIVATION(S0.s6, NL_M, NL_N);
#endif 
#endif 
                }
                if (q1_in) {
#if OUTPUT_LAYOUT_BYXF
#if BIAS_TERM
                    O_write[2*QK2 + 1*K2] = (UNIT_TYPE_2)(ACTIVATION(S0.s3 + bias[bias_index0], NL_M, NL_N), ACTIVATION(S0.s7 + bias[bias_index1], NL_M, NL_N));
#else
                    O_write[2*QK2 + 1*K2] = (UNIT_TYPE_2)(ACTIVATION(S0.s3, NL_M, NL_N), ACTIVATION(S0.s7, NL_M, NL_N));
#endif 
#else
#if BIAS_TERM
                    O_write_2[1] = ACTIVATION(S0.s3 + bias[bias_index0], NL_M, NL_N);
                    O_write_2[1+Q*P] = ACTIVATION(S0.s7 + bias[bias_index1], NL_M, NL_N);
#else
                    O_write_2[1] = ACTIVATION(S0.s3, NL_M, NL_N);
                    O_write_2[1+Q*P] = ACTIVATION(S0.s7, NL_M, NL_N);
#endif 
#endif 
                }
            }

            // row 3
            if (p + 3 < P - OUTPUT_PAD_AFTER_SIZE_Y) {
                if (q0_in) {
#if OUTPUT_LAYOUT_BYXF
#if BIAS_TERM
                    O_write[3*QK2 + 0*K2] = (UNIT_TYPE_2)(ACTIVATION(S1.s2 + bias[bias_index0], NL_M, NL_N), ACTIVATION(S1.s6 + bias[bias_index1], NL_M, NL_N));
#else
                    O_write[3*QK2 + 0*K2] = (UNIT_TYPE_2)(ACTIVATION(S1.s2, NL_M, NL_N), ACTIVATION(S1.s6, NL_M, NL_N));
#endif 
#else
#if BIAS_TERM
                    O_write_3[0] = ACTIVATION(S1.s2 + bias[bias_index0], NL_M, NL_N);
                    O_write_3[0+Q*P] = ACTIVATION(S1.s6 + bias[bias_index1], NL_M, NL_N);
#else
                    O_write_3[0] = ACTIVATION(S1.s2, NL_M, NL_N);
                    O_write_3[0+Q*P] = ACTIVATION(S1.s6, NL_M, NL_N);
#endif 
#endif
                }
                if (q1_in) {
#if OUTPUT_LAYOUT_BYXF
#if BIAS_TERM
                    O_write[3*QK2 + 1*K2] = (UNIT_TYPE_2)(ACTIVATION(S1.s3 + bias[bias_index0], NL_M, NL_N), ACTIVATION(S1.s7 + bias[bias_index1], NL_M, NL_N));
#else
                    O_write[3*QK2 + 1*K2] = (UNIT_TYPE_2)(ACTIVATION(S1.s3, NL_M, NL_N), ACTIVATION(S1.s7, NL_M, NL_N));
#endif 
#else
#if BIAS_TERM
                    O_write_3[1] = ACTIVATION(S1.s3 + bias[bias_index0], NL_M, NL_N);       
                    O_write_3[1+Q*P] = ACTIVATION(S1.s7 + bias[bias_index1], NL_M, NL_N);
#else
                    O_write_3[1] = ACTIVATION(S1.s3, NL_M, NL_N);       
                    O_write_3[1+Q*P] = ACTIVATION(S1.s7, NL_M, NL_N);
#endif   
#endif   
                }
            }
        }


    }
}
#undef UNIT_TYPE_2
#undef UNIT_TYPE_4
#undef UNIT_TYPE_8