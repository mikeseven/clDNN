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

#define DOT8i_0( _result, _A, _B, i)					\
    {									\
	_result = mad(_A.s0, sub_group_broadcast( _B.s0, i), _result);	\
    }
#define DOT8i_1( _result, _A, _B, i)					\
    {									\
	_result = mad(_A.s1, sub_group_broadcast( _B.s1, i), _result);	\
    }
#define DOT8i_2( _result, _A, _B, i)					\
    {									\
	_result = mad(_A.s2, sub_group_broadcast( _B.s2, i), _result);	\
    }
#define DOT8i_3( _result, _A, _B, i)					\
    {									\
	_result = mad(_A.s3, sub_group_broadcast( _B.s3, i), _result);	\
    }
#define DOT8i_4( _result, _A, _B, i)					\
    {									\
	_result = mad(_A.s4, sub_group_broadcast( _B.s4, i), _result);	\
    }
#define DOT8i_5( _result, _A, _B, i)					\
    {									\
	_result = mad(_A.s5, sub_group_broadcast( _B.s5, i), _result);	\
    }
#define DOT8i_6( _result, _A, _B, i)					\
    {									\
	_result = mad(_A.s6, sub_group_broadcast( _B.s6, i), _result);	\
    }
#define DOT8i_7( _result, _A, _B, i)					\
    {									\
	_result = mad(_A.s7, sub_group_broadcast( _B.s7, i), _result);	\
    }

#define DOT8i_( _result, _A, _B, i)					\
    {									\
	_result = mad(_A.s0, sub_group_broadcast( _B.s0, i), _result);	\
	_result = mad(_A.s1, sub_group_broadcast( _B.s1, i), _result);	\
	_result = mad(_A.s2, sub_group_broadcast( _B.s2, i), _result);	\
	_result = mad(_A.s3, sub_group_broadcast( _B.s3, i), _result);	\
	_result = mad(_A.s4, sub_group_broadcast( _B.s4, i), _result);	\
	_result = mad(_A.s5, sub_group_broadcast( _B.s5, i), _result);	\
	_result = mad(_A.s6, sub_group_broadcast( _B.s6, i), _result);	\
	_result = mad(_A.s7, sub_group_broadcast( _B.s7, i), _result);	\
    }

#define UNIT_TYPE_2 CAT(UNIT_TYPE, 2)
#define UNIT_TYPE_4 CAT(UNIT_TYPE, 4)
#define UNIT_TYPE_8 CAT(UNIT_TYPE, 8)

__attribute__((reqd_work_group_size(16, 1, 8)))
__attribute__((intel_reqd_sub_group_size(16)))
KERNEL(convolution_gpu_winograd_6x3_s1_fused)
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
	const uint slmSize = (2 * 8)*(16 * 4);
	__local UNIT_TYPE_4 V[slmSize]; // 8 KB

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

	uint gx = get_group_id(0);
	uint gy = get_group_id(1);
	uint gz = get_group_id(2);	
	uint gk = gz % K16;
	uint gn = gz / K16;

#define lx get_local_id(0)
#define lz get_local_id(2)

	uint lxd8 = lx >> 3;
	uint lxm8 = lx % 8;

	// Load 16x8 input tile, with 2 pixel overlap in X and y.
	// Compute 14x6 output tile.
	// Load 32 filters.
	// 8 threads total
	int x = gx * 14 + lz * 2 + lxd8 - px;
	int y = gy * 6 - py;
	uint k = gk * 16;
	uint c0 = lxm8 * 4;

	// #                                  x->
	// #     M0    M1    M2    M3    M4    M5    M6
	// #   +------------------------------------------
	// # u | 0 1 | 0 1 | 0 1 | 0 1 | 0 1 | 0 1 | 0 1 |
	// # | | 2 3 | 2 3 | 2 3 | 2 3 | 2 3 | 2 3 | 2 3 |
	// # v
	// #

	UNIT_TYPE_2 M0 = (UNIT_TYPE_2)(UNIT_VAL_ZERO, UNIT_VAL_ZERO);
	UNIT_TYPE_2 M1 = (UNIT_TYPE_2)(UNIT_VAL_ZERO, UNIT_VAL_ZERO);
	UNIT_TYPE_2 M2 = (UNIT_TYPE_2)(UNIT_VAL_ZERO, UNIT_VAL_ZERO);
	UNIT_TYPE_2 M3 = (UNIT_TYPE_2)(UNIT_VAL_ZERO, UNIT_VAL_ZERO);
	UNIT_TYPE_2 M4 = (UNIT_TYPE_2)(UNIT_VAL_ZERO, UNIT_VAL_ZERO);
	UNIT_TYPE_2 M5 = (UNIT_TYPE_2)(UNIT_VAL_ZERO, UNIT_VAL_ZERO);
	UNIT_TYPE_2 M6 = (UNIT_TYPE_2)(UNIT_VAL_ZERO, UNIT_VAL_ZERO);

	/*if (gy == 0) {
		y = 0;
	}*/

	uint lxm4 = lx % 4;
	uint lxb2 = (lx & 4) / 4;

#if INPUT0_LAYOUT_BYXF
	uint adr = gn*HWC + ((uint)y)*WC + ((uint)x)*INPUT0_FEATURE_NUM + c0;
	const __global UNIT_TYPE_4 *I_load = ((const __global UNIT_TYPE_4*)&(I[adr]));
#else
	const __global UNIT_TYPE *I_load = (const __global UNIT_TYPE*)&I[gn*HWC + c0*HW + ((uint)y)*W + ((uint)x)];
#endif

	// c, Kdsk
	uint2 coordU0;
	//coordU0.x = (k * 3);
	//coordU0.y = lz*C_;
	coordU0.x = (lz * 48 + k * 24);
	coordU0.y = 0;

	__attribute__((opencl_unroll_hint(1)))
		for (uint c = 0; c < C4; c += 8) {

			__local UNIT_TYPE_4 *V_write = &V[lxb2 * 512 + lz * 8 + lxd8 * 4 + lxm4];
			__local const UNIT_TYPE_8 *V_read = (__local const UNIT_TYPE_8 *)&V[lz * 64 + lx * 2];

			// 2*14 * 3 * 16 = 1344 MADs

			// Transform HxW x C        -> DxUxW x C
			//           6x16x16 inputs -> 4x2x16x16 winograd components.
			{
				bool x_in = 0 <= x && x < W;
				bool y0_in = 0 <= (y + 0) && (y + 0) < H && x_in;
				bool y1_in = 0 <= (y + 1) && (y + 1) < H && x_in;
				bool y2_in = 0 <= (y + 2) && (y + 2) < H && x_in;
				bool y3_in = 0 <= (y + 3) && (y + 3) < H && x_in;
				bool y4_in = 0 <= (y + 4) && (y + 4) < H && x_in;
				bool y5_in = 0 <= (y + 5) && (y + 5) < H && x_in;
				bool y6_in = 0 <= (y + 6) && (y + 6) < H && x_in;
				bool y7_in = 0 <= (y + 7) && (y + 7) < H && x_in;

#if INPUT0_LAYOUT_BYXF

				/*				const  UNIT_TYPE_4 I_load_0 = y0_in ? I_load[0*WC4+c] : (UNIT_TYPE_4)(UNIT_VAL_ZERO);
								const  UNIT_TYPE_4 I_load_1 = y1_in ? I_load[1*WC4+c] : (UNIT_TYPE_4)(UNIT_VAL_ZERO);
								const  UNIT_TYPE_4 I_load_2 = y2_in ? I_load[2*WC4+c] : (UNIT_TYPE_4)(UNIT_VAL_ZERO);
								const  UNIT_TYPE_4 I_load_3 = y3_in ? I_load[3*WC4+c] : (UNIT_TYPE_4)(UNIT_VAL_ZERO);
								const  UNIT_TYPE_4 I_load_4 = y4_in ? I_load[4*WC4+c] : (UNIT_TYPE_4)(UNIT_VAL_ZERO);
								const  UNIT_TYPE_4 I_load_5 = y5_in ? I_load[5*WC4+c] : (UNIT_TYPE_4)(UNIT_VAL_ZERO);*/

				const  UNIT_TYPE_4 I_load_0 = y0_in ? *((const __global UNIT_TYPE_4*)(I + adr + (0 * WC4 + c) * 4)) : (UNIT_TYPE_4)(UNIT_VAL_ZERO);
				const  UNIT_TYPE_4 I_load_1 = y1_in ? *((const __global UNIT_TYPE_4*)(I + adr + (1 * WC4 + c) * 4)) : (UNIT_TYPE_4)(UNIT_VAL_ZERO);
				const  UNIT_TYPE_4 I_load_2 = y2_in ? *((const __global UNIT_TYPE_4*)(I + adr + (2 * WC4 + c) * 4)) : (UNIT_TYPE_4)(UNIT_VAL_ZERO);
				const  UNIT_TYPE_4 I_load_3 = y3_in ? *((const __global UNIT_TYPE_4*)(I + adr + (3 * WC4 + c) * 4)) : (UNIT_TYPE_4)(UNIT_VAL_ZERO);
				const  UNIT_TYPE_4 I_load_4 = y4_in ? *((const __global UNIT_TYPE_4*)(I + adr + (4 * WC4 + c) * 4)) : (UNIT_TYPE_4)(UNIT_VAL_ZERO);
				const  UNIT_TYPE_4 I_load_5 = y5_in ? *((const __global UNIT_TYPE_4*)(I + adr + (5 * WC4 + c) * 4)) : (UNIT_TYPE_4)(UNIT_VAL_ZERO);
				const  UNIT_TYPE_4 I_load_6 = y6_in ? *((const __global UNIT_TYPE_4*)(I + adr + (6 * WC4 + c) * 4)) : (UNIT_TYPE_4)(UNIT_VAL_ZERO);
				const  UNIT_TYPE_4 I_load_7 = y7_in ? *((const __global UNIT_TYPE_4*)(I + adr + (7 * WC4 + c) * 4)) : (UNIT_TYPE_4)(UNIT_VAL_ZERO);


#else
				const __global UNIT_TYPE *I_load_0 = &I_load[0 * W]; //y0_in ? &I_load[0*W] : zeros4;
				const __global UNIT_TYPE *I_load_1 = &I_load[1 * W]; //y1_in ? &I_load[1*W] : zeros4;
				const __global UNIT_TYPE *I_load_2 = &I_load[2 * W]; //y2_in ? &I_load[2*W] : zeros4;
				const __global UNIT_TYPE *I_load_3 = &I_load[3 * W]; //y3_in ? &I_load[3*W] : zeros4;
				const __global UNIT_TYPE *I_load_4 = &I_load[4 * W]; //y4_in ? &I_load[4*W] : zeros4;
				const __global UNIT_TYPE *I_load_5 = &I_load[5 * W]; //y5_in ? &I_load[5*W] : zeros4;
				const __global UNIT_TYPE *I_load_6 = &I_load[6 * W]; //y5_in ? &I_load[5*W] : zeros4;
				const __global UNIT_TYPE *I_load_7 = &I_load[7 * W]; //y5_in ? &I_load[5*W] : zeros4;
#endif

#if INPUT0_LAYOUT_BYXF

				UNIT_TYPE_4 I0 = I_load_0;
				UNIT_TYPE_4 I1 = I_load_1;
				UNIT_TYPE_4 I2 = I_load_2;
				UNIT_TYPE_4 I3 = I_load_3;
				UNIT_TYPE_4 I4 = I_load_4;
				UNIT_TYPE_4 I5 = I_load_5;
				UNIT_TYPE_4 I6 = I_load_6;
				UNIT_TYPE_4 I7 = I_load_7;

#else
				UNIT_TYPE_4 I0 = y0_in ? (UNIT_TYPE_4)(I_load_0[c*HW * 4], I_load_0[c*HW * 4 + HW], I_load_0[c*HW * 4 + HW * 2], I_load_0[c*HW * 4 + HW * 3]) : (UNIT_TYPE_4)(UNIT_VAL_ZERO, UNIT_VAL_ZERO, UNIT_VAL_ZERO, UNIT_VAL_ZERO);
				UNIT_TYPE_4 I1 = y1_in ? (UNIT_TYPE_4)(I_load_1[c*HW * 4], I_load_1[c*HW * 4 + HW], I_load_1[c*HW * 4 + HW * 2], I_load_1[c*HW * 4 + HW * 3]) : (UNIT_TYPE_4)(UNIT_VAL_ZERO, UNIT_VAL_ZERO, UNIT_VAL_ZERO, UNIT_VAL_ZERO);
				UNIT_TYPE_4 I2 = y2_in ? (UNIT_TYPE_4)(I_load_2[c*HW * 4], I_load_2[c*HW * 4 + HW], I_load_2[c*HW * 4 + HW * 2], I_load_2[c*HW * 4 + HW * 3]) : (UNIT_TYPE_4)(UNIT_VAL_ZERO, UNIT_VAL_ZERO, UNIT_VAL_ZERO, UNIT_VAL_ZERO);
				UNIT_TYPE_4 I3 = y3_in ? (UNIT_TYPE_4)(I_load_3[c*HW * 4], I_load_3[c*HW * 4 + HW], I_load_3[c*HW * 4 + HW * 2], I_load_3[c*HW * 4 + HW * 3]) : (UNIT_TYPE_4)(UNIT_VAL_ZERO, UNIT_VAL_ZERO, UNIT_VAL_ZERO, UNIT_VAL_ZERO);
				UNIT_TYPE_4 I4 = y4_in ? (UNIT_TYPE_4)(I_load_4[c*HW * 4], I_load_4[c*HW * 4 + HW], I_load_4[c*HW * 4 + HW * 2], I_load_4[c*HW * 4 + HW * 3]) : (UNIT_TYPE_4)(UNIT_VAL_ZERO, UNIT_VAL_ZERO, UNIT_VAL_ZERO, UNIT_VAL_ZERO);
				UNIT_TYPE_4 I5 = y5_in ? (UNIT_TYPE_4)(I_load_5[c*HW * 4], I_load_5[c*HW * 4 + HW], I_load_5[c*HW * 4 + HW * 2], I_load_5[c*HW * 4 + HW * 3]) : (UNIT_TYPE_4)(UNIT_VAL_ZERO, UNIT_VAL_ZERO, UNIT_VAL_ZERO, UNIT_VAL_ZERO);
				UNIT_TYPE_4 I6 = y6_in ? (UNIT_TYPE_4)(I_load_6[c*HW * 4], I_load_6[c*HW * 4 + HW], I_load_6[c*HW * 4 + HW * 2], I_load_6[c*HW * 4 + HW * 3]) : (UNIT_TYPE_4)(UNIT_VAL_ZERO, UNIT_VAL_ZERO, UNIT_VAL_ZERO, UNIT_VAL_ZERO);
				UNIT_TYPE_4 I7 = y7_in ? (UNIT_TYPE_4)(I_load_7[c*HW * 4], I_load_7[c*HW * 4 + HW], I_load_7[c*HW * 4 + HW * 2], I_load_7[c*HW * 4 + HW * 3]) : (UNIT_TYPE_4)(UNIT_VAL_ZERO, UNIT_VAL_ZERO, UNIT_VAL_ZERO, UNIT_VAL_ZERO);
#endif
				//I_load += 8;

				// Compute Winograd f6x3 data transform and store components in SLM.
				V_write[0 * 64] = I0 - 5.25h*I2 + 5.25h*I4 - I6;

				UNIT_TYPE_4 x0 = I1 - 4.25h*I3 + I5;
				UNIT_TYPE_4 x1 = I2 - 4.25h*I4 + I6;

				V_write[1 * 64] = x1 + x0;
				V_write[2 * 64] = x1 - x0;

				UNIT_TYPE_4 x2 = -5.h*I3 + I1;
				UNIT_TYPE_4 x3 = 4.h*I5 + x2;
				UNIT_TYPE_4 x4 = 0.25h*I2 + I6;
				UNIT_TYPE_4 x5 = -1.25h*I4 + x4;

				V_write[3 * 64] = +0.5h * x3 + x5;
				V_write[4 * 64] = -0.5h * x3 + x5;

				UNIT_TYPE_4 x6 = 4.h*I1 + I5;
				UNIT_TYPE_4 x7 = -5.h*I3 + x6;
				UNIT_TYPE_4 x8 = 4.h*I2 + I6;
				UNIT_TYPE_4 x9 = -5.h*I4 + x8;

				V_write[5 * 64] = +0.5h*x7 + x9;
				V_write[6 * 64] = -0.5h*x7 + x9;

				V_write[7 * 64] = -I1 + 5.25h*I3 - 5.25h*I5 + I7;
			}

			barrier(CLK_LOCAL_MEM_FENCE);

			__local const UNIT_TYPE_8 *V_read_c16 = V_read;

			__attribute__((opencl_unroll_hint(1)))
				for (uint c16 = 0; c16 < 2 ; ++c16) {

					// 2*14 * 3 * 8 = 672 MADs

					// Fetch 16 channels of Winograd input components, spread across subgroup.
					UNIT_TYPE_8 V0 = V_read_c16[0 * 16 + c16 * 256];
					UNIT_TYPE_8 V8 = V_read_c16[1 * 16 + c16 * 256];

					__attribute__((opencl_unroll_hint(2)))
						for (int c8 = 0; c8 < 2; ++c8) {


							// filter 0

							// 2*14 * 3 * 4 = 336 MADs

							const uint2 coordU = coordU0;

							const uint flatA = coordU0.y*FILTER_OFM_NUM*KCOLSW*KROWSW + coordU0.x;

							// Fetch 8 channels of Winograd components from f(k,s)
							const UNIT_TYPE_8 f00 = (UNIT_TYPE_8)(
								as_half(intel_sub_group_block_read_us((__global unsigned short *)&U[flatA + 0 * FILTER_OFM_NUM*KCOLSW*KROWSW])),
								as_half(intel_sub_group_block_read_us((__global unsigned short *)&U[flatA + 1 * FILTER_OFM_NUM*KCOLSW*KROWSW])),
								as_half(intel_sub_group_block_read_us((__global unsigned short *)&U[flatA + 2 * FILTER_OFM_NUM*KCOLSW*KROWSW])),
								as_half(intel_sub_group_block_read_us((__global unsigned short *)&U[flatA + 3 * FILTER_OFM_NUM*KCOLSW*KROWSW])),
								as_half(intel_sub_group_block_read_us((__global unsigned short *)&U[flatA + 4 * FILTER_OFM_NUM*KCOLSW*KROWSW])),
								as_half(intel_sub_group_block_read_us((__global unsigned short *)&U[flatA + 5 * FILTER_OFM_NUM*KCOLSW*KROWSW])),
								as_half(intel_sub_group_block_read_us((__global unsigned short *)&U[flatA + 6 * FILTER_OFM_NUM*KCOLSW*KROWSW])),
								as_half(intel_sub_group_block_read_us((__global unsigned short *)&U[flatA + 7 * FILTER_OFM_NUM*KCOLSW*KROWSW])));


							// f0 x v[0 .. 14]
							DOT8i_0(M0.s0, f00, V0, 0 + c8);
							DOT8i_0(M0.s1, f00, V0, 2 + c8);
							DOT8i_0(M1.s0, f00, V0, 4 + c8);
							DOT8i_0(M1.s1, f00, V0, 6 + c8);

							DOT8i_0(M2.s0, f00, V0, 8 + c8);
							DOT8i_0(M2.s1, f00, V0, 10 + c8);
							DOT8i_0(M3.s0, f00, V0, 12 + c8);
							DOT8i_0(M3.s1, f00, V0, 14 + c8);

							DOT8i_0(M4.s0, f00, V8, 0 + c8);
							DOT8i_0(M4.s1, f00, V8, 2 + c8);
							DOT8i_0(M5.s0, f00, V8, 4 + c8);
							DOT8i_0(M5.s1, f00, V8, 6 + c8);

							DOT8i_0(M6.s0, f00, V8, 8 + c8);
							DOT8i_0(M6.s1, f00, V8, 10 + c8);

							// f0 x v[0 .. 14]
							DOT8i_1(M0.s0, f00, V0, 0 + c8);
							DOT8i_1(M0.s1, f00, V0, 2 + c8);
							DOT8i_1(M1.s0, f00, V0, 4 + c8);
							DOT8i_1(M1.s1, f00, V0, 6 + c8);

							DOT8i_1(M2.s0, f00, V0, 8 + c8);
							DOT8i_1(M2.s1, f00, V0, 10 + c8);
							DOT8i_1(M3.s0, f00, V0, 12 + c8);
							DOT8i_1(M3.s1, f00, V0, 14 + c8);

							DOT8i_1(M4.s0, f00, V8, 0 + c8);
							DOT8i_1(M4.s1, f00, V8, 2 + c8);
							DOT8i_1(M5.s0, f00, V8, 4 + c8);
							DOT8i_1(M5.s1, f00, V8, 6 + c8);

							DOT8i_1(M6.s0, f00, V8, 8 + c8);
							DOT8i_1(M6.s1, f00, V8, 10 + c8);

							// f0 x v[0 .. 14]
							DOT8i_2(M0.s0, f00, V0, 0 + c8);
							DOT8i_2(M0.s1, f00, V0, 2 + c8);
							DOT8i_2(M1.s0, f00, V0, 4 + c8);
							DOT8i_2(M1.s1, f00, V0, 6 + c8);

							DOT8i_2(M2.s0, f00, V0, 8 + c8);
							DOT8i_2(M2.s1, f00, V0, 10 + c8);
							DOT8i_2(M3.s0, f00, V0, 12 + c8);
							DOT8i_2(M3.s1, f00, V0, 14 + c8);

							DOT8i_2(M4.s0, f00, V8, 0 + c8);
							DOT8i_2(M4.s1, f00, V8, 2 + c8);
							DOT8i_2(M5.s0, f00, V8, 4 + c8);
							DOT8i_2(M5.s1, f00, V8, 6 + c8);

							DOT8i_2(M6.s0, f00, V8, 8 + c8);
							DOT8i_2(M6.s1, f00, V8, 10 + c8);


							// f0 x v[0 .. 14]
							DOT8i_3(M0.s0, f00, V0, 0 + c8);
							DOT8i_3(M0.s1, f00, V0, 2 + c8);
							DOT8i_3(M1.s0, f00, V0, 4 + c8);
							DOT8i_3(M1.s1, f00, V0, 6 + c8);

							DOT8i_3(M2.s0, f00, V0, 8 + c8);
							DOT8i_3(M2.s1, f00, V0, 10 + c8);
							DOT8i_3(M3.s0, f00, V0, 12 + c8);
							DOT8i_3(M3.s1, f00, V0, 14 + c8);

							DOT8i_3(M4.s0, f00, V8, 0 + c8);
							DOT8i_3(M4.s1, f00, V8, 2 + c8);
							DOT8i_3(M5.s0, f00, V8, 4 + c8);
							DOT8i_3(M5.s1, f00, V8, 6 + c8);

							DOT8i_3(M6.s0, f00, V8, 8 + c8);
							DOT8i_3(M6.s1, f00, V8, 10 + c8);


							// f0 x v[0 .. 14]
							DOT8i_4(M0.s0, f00, V0, 0 + c8);
							DOT8i_4(M0.s1, f00, V0, 2 + c8);
							DOT8i_4(M1.s0, f00, V0, 4 + c8);
							DOT8i_4(M1.s1, f00, V0, 6 + c8);

							DOT8i_4(M2.s0, f00, V0, 8 + c8);
							DOT8i_4(M2.s1, f00, V0, 10 + c8);
							DOT8i_4(M3.s0, f00, V0, 12 + c8);
							DOT8i_4(M3.s1, f00, V0, 14 + c8);

							DOT8i_4(M4.s0, f00, V8, 0 + c8);
							DOT8i_4(M4.s1, f00, V8, 2 + c8);
							DOT8i_4(M5.s0, f00, V8, 4 + c8);
							DOT8i_4(M5.s1, f00, V8, 6 + c8);

							DOT8i_4(M6.s0, f00, V8, 8 + c8);
							DOT8i_4(M6.s1, f00, V8, 10 + c8);

							// f0 x v[0 .. 14]
							DOT8i_5(M0.s0, f00, V0, 0 + c8);
							DOT8i_5(M0.s1, f00, V0, 2 + c8);
							DOT8i_5(M1.s0, f00, V0, 4 + c8);
							DOT8i_5(M1.s1, f00, V0, 6 + c8);

							DOT8i_5(M2.s0, f00, V0, 8 + c8);
							DOT8i_5(M2.s1, f00, V0, 10 + c8);
							DOT8i_5(M3.s0, f00, V0, 12 + c8);
							DOT8i_5(M3.s1, f00, V0, 14 + c8);

							DOT8i_5(M4.s0, f00, V8, 0 + c8);
							DOT8i_5(M4.s1, f00, V8, 2 + c8);
							DOT8i_5(M5.s0, f00, V8, 4 + c8);
							DOT8i_5(M5.s1, f00, V8, 6 + c8);

							DOT8i_5(M6.s0, f00, V8, 8 + c8);
							DOT8i_5(M6.s1, f00, V8, 10 + c8);

							// f0 x v[0 .. 14]
							DOT8i_6(M0.s0, f00, V0, 0 + c8);
							DOT8i_6(M0.s1, f00, V0, 2 + c8);
							DOT8i_6(M1.s0, f00, V0, 4 + c8);
							DOT8i_6(M1.s1, f00, V0, 6 + c8);

							DOT8i_6(M2.s0, f00, V0, 8 + c8);
							DOT8i_6(M2.s1, f00, V0, 10 + c8);
							DOT8i_6(M3.s0, f00, V0, 12 + c8);
							DOT8i_6(M3.s1, f00, V0, 14 + c8);

							DOT8i_6(M4.s0, f00, V8, 0 + c8);
							DOT8i_6(M4.s1, f00, V8, 2 + c8);
							DOT8i_6(M5.s0, f00, V8, 4 + c8);
							DOT8i_6(M5.s1, f00, V8, 6 + c8);

							DOT8i_6(M6.s0, f00, V8, 8 + c8);
							DOT8i_6(M6.s1, f00, V8, 10 + c8);


							// f0 x v[0 .. 14]
							DOT8i_7(M0.s0, f00, V0, 0 + c8);
							DOT8i_7(M0.s1, f00, V0, 2 + c8);
							DOT8i_7(M1.s0, f00, V0, 4 + c8);
							DOT8i_7(M1.s1, f00, V0, 6 + c8);

							DOT8i_7(M2.s0, f00, V0, 8 + c8);
							DOT8i_7(M2.s1, f00, V0, 10 + c8);
							DOT8i_7(M3.s0, f00, V0, 12 + c8);
							DOT8i_7(M3.s1, f00, V0, 14 + c8);

							DOT8i_7(M4.s0, f00, V8, 0 + c8);
							DOT8i_7(M4.s1, f00, V8, 2 + c8);
							DOT8i_7(M5.s0, f00, V8, 4 + c8);
							DOT8i_7(M5.s1, f00, V8, 6 + c8);

							DOT8i_7(M6.s0, f00, V8, 8 + c8);
							DOT8i_7(M6.s1, f00, V8, 10 + c8);

							const UNIT_TYPE_8 f01 = (UNIT_TYPE_8)(
								as_half(intel_sub_group_block_read_us((__global unsigned short *)&U[flatA + 16 + 0 * FILTER_OFM_NUM*KCOLSW*KROWSW])),
								as_half(intel_sub_group_block_read_us((__global unsigned short *)&U[flatA + 16 + 1 * FILTER_OFM_NUM*KCOLSW*KROWSW])),
								as_half(intel_sub_group_block_read_us((__global unsigned short *)&U[flatA + 16 + 2 * FILTER_OFM_NUM*KCOLSW*KROWSW])),
								as_half(intel_sub_group_block_read_us((__global unsigned short *)&U[flatA + 16 + 3 * FILTER_OFM_NUM*KCOLSW*KROWSW])),
								as_half(intel_sub_group_block_read_us((__global unsigned short *)&U[flatA + 16 + 4 * FILTER_OFM_NUM*KCOLSW*KROWSW])),
								as_half(intel_sub_group_block_read_us((__global unsigned short *)&U[flatA + 16 + 5 * FILTER_OFM_NUM*KCOLSW*KROWSW])),
								as_half(intel_sub_group_block_read_us((__global unsigned short *)&U[flatA + 16 + 6 * FILTER_OFM_NUM*KCOLSW*KROWSW])),
								as_half(intel_sub_group_block_read_us((__global unsigned short *)&U[flatA + 16 + 7 * FILTER_OFM_NUM*KCOLSW*KROWSW])));


							// f1[c8] x v[1 .. 15]
							DOT8i_0(M0.s0, f01, V0, 2 + c8);
							DOT8i_0(M0.s1, f01, V0, 4 + c8);
							DOT8i_0(M1.s0, f01, V0, 6 + c8);
							DOT8i_0(M1.s1, f01, V0, 8 + c8);

							DOT8i_0(M2.s0, f01, V0, 10 + c8);
							DOT8i_0(M2.s1, f01, V0, 12 + c8);
							DOT8i_0(M3.s0, f01, V0, 14 + c8);
							DOT8i_0(M3.s1, f01, V8, 0 + c8);

							DOT8i_0(M4.s0, f01, V8, 2 + c8);
							DOT8i_0(M4.s1, f01, V8, 4 + c8);
							DOT8i_0(M5.s0, f01, V8, 6 + c8);
							DOT8i_0(M5.s1, f01, V8, 8 + c8);

							DOT8i_0(M6.s0, f01, V8, 10 + c8);
							DOT8i_0(M6.s1, f01, V8, 12 + c8);

							// f1[c8] x v[1 .. 15]
							DOT8i_1(M0.s0, f01, V0, 2 + c8);
							DOT8i_1(M0.s1, f01, V0, 4 + c8);
							DOT8i_1(M1.s0, f01, V0, 6 + c8);
							DOT8i_1(M1.s1, f01, V0, 8 + c8);

							DOT8i_1(M2.s0, f01, V0, 10 + c8);
							DOT8i_1(M2.s1, f01, V0, 12 + c8);
							DOT8i_1(M3.s0, f01, V0, 14 + c8);
							DOT8i_1(M3.s1, f01, V8, 0 + c8);

							DOT8i_1(M4.s0, f01, V8, 2 + c8);
							DOT8i_1(M4.s1, f01, V8, 4 + c8);
							DOT8i_1(M5.s0, f01, V8, 6 + c8);
							DOT8i_1(M5.s1, f01, V8, 8 + c8);

							DOT8i_1(M6.s0, f01, V8, 10 + c8);
							DOT8i_1(M6.s1, f01, V8, 12 + c8);

							// f1[c8] x v[1 .. 15]
							DOT8i_2(M0.s0, f01, V0, 2 + c8);
							DOT8i_2(M0.s1, f01, V0, 4 + c8);
							DOT8i_2(M1.s0, f01, V0, 6 + c8);
							DOT8i_2(M1.s1, f01, V0, 8 + c8);

							DOT8i_2(M2.s0, f01, V0, 10 + c8);
							DOT8i_2(M2.s1, f01, V0, 12 + c8);
							DOT8i_2(M3.s0, f01, V0, 14 + c8);
							DOT8i_2(M3.s1, f01, V8, 0 + c8);

							DOT8i_2(M4.s0, f01, V8, 2 + c8);
							DOT8i_2(M4.s1, f01, V8, 4 + c8);
							DOT8i_2(M5.s0, f01, V8, 6 + c8);
							DOT8i_2(M5.s1, f01, V8, 8 + c8);

							DOT8i_2(M6.s0, f01, V8, 10 + c8);
							DOT8i_2(M6.s1, f01, V8, 12 + c8);

							// f1[c8] x v[1 .. 15]
							DOT8i_3(M0.s0, f01, V0, 2 + c8);
							DOT8i_3(M0.s1, f01, V0, 4 + c8);
							DOT8i_3(M1.s0, f01, V0, 6 + c8);
							DOT8i_3(M1.s1, f01, V0, 8 + c8);

							DOT8i_3(M2.s0, f01, V0, 10 + c8);
							DOT8i_3(M2.s1, f01, V0, 12 + c8);
							DOT8i_3(M3.s0, f01, V0, 14 + c8);
							DOT8i_3(M3.s1, f01, V8, 0 + c8);

							DOT8i_3(M4.s0, f01, V8, 2 + c8);
							DOT8i_3(M4.s1, f01, V8, 4 + c8);
							DOT8i_3(M5.s0, f01, V8, 6 + c8);
							DOT8i_3(M5.s1, f01, V8, 8 + c8);

							DOT8i_3(M6.s0, f01, V8, 10 + c8);
							DOT8i_3(M6.s1, f01, V8, 12 + c8);

							// f1[c8] x v[1 .. 15]
							DOT8i_4(M0.s0, f01, V0, 2 + c8);
							DOT8i_4(M0.s1, f01, V0, 4 + c8);
							DOT8i_4(M1.s0, f01, V0, 6 + c8);
							DOT8i_4(M1.s1, f01, V0, 8 + c8);

							DOT8i_4(M2.s0, f01, V0, 10 + c8);
							DOT8i_4(M2.s1, f01, V0, 12 + c8);
							DOT8i_4(M3.s0, f01, V0, 14 + c8);
							DOT8i_4(M3.s1, f01, V8, 0 + c8);

							DOT8i_4(M4.s0, f01, V8, 2 + c8);
							DOT8i_4(M4.s1, f01, V8, 4 + c8);
							DOT8i_4(M5.s0, f01, V8, 6 + c8);
							DOT8i_4(M5.s1, f01, V8, 8 + c8);

							DOT8i_4(M6.s0, f01, V8, 10 + c8);
							DOT8i_4(M6.s1, f01, V8, 12 + c8);


							// f1[c8] x v[1 .. 15]
							DOT8i_5(M0.s0, f01, V0, 2 + c8);
							DOT8i_5(M0.s1, f01, V0, 4 + c8);
							DOT8i_5(M1.s0, f01, V0, 6 + c8);
							DOT8i_5(M1.s1, f01, V0, 8 + c8);

							DOT8i_5(M2.s0, f01, V0, 10 + c8);
							DOT8i_5(M2.s1, f01, V0, 12 + c8);
							DOT8i_5(M3.s0, f01, V0, 14 + c8);
							DOT8i_5(M3.s1, f01, V8, 0 + c8);

							DOT8i_5(M4.s0, f01, V8, 2 + c8);
							DOT8i_5(M4.s1, f01, V8, 4 + c8);
							DOT8i_5(M5.s0, f01, V8, 6 + c8);
							DOT8i_5(M5.s1, f01, V8, 8 + c8);

							DOT8i_5(M6.s0, f01, V8, 10 + c8);
							DOT8i_5(M6.s1, f01, V8, 12 + c8);


							// f1[c8] x v[1 .. 15]
							DOT8i_6(M0.s0, f01, V0, 2 + c8);
							DOT8i_6(M0.s1, f01, V0, 4 + c8);
							DOT8i_6(M1.s0, f01, V0, 6 + c8);
							DOT8i_6(M1.s1, f01, V0, 8 + c8);

							DOT8i_6(M2.s0, f01, V0, 10 + c8);
							DOT8i_6(M2.s1, f01, V0, 12 + c8);
							DOT8i_6(M3.s0, f01, V0, 14 + c8);
							DOT8i_6(M3.s1, f01, V8, 0 + c8);

							DOT8i_6(M4.s0, f01, V8, 2 + c8);
							DOT8i_6(M4.s1, f01, V8, 4 + c8);
							DOT8i_6(M5.s0, f01, V8, 6 + c8);
							DOT8i_6(M5.s1, f01, V8, 8 + c8);

							DOT8i_6(M6.s0, f01, V8, 10 + c8);
							DOT8i_6(M6.s1, f01, V8, 12 + c8);



							// f1[c8] x v[1 .. 15]
							DOT8i_7(M0.s0, f01, V0, 2 + c8);
							DOT8i_7(M0.s1, f01, V0, 4 + c8);
							DOT8i_7(M1.s0, f01, V0, 6 + c8);
							DOT8i_7(M1.s1, f01, V0, 8 + c8);

							DOT8i_7(M2.s0, f01, V0, 10 + c8);
							DOT8i_7(M2.s1, f01, V0, 12 + c8);
							DOT8i_7(M3.s0, f01, V0, 14 + c8);
							DOT8i_7(M3.s1, f01, V8, 0 + c8);

							DOT8i_7(M4.s0, f01, V8, 2 + c8);
							DOT8i_7(M4.s1, f01, V8, 4 + c8);
							DOT8i_7(M5.s0, f01, V8, 6 + c8);
							DOT8i_7(M5.s1, f01, V8, 8 + c8);

							DOT8i_7(M6.s0, f01, V8, 10 + c8);
							DOT8i_7(M6.s1, f01, V8, 12 + c8);

							const UNIT_TYPE_8 f02 = (UNIT_TYPE_8)(
								as_half(intel_sub_group_block_read_us((__global unsigned short *)&U[flatA + 32 + 0 * FILTER_OFM_NUM*KCOLSW*KROWSW])),
								as_half(intel_sub_group_block_read_us((__global unsigned short *)&U[flatA + 32 + 1 * FILTER_OFM_NUM*KCOLSW*KROWSW])),
								as_half(intel_sub_group_block_read_us((__global unsigned short *)&U[flatA + 32 + 2 * FILTER_OFM_NUM*KCOLSW*KROWSW])),
								as_half(intel_sub_group_block_read_us((__global unsigned short *)&U[flatA + 32 + 3 * FILTER_OFM_NUM*KCOLSW*KROWSW])),
								as_half(intel_sub_group_block_read_us((__global unsigned short *)&U[flatA + 32 + 4 * FILTER_OFM_NUM*KCOLSW*KROWSW])),
								as_half(intel_sub_group_block_read_us((__global unsigned short *)&U[flatA + 32 + 5 * FILTER_OFM_NUM*KCOLSW*KROWSW])),
								as_half(intel_sub_group_block_read_us((__global unsigned short *)&U[flatA + 32 + 6 * FILTER_OFM_NUM*KCOLSW*KROWSW])),
								as_half(intel_sub_group_block_read_us((__global unsigned short *)&U[flatA + 32 + 7 * FILTER_OFM_NUM*KCOLSW*KROWSW])));
							coordU0.y += 8;



							// f2[c8] x v[2 .. 16]
							DOT8i_0(M0.s0, f02, V0, 4 + c8);
							DOT8i_0(M0.s1, f02, V0, 6 + c8);
							DOT8i_0(M1.s0, f02, V0, 8 + c8);
							DOT8i_0(M1.s1, f02, V0, 10 + c8);

							DOT8i_0(M2.s0, f02, V0, 12 + c8);
							DOT8i_0(M2.s1, f02, V0, 14 + c8);
							DOT8i_0(M3.s0, f02, V8, 0 + c8);
							DOT8i_0(M3.s1, f02, V8, 2 + c8);

							DOT8i_0(M4.s0, f02, V8, 4 + c8);
							DOT8i_0(M4.s1, f02, V8, 6 + c8);
							DOT8i_0(M5.s0, f02, V8, 8 + c8);
							DOT8i_0(M5.s1, f02, V8, 10 + c8);

							DOT8i_0(M6.s0, f02, V8, 12 + c8);
							DOT8i_0(M6.s1, f02, V8, 14 + c8);


							// f2[c8] x v[2 .. 16]
							DOT8i_1(M0.s0, f02, V0, 4 + c8);
							DOT8i_1(M0.s1, f02, V0, 6 + c8);
							DOT8i_1(M1.s0, f02, V0, 8 + c8);
							DOT8i_1(M1.s1, f02, V0, 10 + c8);

							DOT8i_1(M2.s0, f02, V0, 12 + c8);
							DOT8i_1(M2.s1, f02, V0, 14 + c8);
							DOT8i_1(M3.s0, f02, V8, 0 + c8);
							DOT8i_1(M3.s1, f02, V8, 2 + c8);

							DOT8i_1(M4.s0, f02, V8, 4 + c8);
							DOT8i_1(M4.s1, f02, V8, 6 + c8);
							DOT8i_1(M5.s0, f02, V8, 8 + c8);
							DOT8i_1(M5.s1, f02, V8, 10 + c8);

							DOT8i_1(M6.s0, f02, V8, 12 + c8);
							DOT8i_1(M6.s1, f02, V8, 14 + c8);


							// f2[c8] x v[2 .. 16]
							DOT8i_2(M0.s0, f02, V0, 4 + c8);
							DOT8i_2(M0.s1, f02, V0, 6 + c8);
							DOT8i_2(M1.s0, f02, V0, 8 + c8);
							DOT8i_2(M1.s1, f02, V0, 10 + c8);

							DOT8i_2(M2.s0, f02, V0, 12 + c8);
							DOT8i_2(M2.s1, f02, V0, 14 + c8);
							DOT8i_2(M3.s0, f02, V8, 0 + c8);
							DOT8i_2(M3.s1, f02, V8, 2 + c8);

							DOT8i_2(M4.s0, f02, V8, 4 + c8);
							DOT8i_2(M4.s1, f02, V8, 6 + c8);
							DOT8i_2(M5.s0, f02, V8, 8 + c8);
							DOT8i_2(M5.s1, f02, V8, 10 + c8);

							DOT8i_2(M6.s0, f02, V8, 12 + c8);
							DOT8i_2(M6.s1, f02, V8, 14 + c8);


							// f2[c8] x v[2 .. 16]
							DOT8i_3(M0.s0, f02, V0, 4 + c8);
							DOT8i_3(M0.s1, f02, V0, 6 + c8);
							DOT8i_3(M1.s0, f02, V0, 8 + c8);
							DOT8i_3(M1.s1, f02, V0, 10 + c8);

							DOT8i_3(M2.s0, f02, V0, 12 + c8);
							DOT8i_3(M2.s1, f02, V0, 14 + c8);
							DOT8i_3(M3.s0, f02, V8, 0 + c8);
							DOT8i_3(M3.s1, f02, V8, 2 + c8);

							DOT8i_3(M4.s0, f02, V8, 4 + c8);
							DOT8i_3(M4.s1, f02, V8, 6 + c8);
							DOT8i_3(M5.s0, f02, V8, 8 + c8);
							DOT8i_3(M5.s1, f02, V8, 10 + c8);

							DOT8i_3(M6.s0, f02, V8, 12 + c8);
							DOT8i_3(M6.s1, f02, V8, 14 + c8);


							// f2[c8] x v[2 .. 16]
							DOT8i_4(M0.s0, f02, V0, 4 + c8);
							DOT8i_4(M0.s1, f02, V0, 6 + c8);
							DOT8i_4(M1.s0, f02, V0, 8 + c8);
							DOT8i_4(M1.s1, f02, V0, 10 + c8);

							DOT8i_4(M2.s0, f02, V0, 12 + c8);
							DOT8i_4(M2.s1, f02, V0, 14 + c8);
							DOT8i_4(M3.s0, f02, V8, 0 + c8);
							DOT8i_4(M3.s1, f02, V8, 2 + c8);

							DOT8i_4(M4.s0, f02, V8, 4 + c8);
							DOT8i_4(M4.s1, f02, V8, 6 + c8);
							DOT8i_4(M5.s0, f02, V8, 8 + c8);
							DOT8i_4(M5.s1, f02, V8, 10 + c8);

							DOT8i_4(M6.s0, f02, V8, 12 + c8);
							DOT8i_4(M6.s1, f02, V8, 14 + c8);


							// f2[c8] x v[2 .. 16]
							DOT8i_5(M0.s0, f02, V0, 4 + c8);
							DOT8i_5(M0.s1, f02, V0, 6 + c8);
							DOT8i_5(M1.s0, f02, V0, 8 + c8);
							DOT8i_5(M1.s1, f02, V0, 10 + c8);

							DOT8i_5(M2.s0, f02, V0, 12 + c8);
							DOT8i_5(M2.s1, f02, V0, 14 + c8);
							DOT8i_5(M3.s0, f02, V8, 0 + c8);
							DOT8i_5(M3.s1, f02, V8, 2 + c8);

							DOT8i_5(M4.s0, f02, V8, 4 + c8);
							DOT8i_5(M4.s1, f02, V8, 6 + c8);
							DOT8i_5(M5.s0, f02, V8, 8 + c8);
							DOT8i_5(M5.s1, f02, V8, 10 + c8);

							DOT8i_5(M6.s0, f02, V8, 12 + c8);
							DOT8i_5(M6.s1, f02, V8, 14 + c8);

							// f2[c8] x v[2 .. 16]
							DOT8i_6(M0.s0, f02, V0, 4 + c8);
							DOT8i_6(M0.s1, f02, V0, 6 + c8);
							DOT8i_6(M1.s0, f02, V0, 8 + c8);
							DOT8i_6(M1.s1, f02, V0, 10 + c8);

							DOT8i_6(M2.s0, f02, V0, 12 + c8);
							DOT8i_6(M2.s1, f02, V0, 14 + c8);
							DOT8i_6(M3.s0, f02, V8, 0 + c8);
							DOT8i_6(M3.s1, f02, V8, 2 + c8);

							DOT8i_6(M4.s0, f02, V8, 4 + c8);
							DOT8i_6(M4.s1, f02, V8, 6 + c8);
							DOT8i_6(M5.s0, f02, V8, 8 + c8);
							DOT8i_6(M5.s1, f02, V8, 10 + c8);

							DOT8i_6(M6.s0, f02, V8, 12 + c8);
							DOT8i_6(M6.s1, f02, V8, 14 + c8);


							// f2[c8] x v[2 .. 16]
							DOT8i_7(M0.s0, f02, V0, 4 + c8);
							DOT8i_7(M0.s1, f02, V0, 6 + c8);
							DOT8i_7(M1.s0, f02, V0, 8 + c8);
							DOT8i_7(M1.s1, f02, V0, 10 + c8);

							DOT8i_7(M2.s0, f02, V0, 12 + c8);
							DOT8i_7(M2.s1, f02, V0, 14 + c8);
							DOT8i_7(M3.s0, f02, V8, 0 + c8);
							DOT8i_7(M3.s1, f02, V8, 2 + c8);

							DOT8i_7(M4.s0, f02, V8, 4 + c8);
							DOT8i_7(M4.s1, f02, V8, 6 + c8);
							DOT8i_7(M5.s0, f02, V8, 8 + c8);
							DOT8i_7(M5.s1, f02, V8, 10 + c8);

							DOT8i_7(M6.s0, f02, V8, 12 + c8);
							DOT8i_7(M6.s1, f02, V8, 14 + c8);


						}
				}
				barrier(CLK_LOCAL_MEM_FENCE);
		}

	//barrier(CLK_LOCAL_MEM_FENCE);


	// Store multiplies in SLM.
		{
			//barrier(CLK_LOCAL_MEM_FENCE);
			__local UNIT_TYPE_2 *M_write = (__local UNIT_TYPE_2 *)&V[lz * 7 * 8];
			M_write += lx;

			M_write[0 * 16] = M0;
			M_write[1 * 16] = M1;
			M_write[2 * 16] = M2;
			M_write[3 * 16] = M3;
			M_write[4 * 16] = M4;
			M_write[5 * 16] = M5;
			M_write[6 * 16] = M6;

			barrier(CLK_LOCAL_MEM_FENCE);
		}

		//if ((gz) % 2) return;

		if (lz < 7)
		{
			// Load multiplies from SLM.
			__local const UNIT_TYPE_2 *M_read = (__local UNIT_TYPE_2*)&V[lz * 8 ];
			M_read += lx;

			UNIT_TYPE_2 M0 = M_read[0 * 112];
			UNIT_TYPE_2 M1 = M_read[1 * 112];
			UNIT_TYPE_2 M2 = M_read[2 * 112];
			UNIT_TYPE_2 M3 = M_read[3 * 112];
			UNIT_TYPE_2 M4 = M_read[4 * 112];
			UNIT_TYPE_2 M5 = M_read[5 * 112];
			UNIT_TYPE_2 M6 = M_read[6 * 112];
			UNIT_TYPE_2 M7 = M_read[7 * 112];

			// Inverse Transform.
			UNIT_TYPE_2 x0 = M1 + M2;
			UNIT_TYPE_2 x1 = M1 - M2;

			UNIT_TYPE_2 x2 = M3 + M4;
			UNIT_TYPE_2 x3 = M3 - M4;

			UNIT_TYPE_2 x4 = M5 + M6;
			UNIT_TYPE_2 x5 = M5 - M6;

			UNIT_TYPE_2 S0 = M0 + x0 + x2 + x4;
			UNIT_TYPE_2 S1 = x1 + ((UNIT_TYPE)2.f)*x3 + ((UNIT_TYPE)0.5f)*x5;
			UNIT_TYPE_2 S2 = x0 + ((UNIT_TYPE)4.f)*x2 + ((UNIT_TYPE)0.25f)*x4;
			UNIT_TYPE_2 S3 = x1 + ((UNIT_TYPE)8.f)*x3 + ((UNIT_TYPE)0.125f)*x5;
			UNIT_TYPE_2 S4 = x0 + ((UNIT_TYPE)16.f)*x2 + ((UNIT_TYPE)0.0625f)*x4;
			UNIT_TYPE_2 S5 = x1 + ((UNIT_TYPE)32.f)*x3 + ((UNIT_TYPE)0.03125f)*x5 + M7;

			// Store output to global memory.
			uint p = gy * 6 + OUTPUT_PAD_BEFORE_SIZE_Y;
			uint q = gx * 14 + lz * 2 + OUTPUT_PAD_BEFORE_SIZE_X;
			uint k = gk * 16 + lx;

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
			__global UNIT_TYPE *O_write = (__global UNIT_TYPE *)&O[outindex];
#else
			__global UNIT_TYPE *O_write_0 = (__global UNIT_TYPE *)(&O[gn*PQK + k*Q*P + (p + 0)*Q + q]);
			__global UNIT_TYPE *O_write_1 = (__global UNIT_TYPE *)(&O[gn*PQK + k*Q*P + (p + 1)*Q + q]);
			__global UNIT_TYPE *O_write_2 = (__global UNIT_TYPE *)(&O[gn*PQK + k*Q*P + (p + 2)*Q + q]);
			__global UNIT_TYPE *O_write_3 = (__global UNIT_TYPE *)(&O[gn*PQK + k*Q*P + (p + 3)*Q + q]);
			__global UNIT_TYPE *O_write_4 = (__global UNIT_TYPE *)(&O[gn*PQK + k*Q*P + (p + 4)*Q + q]);
			__global UNIT_TYPE *O_write_5 = (__global UNIT_TYPE *)(&O[gn*PQK + k*Q*P + (p + 5)*Q + q]);
#endif

			// TODO: clip output by P, Q
			bool q0_in = q < Q - OUTPUT_PAD_AFTER_SIZE_X;
			bool q1_in = q + 1 < Q - OUTPUT_PAD_AFTER_SIZE_X;

			const uint K = FILTER_OFM_NUM;

			if (k < FILTER_OFM_NUM) {
				if (p < P - OUTPUT_PAD_AFTER_SIZE_Y) {
					if (q0_in) {

#if OUTPUT_LAYOUT_BYXF
#if BIAS_TERM
						O_write[0 * QK + 0 * K] = ACTIVATION(S0.s0 + bias[bias_index0], NL_M, NL_N);
#else
						O_write[0 * QK + 0 * K] = ACTIVATION(S0.s0, NL_M, NL_N);
#endif
#else
#if BIAS_TERM
						O_write_0[0] = ACTIVATION(S0.s0 + bias[bias_index0], NL_M, NL_N);
#else
						O_write_0[0] = ACTIVATION(S0.s0, NL_M, NL_N);
#endif
#endif 
					}
					if (q1_in) {
#if OUTPUT_LAYOUT_BYXF
#if BIAS_TERM
						O_write[0 * QK + 1 * K] = ACTIVATION(S0.s1 + bias[bias_index0], NL_M, NL_N);
#else
						O_write[0 * QK + 1 * K] = ACTIVATION(S0.s1, NL_M, NL_N);
#endif 
#else
#if BIAS_TERM
						O_write_0[1] = ACTIVATION(S0.s1 + bias[bias_index0], NL_M, NL_N);
#else
						O_write_0[1] = ACTIVATION(S0.s1, NL_M, NL_N);
#endif 
#endif 
					}
				}

				// row 1
				if (p + 1 < P - OUTPUT_PAD_AFTER_SIZE_Y) {
					if (q0_in) {
#if OUTPUT_LAYOUT_BYXF
#if BIAS_TERM
						O_write[1 * QK + 0 * K] = ACTIVATION(S1.s0 + bias[bias_index0], NL_M, NL_N);
#else
						O_write[1 * QK + 0 * K] = ACTIVATION(S1.s0, NL_M, NL_N);
#endif 
#else
#if BIAS_TERM
						O_write_1[0] = ACTIVATION(S1.s0 + bias[bias_index0], NL_M, NL_N);
#else
						O_write_1[0] = ACTIVATION(S1.s0, NL_M, NL_N);
#endif 
#endif 
					}
					if (q1_in) {
#if OUTPUT_LAYOUT_BYXF
#if BIAS_TERM
						O_write[1 * QK + 1 * K] = ACTIVATION(S1.s1 + bias[bias_index0], NL_M, NL_N);
#else
						O_write[1 * QK + 1 * K] = ACTIVATION(S1.s1, NL_M, NL_N);
#endif 
#else
#if BIAS_TERM
						O_write_1[1] = ACTIVATION(S1.s1 + bias[bias_index0], NL_M, NL_N);
#else
						O_write_1[1] = ACTIVATION(S1.s1, NL_M, NL_N);
#endif 
#endif 
					}
				}

				// row 2
				if (p + 2 < P - OUTPUT_PAD_AFTER_SIZE_Y) {
					if (q0_in) {
#if OUTPUT_LAYOUT_BYXF
#if BIAS_TERM
						O_write[2 * QK + 0 * K] = ACTIVATION(S2.s0 + bias[bias_index0], NL_M, NL_N);
#else
						O_write[2 * QK + 0 * K] = ACTIVATION(S2.s0, NL_M, NL_N);
#endif 
#else
#if BIAS_TERM
						O_write_2[0] = ACTIVATION(S2.s0 + bias[bias_index0], NL_M, NL_N);
#else
						O_write_2[0] = ACTIVATION(S2.s0, NL_M, NL_N);
#endif 
#endif 
					}
					if (q1_in) {
#if OUTPUT_LAYOUT_BYXF
#if BIAS_TERM
						O_write[2 * QK + 1 * K] = ACTIVATION(S2.s1 + bias[bias_index0], NL_M, NL_N);
#else
						O_write[2 * QK + 1 * K] = ACTIVATION(S2.s1, NL_M, NL_N);
#endif 
#else
#if BIAS_TERM
						O_write_2[1] = ACTIVATION(S2.s1 + bias[bias_index0], NL_M, NL_N);
#else
						O_write_2[1] = ACTIVATION(S2.s1, NL_M, NL_N);
#endif 
#endif 
					}
				}

				// row 3
				if (p + 3 < P - OUTPUT_PAD_AFTER_SIZE_Y) {
					if (q0_in) {
#if OUTPUT_LAYOUT_BYXF
#if BIAS_TERM
						O_write[3 * QK + 0 * K] = ACTIVATION(S3.s0 + bias[bias_index0], NL_M, NL_N);
#else
						O_write[3 * QK + 0 * K] = ACTIVATION(S3.s0, NL_M, NL_N);
#endif 
#else
#if BIAS_TERM
						O_write_3[0] = ACTIVATION(S3.s0 + bias[bias_index0], NL_M, NL_N);
#else
						O_write_3[0] = ACTIVATION(S3.s0, NL_M, NL_N);
#endif 
#endif
					}
					if (q1_in) {
#if OUTPUT_LAYOUT_BYXF
#if BIAS_TERM
						O_write[3 * QK + 1 * K] = ACTIVATION(S3.s1 + bias[bias_index0], NL_M, NL_N);
#else
						O_write[3 * QK + 1 * K] = ACTIVATION(S3.s1, NL_M, NL_N);
#endif 
#else
#if BIAS_TERM
						O_write_3[1] = ACTIVATION(S3.s1 + bias[bias_index0], NL_M, NL_N);
#else
						O_write_3[1] = ACTIVATION(S3.s1, NL_M, NL_N);
#endif   
#endif   
					}
				}
			}

			// row 4
			if (p + 4 < P - OUTPUT_PAD_AFTER_SIZE_Y) {
				if (q0_in) {
#if OUTPUT_LAYOUT_BYXF
#if BIAS_TERM
					O_write[4 * QK + 0 * K] = ACTIVATION(S4.s0 + bias[bias_index0], NL_M, NL_N);
#else
					O_write[4 * QK + 0 * K] = ACTIVATION(S4.s0, NL_M, NL_N);
#endif 
#else
#if BIAS_TERM
					O_write_4[0] = ACTIVATION(S4.s0 + bias[bias_index0], NL_M, NL_N);
#else
					O_write_4[0] = ACTIVATION(S4.s0, NL_M, NL_N);
#endif 
#endif
				}
				if (q1_in) {
#if OUTPUT_LAYOUT_BYXF
#if BIAS_TERM
					O_write[4 * QK + 1 * K] = ACTIVATION(S4.s1 + bias[bias_index0], NL_M, NL_N);
#else
					O_write[4 * QK + 1 * K] = ACTIVATION(S4.s1, NL_M, NL_N);
#endif 
#else
#if BIAS_TERM
					O_write_4[1] = ACTIVATION(S4.s1 + bias[bias_index0], NL_M, NL_N);
#else
					O_write_4[1] = ACTIVATION(S4.s1, NL_M, NL_N);
#endif   
#endif   
				}
			}

			// row 5
			if (p + 5 < P - OUTPUT_PAD_AFTER_SIZE_Y) {
				if (q0_in) {
#if OUTPUT_LAYOUT_BYXF
#if BIAS_TERM
					O_write[5 * QK + 0 * K] = ACTIVATION(S5.s0 + bias[bias_index0], NL_M, NL_N);
#else
					O_write[5 * QK + 0 * K] = ACTIVATION(S5.s0, NL_M, NL_N);
#endif 
#else
#if BIAS_TERM
					O_write_5[0] = ACTIVATION(S5.s0 + bias[bias_index0], NL_M, NL_N);
#else
					O_write_5[0] = ACTIVATION(S5.s0, NL_M, NL_N);
#endif 
#endif
				}
				if (q1_in) {
#if OUTPUT_LAYOUT_BYXF
#if BIAS_TERM
					O_write[5 * QK + 1 * K] = ACTIVATION(S5.s1 + bias[bias_index0], NL_M, NL_N);
#else
					O_write[5 * QK + 1 * K] = ACTIVATION(S5.s1, NL_M, NL_N);
#endif 
#else
#if BIAS_TERM
					O_write_5[1] = ACTIVATION(S5.s1 + bias[bias_index0], NL_M, NL_N);
#else
					O_write_5[1] = ACTIVATION(S5.s1, NL_M, NL_N);
#endif   
#endif   
				}
			}
		}

}
#undef UNIT_TYPE_2
#undef UNIT_TYPE_4
#undef UNIT_TYPE_8