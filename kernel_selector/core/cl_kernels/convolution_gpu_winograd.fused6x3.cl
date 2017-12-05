/*/// -----------------------------------------------------------------------------------------------------------------------------
Copyright (c) 2016, Intel Corporation
/*/// -----------------------------------------------------------------------------------------------------------------------------

// --------------------------------------------------------------------------------------------------------------------------------
// L3_SIMD_4x8
// Input matrices dimensions: M x K x N
// Output matrix dimensions: M x N
// --------------------------------------------------------------------------------------------------------------------------------

#define DOT4i_LO( _result, _A, _B, i)					\
	{									\
	_result = mad(_A.s0, sub_group_broadcast( _B.s0, i), _result);	\
	_result = mad(_A.s1, sub_group_broadcast( _B.s1, i), _result);	\
	_result = mad(_A.s2, sub_group_broadcast( _B.s2, i), _result);	\
	_result = mad(_A.s3, sub_group_broadcast( _B.s3, i), _result);	\
    }

#define 	as_float2(x)   __builtin_astype((x), float2)

#define DOT4i_HI( _result, _A, _B, i)					\
    {									\
	_result = mad(_A.s4, sub_group_broadcast( _B.s0, i), _result);	\
	_result = mad(_A.s5, sub_group_broadcast( _B.s1, i), _result);	\
	_result = mad(_A.s6, sub_group_broadcast( _B.s2, i), _result);	\
	_result = mad(_A.s7, sub_group_broadcast( _B.s3, i), _result);	\
    }

__attribute__((reqd_work_group_size(8, 1, 8)))
__attribute__((intel_reqd_sub_group_size(8)))
__kernel void winograd_fused6x3(__global float *I,
                             //__read_only image2d_t U,
							 __global float * U,
                             #if BIAS_TERM
									__global float * bias,
							 #endif 
							 __global float *O
							 #ifdef SCHEDULE
									//,const __global ushort4 * schedule
							 #endif
)
{
    //               (C*D)x( Wxc)
    __local float4 V[(2*8)*(16*2)]; // 8 KB

    /* These constants are defined as precompiler macros during compilation. */
     const uint WC = W*C; 
	 const uint HW = H*W; 
     const uint HWC = H*WC; 
     const uint WC4 = WC >> 2; 
     const uint K16 = K >> 4; 
     const uint C4 = C >> 2; 
     const uint K2 = K >> 1; 
     const uint QK2 = Q*K2; 
     const uint QK = Q*K; 
     const uint PQK = P*QK; 


    const uint gx = get_group_id(0);
    const uint gy = get_group_id(1);
    const uint gz = get_group_id(2);
    uint gk = gz % K16;
    uint gn = gz / K16;

	uint glbx = get_global_id(0);

    const uint lx  = get_local_id(0);
    const uint lz  = get_local_id(2);

    uint lxd4 = lx >> 2;
    uint lxm4 = lx % 4;

    // Load 16x6 input tile, with 2 pixel overlap in X and y.
    // Compute 14x4 output tile.
    // Load 8 filters / thread.
    // 8 threads total: 2 filters x 4 winograd components. 16 filters total.
    uint x = gx*14 + lz*2 + lxd4 - px;
	uint x_ = gx*14;
    uint y = gy*6 - py;
    uint k = gk*16;
    uint c0 = lxm4 * 4;
    
    bool x_in = x < W;
    bool y0_in = (y + 0) < H && x_in;
    bool y1_in = (y + 1) < H && x_in;
    bool y2_in = (y + 2) < H && x_in;
    bool y3_in = (y + 3) < H && x_in;
    bool y4_in = (y + 4) < H && x_in;
    bool y5_in = (y + 5) < H && x_in;
    bool y6_in = (y + 6) < H && x_in;
    bool y7_in = (y + 7) < H && x_in;

    // #                                  x->
    // #     M0    M1    M2    M3    M4    M5    M6
    // #   +------------------------------------------
    // # k | 0 1 | 0 1 | 0 1 | 0 1 | 0 1 | 0 1 | 0 1 |
    // # | | 2 3 | 2 3 | 2 3 | 2 3 | 2 3 | 2 3 | 2 3 |
    // # v
    // #

    float4 M0 = (float4)(0, 0, 0, 0);
    float4 M1 = (float4)(0, 0, 0, 0);
    float4 M2 = (float4)(0, 0, 0, 0);
    float4 M3 = (float4)(0, 0, 0, 0);
    float4 M4 = (float4)(0, 0, 0, 0);
    float4 M5 = (float4)(0, 0, 0, 0);
    float4 M6 = (float4)(0, 0, 0, 0);


    //const __global float4 *I_load = (const __global float4*)&I[gn*HWC + ((uint)y)*WC + ((uint)x)*C];
	const __global float *I_load = (const __global float*)&I[gn*HWC + ((uint)y)*W + ((uint)x_)];


    uint lxm2 = lx % 2;
    uint lxb1 = (lx & 2)/2;

    //                            C      (D)     W            c
    //                            2       8     16            8
    __local float4 *V_write = &V[lxb1*256 + lz*4 + lxd4*2 + lxm2];

    //                                  D     WC
    __local const float4 *V_read = &V[lz*32 + lx];

    int2 coordU0;
    coordU0.x = (lz*48 + k*24);
    coordU0.y = 0;

    //const __global float *zeros4 = (const __global float *)O;
    /*const __global float *I_load_0 = &I_load[0*W]; //y0_in ? &I_load[0*W] : zeros4;
    const __global float *I_load_1 = &I_load[1*W]; //y1_in ? &I_load[1*W] : zeros4;
    const __global float *I_load_2 = &I_load[2*W]; //y2_in ? &I_load[2*W] : zeros4;
    const __global float *I_load_3 = &I_load[3*W]; //y3_in ? &I_load[3*W] : zeros4;
    const __global float *I_load_4 = &I_load[4*W]; //y4_in ? &I_load[4*W] : zeros4;
    const __global float *I_load_5 = &I_load[5*W]; //y5_in ? &I_load[5*W] : zeros4;
	const __global float *I_load_6 = &I_load[6*W];
	const __global float *I_load_7 = &I_load[7*W];*/

    __attribute__((opencl_unroll_hint(1)))
    for (uint c = 0; c < C4; c += 4) {
		{
            // Workgroup loads 8x16x16 inputs.
            float4 I0;
			float4 I1;
			float4 I2;
			float4 I3;
			float4 I4;
			float4 I5;
			float4 I6;
			float4 I7;
			//if (lz==0)
			{
				((float2*)V)[lx+8*(lz*16   )] = ((float2*)(&I_load[c*HW*4      +W*lz+4*HW*0]))[lx];
				((float2*)V)[lx+8*(lz*16+1 )] = ((float2*)(&I_load[c*HW*4      +W*lz+4*HW*1]))[lx];
				((float2*)V)[lx+8*(lz*16+2 )] = ((float2*)(&I_load[c*HW*4      +W*lz+4*HW*2]))[lx];
				((float2*)V)[lx+8*(lz*16+3 )] = ((float2*)(&I_load[c*HW*4      +W*lz+4*HW*3]))[lx];
				((float2*)V)[lx+8*(lz*16+4 )] = ((float2*)(&I_load[c*HW*4+HW   +W*lz+4*HW*0]))[lx];
				((float2*)V)[lx+8*(lz*16+5 )] = ((float2*)(&I_load[c*HW*4+HW   +W*lz+4*HW*1]))[lx];
				((float2*)V)[lx+8*(lz*16+6 )] = ((float2*)(&I_load[c*HW*4+HW   +W*lz+4*HW*2]))[lx];
				((float2*)V)[lx+8*(lz*16+7 )] = ((float2*)(&I_load[c*HW*4+HW   +W*lz+4*HW*3]))[lx];
				((float2*)V)[lx+8*(lz*16+8 )] = ((float2*)(&I_load[c*HW*4+HW*2 +W*lz+4*HW*0]))[lx];
				((float2*)V)[lx+8*(lz*16+9 )] = ((float2*)(&I_load[c*HW*4+HW*2 +W*lz+4*HW*1]))[lx];
				((float2*)V)[lx+8*(lz*16+10)] = ((float2*)(&I_load[c*HW*4+HW*2 +W*lz+4*HW*2]))[lx];
				((float2*)V)[lx+8*(lz*16+11)] = ((float2*)(&I_load[c*HW*4+HW*2 +W*lz+4*HW*3]))[lx];
				((float2*)V)[lx+8*(lz*16+12)] = ((float2*)(&I_load[c*HW*4+HW*3 +W*lz+4*HW*0]))[lx];
				((float2*)V)[lx+8*(lz*16+13)] = ((float2*)(&I_load[c*HW*4+HW*3 +W*lz+4*HW*1]))[lx];
				((float2*)V)[lx+8*(lz*16+14)] = ((float2*)(&I_load[c*HW*4+HW*3 +W*lz+4*HW*2]))[lx];
				((float2*)V)[lx+8*(lz*16+15)] = ((float2*)(&I_load[c*HW*4+HW*3 +W*lz+4*HW*3]))[lx];
			}
			
			barrier(CLK_LOCAL_MEM_FENCE);
			
			{
				__local float* Vf = (__local float*)(&V[lxm4*16+lxd4+2*lz]);
				I0 = y0_in * (float4)(Vf[16*4*0 ], Vf[16*4*1 ],  Vf[16*4*2 ], Vf[16*4*3 ]);
				I1 = y1_in * (float4)(Vf[16*4*4 ], Vf[16*4*5 ],  Vf[16*4*6 ], Vf[16*4*7 ]);
				I2 = y2_in * (float4)(Vf[16*4*8 ], Vf[16*4*9 ],  Vf[16*4*10], Vf[16*4*11]);
				I3 = y3_in * (float4)(Vf[16*4*12], Vf[16*4*13],  Vf[16*4*14], Vf[16*4*15]);
				I4 = y4_in * (float4)(Vf[16*4*16], Vf[16*4*17],  Vf[16*4*18], Vf[16*4*19]);
				I5 = y5_in * (float4)(Vf[16*4*20], Vf[16*4*21],  Vf[16*4*22], Vf[16*4*23]);
				I6 = y6_in * (float4)(Vf[16*4*24], Vf[16*4*25],  Vf[16*4*26], Vf[16*4*27]);
				I7 = y7_in * (float4)(Vf[16*4*28], Vf[16*4*29],  Vf[16*4*30], Vf[16*4*31]);			
			}

			barrier(CLK_LOCAL_MEM_FENCE);

            // Compute Winograd f6x3 data transform and store components in SLM.
            V_write[0*32] = I0 - 21.f/4*I2 + 21.f/4*I4 - I6;

            float4 x0 = I1 - 17.f/4*I3 + I5;
            float4 x1 = I2 - 17.f/4*I4 + I6;

            V_write[1*32] =  x1 + x0;
            V_write[2*32] =  x1 - x0;

            float4 x2 =  - 5.f*I3 + I1;
            float4 x3 = 4.f*I5 + x2;
            float4 x4 = 1.f/4*I2 + I6;
            float4 x5 = - 5.f/4*I4 + x4;

            V_write[3*32] = + 1.f/2 * x3 + x5;
            V_write[4*32] = - 1.f/2 * x3 + x5;

            float4 x6 = 4.f*I1 + I5;
            float4 x7 = - 5.f*I3 + x6;
            float4 x8 = 4.f*I2 + I6;
            float4 x9 = - 5.f*I4 + x8;

            V_write[5*32] = + 0.5f*x7 + x9;
            V_write[6*32] = - 0.5f*x7 + x9;

            V_write[7*32] = -I1 + 21.f/4*I3 -21.f/4*I5 + I7;

        }

        barrier(CLK_LOCAL_MEM_FENCE);

        __local const float4 *V_read_c8 = V_read;

		__attribute__((opencl_unroll_hint(1)))
        for (int c8 = 0; c8 < 2; ++c8) {

            // 2*14 * 3 * 8 = 672 MADs

            // Fetch 8 channels of Winograd input components, spread across subgroup.
            float4 V0 = V_read_c8[0*8 + 32*0];
            float4 V1 = V_read_c8[1*8 + 32*0];
            float4 V2 = V_read_c8[2*8 + 32*0];
            float4 V3 = V_read_c8[3*8 + 32*0];

            V_read_c8 += 256;

            {
                // filter 0

                // 2*14 * 3 * 4 = 336 MADs

                int2 coordU = coordU0;

				// Fetch 8 channels of Winograd components from f(k,s)
				uint coordU_x = coordU.x + get_sub_group_local_id();
                float8 f00;
				float8 f10;
				float2 temp;
				temp = as_float2(intel_sub_group_block_read2((__global uint *)&U[(coordU.y+0)*ODEPTH*KCOLSW*KROWSW + coordU_x]));
				f00.s0 = temp.s0;
				f10.s0 = temp.s1;
				temp = as_float2(intel_sub_group_block_read2((__global uint *)&U[(coordU.y+1)*ODEPTH*KCOLSW*KROWSW + coordU_x]));
				f00.s1 = temp.s0;
				f10.s1 = temp.s1;
				temp = as_float2(intel_sub_group_block_read2((__global uint *)&U[(coordU.y+2)*ODEPTH*KCOLSW*KROWSW + coordU_x]));
				f00.s2 = temp.s0;
				f10.s2 = temp.s1;
				temp = as_float2(intel_sub_group_block_read2((__global uint *)&U[(coordU.y+3)*ODEPTH*KCOLSW*KROWSW + coordU_x]));
				f00.s3 = temp.s0;
				f10.s3 = temp.s1;
				temp = as_float2(intel_sub_group_block_read2((__global uint *)&U[(coordU.y+4)*ODEPTH*KCOLSW*KROWSW + coordU_x]));
				f00.s4 = temp.s0;
				f10.s4 = temp.s1;
				temp = as_float2(intel_sub_group_block_read2((__global uint *)&U[(coordU.y+5)*ODEPTH*KCOLSW*KROWSW + coordU_x]));
				f00.s5 = temp.s0;
				f10.s5 = temp.s1;
				temp = as_float2(intel_sub_group_block_read2((__global uint *)&U[(coordU.y+6)*ODEPTH*KCOLSW*KROWSW + coordU_x]));
				f00.s6 = temp.s0;
				f10.s6 = temp.s1;
				temp = as_float2(intel_sub_group_block_read2((__global uint *)&U[(coordU.y+7)*ODEPTH*KCOLSW*KROWSW + coordU_x]));
				f00.s7 = temp.s0;
				f10.s7 = temp.s1;

				#define c4 0
                // f0 x v[0 .. 14]
                DOT4i_LO(M0.s0, f00, V0, 0 + c4);
                DOT4i_LO(M0.s1, f00, V0, 2 + c4);
                DOT4i_LO(M1.s0, f00, V0, 4 + c4);
                DOT4i_LO(M1.s1, f00, V0, 6 + c4);
                
                DOT4i_LO(M2.s0, f00, V1, 0 + c4);
                DOT4i_LO(M2.s1, f00, V1, 2 + c4);
                DOT4i_LO(M3.s0, f00, V1, 4 + c4);
                DOT4i_LO(M3.s1, f00, V1, 6 + c4);

                DOT4i_LO(M4.s0, f00, V2, 0 + c4);
                DOT4i_LO(M4.s1, f00, V2, 2 + c4);
                DOT4i_LO(M5.s0, f00, V2, 4 + c4);
                DOT4i_LO(M5.s1, f00, V2, 6 + c4);

                DOT4i_LO(M6.s0, f00, V3, 0 + c4);
                DOT4i_LO(M6.s1, f00, V3, 2 + c4);

				DOT4i_LO(M0.s2, f10, V0, 0 + c4);
                DOT4i_LO(M0.s3, f10, V0, 2 + c4);
                DOT4i_LO(M1.s2, f10, V0, 4 + c4);
                DOT4i_LO(M1.s3, f10, V0, 6 + c4);
                
                DOT4i_LO(M2.s2, f10, V1, 0 + c4);
                DOT4i_LO(M2.s3, f10, V1, 2 + c4);
                DOT4i_LO(M3.s2, f10, V1, 4 + c4);
                DOT4i_LO(M3.s3, f10, V1, 6 + c4);

                DOT4i_LO(M4.s2, f10, V2, 0 + c4);
                DOT4i_LO(M4.s3, f10, V2, 2 + c4);
                DOT4i_LO(M5.s2, f10, V2, 4 + c4);
                DOT4i_LO(M5.s3, f10, V2, 6 + c4);

                DOT4i_LO(M6.s2, f10, V3, 0 + c4);
                DOT4i_LO(M6.s3, f10, V3, 2 + c4);

				#undef c4

				#define c4 1
                // f0 x v[0 .. 14]
                DOT4i_HI(M0.s0, f00, V0, 0 + c4);
                DOT4i_HI(M0.s1, f00, V0, 2 + c4);
                DOT4i_HI(M1.s0, f00, V0, 4 + c4);
                DOT4i_HI(M1.s1, f00, V0, 6 + c4);
                
                DOT4i_HI(M2.s0, f00, V1, 0 + c4);
                DOT4i_HI(M2.s1, f00, V1, 2 + c4);
                DOT4i_HI(M3.s0, f00, V1, 4 + c4);
                DOT4i_HI(M3.s1, f00, V1, 6 + c4);

                DOT4i_HI(M4.s0, f00, V2, 0 + c4);
                DOT4i_HI(M4.s1, f00, V2, 2 + c4);
                DOT4i_HI(M5.s0, f00, V2, 4 + c4);
                DOT4i_HI(M5.s1, f00, V2, 6 + c4);

                DOT4i_HI(M6.s0, f00, V3, 0 + c4);
                DOT4i_HI(M6.s1, f00, V3, 2 + c4);

				// f0 x v[0 .. 14]
                DOT4i_HI(M0.s2, f10, V0, 0 + c4);
                DOT4i_HI(M0.s3, f10, V0, 2 + c4);
                DOT4i_HI(M1.s2, f10, V0, 4 + c4);
                DOT4i_HI(M1.s3, f10, V0, 6 + c4);
                
                DOT4i_HI(M2.s2, f10, V1, 0 + c4);
                DOT4i_HI(M2.s3, f10, V1, 2 + c4);
                DOT4i_HI(M3.s2, f10, V1, 4 + c4);
                DOT4i_HI(M3.s3, f10, V1, 6 + c4);

                DOT4i_HI(M4.s2, f10, V2, 0 + c4);
                DOT4i_HI(M4.s3, f10, V2, 2 + c4);
                DOT4i_HI(M5.s2, f10, V2, 4 + c4);
                DOT4i_HI(M5.s3, f10, V2, 6 + c4);

                DOT4i_HI(M6.s2, f10, V3, 0 + c4);
                DOT4i_HI(M6.s3, f10, V3, 2 + c4);

				#undef c4


                coordU.x += 16;
				coordU_x = coordU.x + get_sub_group_local_id();
                float8 f01;
				float8 f11;
				temp = as_float2(intel_sub_group_block_read2((__global uint *)&U[(coordU.y+0)*ODEPTH*KCOLSW*KROWSW + coordU_x]));
				f01.s0 = temp.s0;
				f11.s0 = temp.s1;
				temp = as_float2(intel_sub_group_block_read2((__global uint *)&U[(coordU.y+1)*ODEPTH*KCOLSW*KROWSW + coordU_x]));
				f01.s1 = temp.s0;
				f11.s1 = temp.s1;
				temp = as_float2(intel_sub_group_block_read2((__global uint *)&U[(coordU.y+2)*ODEPTH*KCOLSW*KROWSW + coordU_x]));
				f01.s2 = temp.s0;
				f11.s2 = temp.s1;
				temp = as_float2(intel_sub_group_block_read2((__global uint *)&U[(coordU.y+3)*ODEPTH*KCOLSW*KROWSW + coordU_x]));
				f01.s3 = temp.s0;
				f11.s3 = temp.s1;
				temp = as_float2(intel_sub_group_block_read2((__global uint *)&U[(coordU.y+4)*ODEPTH*KCOLSW*KROWSW + coordU_x]));
				f01.s4 = temp.s0;
				f11.s4 = temp.s1;
				temp = as_float2(intel_sub_group_block_read2((__global uint *)&U[(coordU.y+5)*ODEPTH*KCOLSW*KROWSW + coordU_x]));
				f01.s5 = temp.s0;
				f11.s5 = temp.s1;
				temp = as_float2(intel_sub_group_block_read2((__global uint *)&U[(coordU.y+6)*ODEPTH*KCOLSW*KROWSW + coordU_x]));
				f01.s6 = temp.s0;
				f11.s6 = temp.s1;
				temp = as_float2(intel_sub_group_block_read2((__global uint *)&U[(coordU.y+7)*ODEPTH*KCOLSW*KROWSW + coordU_x]));
				f01.s7 = temp.s0;
				f11.s7 = temp.s1;

				#define c4 0
                // f1[c4] x v[1 .. 15]
                DOT4i_LO(M0.s0, f01, V0, 2 + c4);
                DOT4i_LO(M0.s1, f01, V0, 4 + c4);
                DOT4i_LO(M1.s0, f01, V0, 6 + c4);
                DOT4i_LO(M1.s1, f01, V1, 0 + c4);

                DOT4i_LO(M2.s0, f01, V1, 2 + c4);
                DOT4i_LO(M2.s1, f01, V1, 4 + c4);
                DOT4i_LO(M3.s0, f01, V1, 6 + c4);
                DOT4i_LO(M3.s1, f01, V2, 0 + c4);

                DOT4i_LO(M4.s0, f01, V2, 2 + c4);
                DOT4i_LO(M4.s1, f01, V2, 4 + c4);
                DOT4i_LO(M5.s0, f01, V2, 6 + c4);
                DOT4i_LO(M5.s1, f01, V3, 0 + c4);

                DOT4i_LO(M6.s0, f01, V3, 2 + c4);
                DOT4i_LO(M6.s1, f01, V3, 4 + c4);

				// f1 x v[1 .. 15]
                DOT4i_LO(M0.s2, f11, V0, 2 + c4);
                DOT4i_LO(M0.s3, f11, V0, 4 + c4);
                DOT4i_LO(M1.s2, f11, V0, 6 + c4);
                DOT4i_LO(M1.s3, f11, V1, 0 + c4);

                DOT4i_LO(M2.s2, f11, V1, 2 + c4);
                DOT4i_LO(M2.s3, f11, V1, 4 + c4);
                DOT4i_LO(M3.s2, f11, V1, 6 + c4);
                DOT4i_LO(M3.s3, f11, V2, 0 + c4);

                DOT4i_LO(M4.s2, f11, V2, 2 + c4);
                DOT4i_LO(M4.s3, f11, V2, 4 + c4);
                DOT4i_LO(M5.s2, f11, V2, 6 + c4);
                DOT4i_LO(M5.s3, f11, V3, 0 + c4);

                DOT4i_LO(M6.s2, f11, V3, 2 + c4);
                DOT4i_LO(M6.s3, f11, V3, 4 + c4);

				#undef c4

				#define c4 1
                // f1[c4] x v[1 .. 15]
                DOT4i_HI(M0.s0, f01, V0, 2 + c4);
                DOT4i_HI(M0.s1, f01, V0, 4 + c4);
                DOT4i_HI(M1.s0, f01, V0, 6 + c4);
                DOT4i_HI(M1.s1, f01, V1, 0 + c4);

                DOT4i_HI(M2.s0, f01, V1, 2 + c4);
                DOT4i_HI(M2.s1, f01, V1, 4 + c4);
                DOT4i_HI(M3.s0, f01, V1, 6 + c4);
                DOT4i_HI(M3.s1, f01, V2, 0 + c4);

                DOT4i_HI(M4.s0, f01, V2, 2 + c4);
                DOT4i_HI(M4.s1, f01, V2, 4 + c4);
                DOT4i_HI(M5.s0, f01, V2, 6 + c4);
                DOT4i_HI(M5.s1, f01, V3, 0 + c4);

                DOT4i_HI(M6.s0, f01, V3, 2 + c4);
                DOT4i_HI(M6.s1, f01, V3, 4 + c4);

                // f1 x v[1 .. 15]
                DOT4i_HI(M0.s2, f11, V0, 2 + c4);
                DOT4i_HI(M0.s3, f11, V0, 4 + c4);
                DOT4i_HI(M1.s2, f11, V0, 6 + c4);
                DOT4i_HI(M1.s3, f11, V1, 0 + c4);

                DOT4i_HI(M2.s2, f11, V1, 2 + c4);
                DOT4i_HI(M2.s3, f11, V1, 4 + c4);
                DOT4i_HI(M3.s2, f11, V1, 6 + c4);
                DOT4i_HI(M3.s3, f11, V2, 0 + c4);

                DOT4i_HI(M4.s2, f11, V2, 2 + c4);
                DOT4i_HI(M4.s3, f11, V2, 4 + c4);
                DOT4i_HI(M5.s2, f11, V2, 6 + c4);
                DOT4i_HI(M5.s3, f11, V3, 0 + c4);

                DOT4i_HI(M6.s2, f11, V3, 2 + c4);
                DOT4i_HI(M6.s3, f11, V3, 4 + c4);

				#undef c4


                coordU.x += 16;
				coordU_x = coordU.x + get_sub_group_local_id();
                float8 f02;
				float8 f12;
				temp = as_float2(intel_sub_group_block_read2((__global uint *)&U[(coordU.y+0)*ODEPTH*KCOLSW*KROWSW + coordU_x]));
				f02.s0 = temp.s0;
				f12.s0 = temp.s1;
				temp = as_float2(intel_sub_group_block_read2((__global uint *)&U[(coordU.y+1)*ODEPTH*KCOLSW*KROWSW + coordU_x]));
				f02.s1 = temp.s0;
				f12.s1 = temp.s1;
				temp = as_float2(intel_sub_group_block_read2((__global uint *)&U[(coordU.y+2)*ODEPTH*KCOLSW*KROWSW + coordU_x]));
				f02.s2 = temp.s0;
				f12.s2 = temp.s1;
				temp = as_float2(intel_sub_group_block_read2((__global uint *)&U[(coordU.y+3)*ODEPTH*KCOLSW*KROWSW + coordU_x]));
				f02.s3 = temp.s0;
				f12.s3 = temp.s1;
				temp = as_float2(intel_sub_group_block_read2((__global uint *)&U[(coordU.y+4)*ODEPTH*KCOLSW*KROWSW + coordU_x]));
				f02.s4 = temp.s0;
				f12.s4 = temp.s1;
				temp = as_float2(intel_sub_group_block_read2((__global uint *)&U[(coordU.y+5)*ODEPTH*KCOLSW*KROWSW + coordU_x]));
				f02.s5 = temp.s0;
				f12.s5 = temp.s1;
				temp = as_float2(intel_sub_group_block_read2((__global uint *)&U[(coordU.y+6)*ODEPTH*KCOLSW*KROWSW + coordU_x]));
				f02.s6 = temp.s0;
				f12.s6 = temp.s1;
				temp = as_float2(intel_sub_group_block_read2((__global uint *)&U[(coordU.y+7)*ODEPTH*KCOLSW*KROWSW + coordU_x]));
				f02.s7 = temp.s0;
				f12.s7 = temp.s1;

                //coordU0.y += 4;
				coordU.x += 16;
				
                // multiply channels 0..3

#define c4 0
                // f2[c4] x v[2 .. 16]
                DOT4i_LO(M0.s0, f02, V0, 4 + c4);
                DOT4i_LO(M0.s1, f02, V0, 6 + c4);
                DOT4i_LO(M1.s0, f02, V1, 0 + c4);
                DOT4i_LO(M1.s1, f02, V1, 2 + c4);

                DOT4i_LO(M2.s0, f02, V1, 4 + c4);
                DOT4i_LO(M2.s1, f02, V1, 6 + c4);
                DOT4i_LO(M3.s0, f02, V2, 0 + c4);
                DOT4i_LO(M3.s1, f02, V2, 2 + c4);

                DOT4i_LO(M4.s0, f02, V2, 4 + c4);
                DOT4i_LO(M4.s1, f02, V2, 6 + c4);
                DOT4i_LO(M5.s0, f02, V3, 0 + c4);
                DOT4i_LO(M5.s1, f02, V3, 2 + c4);

                DOT4i_LO(M6.s0, f02, V3, 4 + c4);
                DOT4i_LO(M6.s1, f02, V3, 6 + c4);

                // f2 x v[2 .. 16]
                DOT4i_LO(M0.s2, f12, V0, 4 + c4);
                DOT4i_LO(M0.s3, f12, V0, 6 + c4);
                DOT4i_LO(M1.s2, f12, V1, 0 + c4);
                DOT4i_LO(M1.s3, f12, V1, 2 + c4);

                DOT4i_LO(M2.s2, f12, V1, 4 + c4);
                DOT4i_LO(M2.s3, f12, V1, 6 + c4);
                DOT4i_LO(M3.s2, f12, V2, 0 + c4);
                DOT4i_LO(M3.s3, f12, V2, 2 + c4);

                DOT4i_LO(M4.s2, f12, V2, 4 + c4);
                DOT4i_LO(M4.s3, f12, V2, 6 + c4);
                DOT4i_LO(M5.s2, f12, V3, 0 + c4);
                DOT4i_LO(M5.s3, f12, V3, 2 + c4);

                DOT4i_LO(M6.s2, f12, V3, 4 + c4);
                DOT4i_LO(M6.s3, f12, V3, 6 + c4);
#undef c4
#define c4 1

                // f2[c4] x v[2 .. 16]
                DOT4i_HI(M0.s0, f02, V0, 4 + c4);
                DOT4i_HI(M0.s1, f02, V0, 6 + c4);
                DOT4i_HI(M1.s0, f02, V1, 0 + c4);
                DOT4i_HI(M1.s1, f02, V1, 2 + c4);

                DOT4i_HI(M2.s0, f02, V1, 4 + c4);
                DOT4i_HI(M2.s1, f02, V1, 6 + c4);
                DOT4i_HI(M3.s0, f02, V2, 0 + c4);
                DOT4i_HI(M3.s1, f02, V2, 2 + c4);

                DOT4i_HI(M4.s0, f02, V2, 4 + c4);
                DOT4i_HI(M4.s1, f02, V2, 6 + c4);
                DOT4i_HI(M5.s0, f02, V3, 0 + c4);
                DOT4i_HI(M5.s1, f02, V3, 2 + c4);

                DOT4i_HI(M6.s0, f02, V3, 4 + c4);
                DOT4i_HI(M6.s1, f02, V3, 6 + c4);

				// f2 x v[2 .. 16]
                DOT4i_HI(M0.s2, f12, V0, 4 + c4);
                DOT4i_HI(M0.s3, f12, V0, 6 + c4);
                DOT4i_HI(M1.s2, f12, V1, 0 + c4);
                DOT4i_HI(M1.s3, f12, V1, 2 + c4);

                DOT4i_HI(M2.s2, f12, V1, 4 + c4);
                DOT4i_HI(M2.s3, f12, V1, 6 + c4);
                DOT4i_HI(M3.s2, f12, V2, 0 + c4);
                DOT4i_HI(M3.s3, f12, V2, 2 + c4);

                DOT4i_HI(M4.s2, f12, V2, 4 + c4);
                DOT4i_HI(M4.s3, f12, V2, 6 + c4);
                DOT4i_HI(M5.s2, f12, V3, 0 + c4);
                DOT4i_HI(M5.s3, f12, V3, 2 + c4);

                DOT4i_HI(M6.s2, f12, V3, 4 + c4);
                DOT4i_HI(M6.s3, f12, V3, 6 + c4);
#undef c4
            }

            {
                // filter 1

                coordU0.y += 8;
				//coordU.x += 16;

            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store multiplies in SLM.
    {
        __local float2 *M_write = (__local float2 *)&V[lz*56];
        M_write += lx;

        M_write[0*16] = M0.s01;
        M_write[1*16] = M1.s01;
        M_write[2*16] = M2.s01;
        M_write[3*16] = M3.s01;
        M_write[4*16] = M4.s01;
        M_write[5*16] = M5.s01;
        M_write[6*16] = M6.s01;

        M_write += 8;

        M_write[0*16] = M0.s23;
        M_write[1*16] = M1.s23;
        M_write[2*16] = M2.s23;
        M_write[3*16] = M3.s23;
        M_write[4*16] = M4.s23;
        M_write[5*16] = M5.s23;
        M_write[6*16] = M6.s23;

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lz < 7) 
	{
        // Load multiplies from SLM.
        __local const float4 *M_read = (__local float4*)&V[lz*8 + lx];
        
        float4 M0 = M_read[0*56];
        float4 M1 = M_read[1*56];
        float4 M2 = M_read[2*56];
        float4 M3 = M_read[3*56];
        float4 M4 = M_read[4*56];
        float4 M5 = M_read[5*56];
        float4 M6 = M_read[6*56];
        float4 M7 = M_read[7*56];

        // Inverse Transform.
        float4 x0 = M1 + M2;
        float4 x1 = M1 - M2;

        float4 x2 = M3 + M4;
        float4 x3 = M3 - M4;
        
        float4 x4 = M5 + M6;
        float4 x5 = M5 - M6;

        float4 S0 = M0 + x0 +      x2 +          x4;
        float4 S1 =      x1 +  2.f*x3 + 1.f/ 2.f*x5;
        float4 S2 =      x0 +  4.f*x2 + 1.f/ 4.f*x4;
        float4 S3 =      x1 +  8.f*x3 + 1.f/ 8.f*x5;
        float4 S4 =      x0 + 16.f*x2 + 1.f/16.f*x4;
        float4 S5 =      x1 + 32.f*x3 + 1.f/32.f*x5 + M7;

		// Store output to global memory.
        uint p = gy*6 + Y_PADDING_BEFORE;
        uint q = gx*14 + lz*2 + X_PADDING_BEFORE;
        uint k = gk*16 + lx*2;

		// bias and activation
		#if BIAS_TERM
		#if BIAS_PER_OUTPUT
            const unsigned bias_index0 = k*OUTPUT_SIZE_X*OUTPUT_SIZE_Y + trow*OUTPUT_SIZE_X + q;
			const unsigned bias_index1 = bias_index0 + 1;
		#else
            const unsigned bias_index0 = k;
			const unsigned bias_index1 = bias_index0;
		#endif
		#else
			float bias[1] = { 0.0f };
	        const unsigned bias_index0 = 0;
			const unsigned bias_index1 = 0;
		#endif
		//S0 = fmax(0.0f, S0 + bias[bias_index0]);
		//S1 = fmax(0.0f, S1 + bias[bias_index1]);

		float bias0 = bias[bias_index0  ];
		float bias1 = bias[bias_index0+1];
		S0.s01 = (float2)fmax(0.0f, S0.s01 + bias0);
        S0.s23 = (float2)fmax(0.0f, S0.s23 + bias1);
		S1.s01 = (float2)fmax(0.0f, S1.s01 + bias0);
        S1.s23 = (float2)fmax(0.0f, S1.s23 + bias1);
		S2.s01 = (float2)fmax(0.0f, S2.s01 + bias0);
        S2.s23 = (float2)fmax(0.0f, S2.s23 + bias1);
		S3.s01 = (float2)fmax(0.0f, S3.s01 + bias0);
        S3.s23 = (float2)fmax(0.0f, S3.s23 + bias1);
		S4.s01 = (float2)fmax(0.0f, S4.s01 + bias0);
        S4.s23 = (float2)fmax(0.0f, S4.s23 + bias1);
		S5.s01 = (float2)fmax(0.0f, S5.s01 + bias0);
        S5.s23 = (float2)fmax(0.0f, S5.s23 + bias1);
        
        __global float *O_write_0 = (__global float *)(&O[gn*PQK + k*Q*P + (p+0)*Q + q]);
        __global float *O_write_1 = (__global float *)(&O[gn*PQK + k*Q*P + (p+1)*Q + q]);
        __global float *O_write_2 = (__global float *)(&O[gn*PQK + k*Q*P + (p+2)*Q + q]);
        __global float *O_write_3 = (__global float *)(&O[gn*PQK + k*Q*P + (p+3)*Q + q]);
		__global float *O_write_4 = (__global float *)(&O[gn*PQK + k*Q*P + (p+4)*Q + q]);
		__global float *O_write_5 = (__global float *)(&O[gn*PQK + k*Q*P + (p+5)*Q + q]);

        // TODO: clip output by P, Q
        bool q0_in = q < Q - X_PADDING_AFTER;
        bool q1_in = q + 1 < Q - X_PADDING_AFTER;

		if (k < K) {
            if (p < P - Y_PADDING_AFTER) {
                if (q0_in) {
                    O_write_0[0] = S0.s0;
                    O_write_0[0+Q*P] = S0.s2;
                }
                if (q1_in) {
                    O_write_0[1] = S0.s1;
                    O_write_0[1+Q*P] = S0.s3;
                }
            }

            // row 1
            if (p + 1 < P - Y_PADDING_AFTER) {
                if (q0_in) {
                    O_write_1[0] = S1.s0;
                    O_write_1[0+Q*P] = S1.s2;
                }
                if (q1_in) {
                    O_write_1[1] = S1.s1;
                    O_write_1[1+Q*P] = S1.s3;
                }
            }

            // row 2
            if (p + 2 < P - Y_PADDING_AFTER) {
                if (q0_in) {
                    O_write_2[0] = S2.s0;
                    O_write_2[0+Q*P] = S2.s2;
                }
                if (q1_in) {
                    O_write_2[1] = S2.s1;
                    O_write_2[1+Q*P] = S2.s3;
                }
            }

            // row 3
            if (p + 3 < P - Y_PADDING_AFTER) {
                if (q0_in) {
                    O_write_3[0] = S3.s0;
                    O_write_3[0+Q*P] = S3.s2;
                }
                if (q1_in) {
                    O_write_3[1] = S3.s1;        
                    O_write_3[1+Q*P] = S3.s3;        
                }
            }

			// row 4
            if (p + 4 < P - Y_PADDING_AFTER) {
                if (q0_in) {
                    O_write_4[0] = S4.s0;
                    O_write_4[0+Q*P] = S4.s2;
                }
                if (q1_in) {
                    O_write_4[1] = S4.s1;        
                    O_write_4[1+Q*P] = S4.s3;        
                }
            }

			// row 5
            if (p + 5 < P - Y_PADDING_AFTER) {
                if (q0_in) {
                    O_write_5[0] = S5.s0;
                    O_write_5[0+Q*P] = S5.s2;
                }
                if (q1_in) {
                    O_write_5[1] = S5.s1;        
                    O_write_5[1+Q*P] = S5.s3;        
                }
            }
        }


    }
}

