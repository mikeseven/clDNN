/*/// -----------------------------------------------------------------------------------------------------------------------------
Copyright (c) 2016, Intel Corporation
/*/// -----------------------------------------------------------------------------------------------------------------------------

// --------------------------------------------------------------------------------------------------------------------------------
// L3_SIMD_4x8
// Input matrices dimensions: M x K x N
// Output matrix dimensions: M x N
// --------------------------------------------------------------------------------------------------------------------------------

#define DOT4i( _result, _A, _B, i)					\
    {									\
	_result = mad(_A.s0, sub_group_broadcast( _B.s0, i), _result);	\
	_result = mad(_A.s1, sub_group_broadcast( _B.s1, i), _result);	\
	_result = mad(_A.s2, sub_group_broadcast( _B.s2, i), _result);	\
	_result = mad(_A.s3, sub_group_broadcast( _B.s3, i), _result);	\
    }

__attribute__((reqd_work_group_size(8, 1, 8)))
__attribute__((intel_reqd_sub_group_size(8)))
__kernel void winograd_fused2x3(__global float *I,
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
    //               (DxC2)x(UxWx8c)
    __local float4 V[(4*2)*(2*16*2)]; // 8 KB

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


    uint gx = get_group_id(0);
    uint gy = get_group_id(1);
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
    
    bool x_in =  0 <= x && x < W;
    bool y0_in = 0 <= (y + 0) && (y + 0) < H && x_in;
    bool y1_in = 0 <= (y + 1) && (y + 1) < H && x_in;
    bool y2_in = 0 <= (y + 2) && (y + 2) < H && x_in;
    bool y3_in = 0 <= (y + 3) && (y + 3) < H && x_in;
    bool y4_in = 0 <= (y + 4) && (y + 4) < H && x_in;
    bool y5_in = 0 <= (y + 5) && (y + 5) < H && x_in;

    // #                                  x->
    // #     M0    M1    M2    M3    M4    M5    M6
    // #   +------------------------------------------
    // # u | 0 1 | 0 1 | 0 1 | 0 1 | 0 1 | 0 1 | 0 1 |
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
	const __global float *I_load = (const __global float*)&I[gn*HWC + ((uint)y)*W + ((uint)x)];


    uint lxm2 = lx % 2;
    uint lxb1 = (lx & 2)/2;

    //                                      
    __local float4 *V_write = &V[lxb1*256 + lz*4 + lxd4*2 + lxm2];
    __local const float4 *V_read = &V[lzm4*64 + lx];

    int2 coordU0;
    coordU0.x = (lzm4*24 + k*12);
    coordU0.y = 0;

    //const __global float *zeros4 = (const __global float *)O;
    const __global float *I_load_0 = &I_load[0*W]; //y0_in ? &I_load[0*W] : zeros4;
    const __global float *I_load_1 = &I_load[1*W]; //y1_in ? &I_load[1*W] : zeros4;
    const __global float *I_load_2 = &I_load[2*W]; //y2_in ? &I_load[2*W] : zeros4;
    const __global float *I_load_3 = &I_load[3*W]; //y3_in ? &I_load[3*W] : zeros4;
    const __global float *I_load_4 = &I_load[4*W]; //y4_in ? &I_load[4*W] : zeros4;
    const __global float *I_load_5 = &I_load[5*W]; //y5_in ? &I_load[5*W] : zeros4;
	
    __attribute__((opencl_unroll_hint(1)))
    for (uint c = lxm4; c < C4_up16; c += 4) {

        // 2*14 * 3 * 16 = 1344 MADs

        // Transform HxW x C        -> DxUxW x C
        //           6x16x16 inputs -> 4x2x16x16 winograd components.
        {
            // Workgroup loads 6x16x16 inputs.
            float4 I0 = y0_in ? (float4)(I_load_0[c*HW*4], I_load_0[c*HW*4+HW], I_load_0[c*HW*4+HW*2], I_load_0[c*HW*4+HW*3]):(float4)(0.0f,0.0f,0.0f,0.0f);
			float4 I1 = y1_in ? (float4)(I_load_1[c*HW*4], I_load_1[c*HW*4+HW], I_load_1[c*HW*4+HW*2], I_load_1[c*HW*4+HW*3]):(float4)(0.0f,0.0f,0.0f,0.0f);
			float4 I2 = y2_in ? (float4)(I_load_2[c*HW*4], I_load_2[c*HW*4+HW], I_load_2[c*HW*4+HW*2], I_load_2[c*HW*4+HW*3]):(float4)(0.0f,0.0f,0.0f,0.0f);
			float4 I3 = y3_in ? (float4)(I_load_3[c*HW*4], I_load_3[c*HW*4+HW], I_load_3[c*HW*4+HW*2], I_load_3[c*HW*4+HW*3]):(float4)(0.0f,0.0f,0.0f,0.0f);
			float4 I4 = y4_in ? (float4)(I_load_4[c*HW*4], I_load_4[c*HW*4+HW], I_load_4[c*HW*4+HW*2], I_load_4[c*HW*4+HW*3]):(float4)(0.0f,0.0f,0.0f,0.0f);
			float4 I5 = y5_in ? (float4)(I_load_5[c*HW*4], I_load_5[c*HW*4+HW], I_load_5[c*HW*4+HW*2], I_load_5[c*HW*4+HW*3]):(float4)(0.0f,0.0f,0.0f,0.0f);

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

        __local const float4 *V_read_c8 = V_read;

        __attribute__((opencl_unroll_hint(1)))
        for (int c8 = 0; c8 < 2; ++c8) {

            // 2*14 * 3 * 8 = 672 MADs

            // Fetch 8 channels of Winograd input components, spread across subgroup.
            // row 0
            float4 V00 = V_read_c8[0*8 + 32*0];
            float4 V01 = V_read_c8[1*8 + 32*0];
            float4 V02 = V_read_c8[2*8 + 32*0];
            float4 V03 = V_read_c8[3*8 + 32*0];

            // row 1
            float4 V10 = V_read_c8[0*8 + 32*1];
            float4 V11 = V_read_c8[1*8 + 32*1];
            float4 V12 = V_read_c8[2*8 + 32*1];
            float4 V13 = V_read_c8[3*8 + 32*1];

            V_read_c8 += 256;
			
            __attribute__((opencl_unroll_hint(2)))
            for (int c4 = 0; c4 < 2; ++c4) {

                // 2*14 * 3 * 4 = 336 MADs

                // Fetch 4 channels of Winograd filter components.
                int2 coordU = coordU0;
				uint coordU_x = coordU.x + get_sub_group_local_id();
                float4 f0;
				f0.s0 = *(float *)(&U[(coordU.y+0)*ODEPTH*KCOLSW*KROWSW + coordU_x]); // as_float4(intel_sub_group_block_read4(U, coordU));
				f0.s1 = *(float *)(&U[(coordU.y+1)*ODEPTH*KCOLSW*KROWSW + coordU_x]); // as_float4(intel_sub_group_block_read4(U, coordU));
				f0.s2 = *(float *)(&U[(coordU.y+2)*ODEPTH*KCOLSW*KROWSW + coordU_x]); // as_float4(intel_sub_group_block_read4(U, coordU));
				f0.s3 = *(float *)(&U[(coordU.y+3)*ODEPTH*KCOLSW*KROWSW + coordU_x]); // as_float4(intel_sub_group_block_read4(U, coordU));

                coordU.x += 8;
				coordU_x = coordU.x + get_sub_group_local_id();
                float4 f1;
				f1.s0 = *(float *)(&U[(coordU.y+0)*ODEPTH*KCOLSW*KROWSW + coordU_x]);
				f1.s1 = *(float *)(&U[(coordU.y+1)*ODEPTH*KCOLSW*KROWSW + coordU_x]);
				f1.s2 = *(float *)(&U[(coordU.y+2)*ODEPTH*KCOLSW*KROWSW + coordU_x]);
				f1.s3 = *(float *)(&U[(coordU.y+3)*ODEPTH*KCOLSW*KROWSW + coordU_x]);

                coordU.x += 8;
				coordU_x = coordU.x + get_sub_group_local_id();
                float4 f2;
				f2.s0 = *(float *)(&U[(coordU.y+0)*ODEPTH*KCOLSW*KROWSW + coordU_x]);
				f2.s1 = *(float *)(&U[(coordU.y+1)*ODEPTH*KCOLSW*KROWSW + coordU_x]);
				f2.s2 = *(float *)(&U[(coordU.y+2)*ODEPTH*KCOLSW*KROWSW + coordU_x]);
				f2.s3 = *(float *)(&U[(coordU.y+3)*ODEPTH*KCOLSW*KROWSW + coordU_x]);

                coordU0.y += 4;

                // row 0

                // f0 x v[0 .. 14]
                DOT4i(M0.s0, f0, V00, 0 + c4);
                DOT4i(M0.s1, f0, V00, 2 + c4);
                DOT4i(M1.s0, f0, V00, 4 + c4);
                DOT4i(M1.s1, f0, V00, 6 + c4);
                
                DOT4i(M2.s0, f0, V01, 0 + c4);
                DOT4i(M2.s1, f0, V01, 2 + c4);
                DOT4i(M3.s0, f0, V01, 4 + c4);
                DOT4i(M3.s1, f0, V01, 6 + c4);

                DOT4i(M4.s0, f0, V02, 0 + c4);
                DOT4i(M4.s1, f0, V02, 2 + c4);
                DOT4i(M5.s0, f0, V02, 4 + c4);
                DOT4i(M5.s1, f0, V02, 6 + c4);

                DOT4i(M6.s0, f0, V03, 0 + c4);
                DOT4i(M6.s1, f0, V03, 2 + c4);

                // f1[c4] x v[1 .. 15]
                DOT4i(M0.s0, f1, V00, 2 + c4);
                DOT4i(M0.s1, f1, V00, 4 + c4);
                DOT4i(M1.s0, f1, V00, 6 + c4);
                DOT4i(M1.s1, f1, V01, 0 + c4);

                DOT4i(M2.s0, f1, V01, 2 + c4);
                DOT4i(M2.s1, f1, V01, 4 + c4);
                DOT4i(M3.s0, f1, V01, 6 + c4);
                DOT4i(M3.s1, f1, V02, 0 + c4);

                DOT4i(M4.s0, f1, V02, 2 + c4);
                DOT4i(M4.s1, f1, V02, 4 + c4);
                DOT4i(M5.s0, f1, V02, 6 + c4);
                DOT4i(M5.s1, f1, V03, 0 + c4);

                DOT4i(M6.s0, f1, V03, 2 + c4);
                DOT4i(M6.s1, f1, V03, 4 + c4);

                // f2[c4] x v[2 .. 16]
                DOT4i(M0.s0, f2, V00, 4 + c4);
                DOT4i(M0.s1, f2, V00, 6 + c4);
                DOT4i(M1.s0, f2, V01, 0 + c4);
                DOT4i(M1.s1, f2, V01, 2 + c4);

                DOT4i(M2.s0, f2, V01, 4 + c4);
                DOT4i(M2.s1, f2, V01, 6 + c4);
                DOT4i(M3.s0, f2, V02, 0 + c4);
                DOT4i(M3.s1, f2, V02, 2 + c4);

                DOT4i(M4.s0, f2, V02, 4 + c4);
                DOT4i(M4.s1, f2, V02, 6 + c4);
                DOT4i(M5.s0, f2, V03, 0 + c4);
                DOT4i(M5.s1, f2, V03, 2 + c4);

                DOT4i(M6.s0, f2, V03, 4 + c4);
                DOT4i(M6.s1, f2, V03, 6 + c4);

                // row 1

                // f0 x v[0 .. 14]
                DOT4i(M0.s2, f0, V10, 0 + c4);
                DOT4i(M0.s3, f0, V10, 2 + c4);
                DOT4i(M1.s2, f0, V10, 4 + c4);
                DOT4i(M1.s3, f0, V10, 6 + c4);
                
                DOT4i(M2.s2, f0, V11, 0 + c4);
                DOT4i(M2.s3, f0, V11, 2 + c4);
                DOT4i(M3.s2, f0, V11, 4 + c4);
                DOT4i(M3.s3, f0, V11, 6 + c4);

                DOT4i(M4.s2, f0, V12, 0 + c4);
                DOT4i(M4.s3, f0, V12, 2 + c4);
                DOT4i(M5.s2, f0, V12, 4 + c4);
                DOT4i(M5.s3, f0, V12, 6 + c4);

                DOT4i(M6.s2, f0, V13, 0 + c4);
                DOT4i(M6.s3, f0, V13, 2 + c4);

                // f1 x v[1 .. 15]
                DOT4i(M0.s2, f1, V10, 2 + c4);
                DOT4i(M0.s3, f1, V10, 4 + c4);
                DOT4i(M1.s2, f1, V10, 6 + c4);
                DOT4i(M1.s3, f1, V11, 0 + c4);

                DOT4i(M2.s2, f1, V11, 2 + c4);
                DOT4i(M2.s3, f1, V11, 4 + c4);
                DOT4i(M3.s2, f1, V11, 6 + c4);
                DOT4i(M3.s3, f1, V12, 0 + c4);

                DOT4i(M4.s2, f1, V12, 2 + c4);
                DOT4i(M4.s3, f1, V12, 4 + c4);
                DOT4i(M5.s2, f1, V12, 6 + c4);
                DOT4i(M5.s3, f1, V13, 0 + c4);

                DOT4i(M6.s2, f1, V13, 2 + c4);
                DOT4i(M6.s3, f1, V13, 4 + c4);

                // f2 x v[2 .. 16]
                DOT4i(M0.s2, f2, V10, 4 + c4);
                DOT4i(M0.s3, f2, V10, 6 + c4);
                DOT4i(M1.s2, f2, V11, 0 + c4);
                DOT4i(M1.s3, f2, V11, 2 + c4);

                DOT4i(M2.s2, f2, V11, 4 + c4);
                DOT4i(M2.s3, f2, V11, 6 + c4);
                DOT4i(M3.s2, f2, V12, 0 + c4);
                DOT4i(M3.s3, f2, V12, 2 + c4);

                DOT4i(M4.s2, f2, V12, 4 + c4);
                DOT4i(M4.s3, f2, V12, 6 + c4);
                DOT4i(M5.s2, f2, V13, 0 + c4);
                DOT4i(M5.s3, f2, V13, 2 + c4);

                DOT4i(M6.s2, f2, V13, 4 + c4);
                DOT4i(M6.s3, f2, V13, 6 + c4);
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store multiplies in SLM.
    {
        __local float4 *M_write = &V[lz*7*8 + lx];

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
        __local const float8 *M_read = (__local float8*)&V[lz*8 + lxd4*224 + lxm4*2];
        
        float8 M0 = M_read[0*28];
        float8 M1 = M_read[1*28];
        float8 M2 = M_read[2*28];
        float8 M3 = M_read[3*28];

        // Inverse Transform.
        float8 S0 = M0 + M1 + M2;
        float8 S1 = M1 - M2 - M3;

        // Store output to global memory.
		static uint icntr = 0;
        uint p = gy*4 + Y_PADDING_BEFORE;
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

        __global float *O_write_0 = (__global float *)(&O[gn*PQK + k*Q*P + (p+0)*Q + q]);
        __global float *O_write_1 = (__global float *)(&O[gn*PQK + k*Q*P + (p+1)*Q + q]);
        __global float *O_write_2 = (__global float *)(&O[gn*PQK + k*Q*P + (p+2)*Q + q]);
        __global float *O_write_3 = (__global float *)(&O[gn*PQK + k*Q*P + (p+3)*Q + q]);

        // TODO: clip output by P, Q
        bool q0_in = q < Q - X_PADDING_AFTER;
        bool q1_in = q + 1 < Q - X_PADDING_AFTER;

		if (k < K) {
            if (p < P - Y_PADDING_AFTER) {
                if (q0_in) {
                    O_write_0[0] = (float)fmax(0.0f, S0.s0 + bias[bias_index0]);
                    O_write_0[0+Q*P] = (float)fmax(0.0f, S0.s4 + bias[bias_index0+1]);
                }
                if (q1_in) {
                    O_write_0[1] = (float)fmax(0.0f, S0.s1 + bias[bias_index0]);
                    O_write_0[1+Q*P] = (float)fmax(0.0f, S0.s5 + bias[bias_index0+1]);
                }
            }

            // row 1
            if (p + 1 < P - Y_PADDING_AFTER) {
                if (q0_in) {
                    O_write_1[0] = (float)fmax(0.0f, S1.s0 + bias[bias_index0]);
                    O_write_1[0+Q*P] = (float)fmax(0.0f, S1.s4 + bias[bias_index0+1]);
                }
                if (q1_in) {
                    O_write_1[1] = (float)fmax(0.0f, S1.s1 + bias[bias_index0]);
                    O_write_1[1+Q*P] = (float)fmax(0.0f, S1.s5 + bias[bias_index0+1]);
                }
            }

            // row 2
            if (p + 2 < P - Y_PADDING_AFTER) {
                if (q0_in) {
                    O_write_2[0] = (float)fmax(0.0f, S0.s2 + bias[bias_index0]);
                    O_write_2[0+Q*P] = (float)fmax(0.0f, S0.s6 + bias[bias_index0+1]);
                }
                if (q1_in) {
                    O_write_2[1] = (float)fmax(0.0f, S0.s3 + bias[bias_index0]);
                    O_write_2[1+Q*P] = (float)fmax(0.0f, S0.s7 + bias[bias_index0+1]);
                }
            }

            // row 3
            if (p + 3 < P - Y_PADDING_AFTER) {
                if (q0_in) {
                    O_write_3[0] = (float)fmax(0.0f, S1.s2 + bias[bias_index0]);
                    O_write_3[0+Q*P] = (float)fmax(0.0f, S1.s6 + bias[bias_index0+1]);
                }
                if (q1_in) {
                    O_write_3[1] = (float)fmax(0.0f, S1.s3 + bias[bias_index0]);        
                    O_write_3[1+Q*P] = (float)fmax(0.0f, S1.s7 + bias[bias_index0+1]);        
                }
            }
        }


    }
}

