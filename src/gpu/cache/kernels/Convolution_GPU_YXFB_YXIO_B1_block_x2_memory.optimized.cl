#ifdef RELU
#define ACTIVATION(output, input) output = max(input, 0.0f) + NEGATIVE_SLOPE * min(input, 0.0f);
#else
#define ACTIVATION(output, input) output = input;
#endif

__attribute__((reqd_work_group_size(LOCAL_WORK_GROUP_SIZE, 1, 1)))
KERNEL(Convolution_GPU_YXFB_YXIO_B1_block_x2_memory)(
	const __global float* input,
	__global float* output,
	const __global float* filter,
	const __global float* bias,
	uint split_idx)
{
#ifdef USE_VECTOR_8
        #define VECTOR_FLOAT float8
        #define BLOCK_READ(IN) as_float8(intel_sub_group_block_read8((const __global uint*)IN))
        #define BLOCK_WRITE(OUT, DATA) intel_sub_group_block_write8((__global uint*)OUT, as_uint8(DATA));
#endif
#ifdef USE_VECTOR_4
        #define VECTOR_FLOAT float4
        #define BLOCK_READ(IN) as_float4(intel_sub_group_block_read4((const __global uint*)IN))
        #define BLOCK_WRITE(OUT, DATA) intel_sub_group_block_write4((__global uint*)OUT, as_uint4(DATA));
#endif
#ifdef USE_VECTOR_2
        #define VECTOR_FLOAT float2
        #define BLOCK_READ(IN) as_float2(intel_sub_group_block_read2((const __global uint*)IN))
        #define BLOCK_WRITE(OUT, DATA) intel_sub_group_block_write2((__global uint*)OUT, as_uint2(DATA));
#endif

        const uint batch_num = INPUT_BATCH_NUM;
        const uint linear_id_xy = get_group_id(1) * X_PER_WORK_ITEM + OUTPUT_SIZE_X * get_group_id(2);
        uint global_id0 = ((get_group_id(0) * LOCAL_WORK_GROUP_SIZE) / batch_num) * batch_num + (linear_id_xy * FILTER_ARRAY_NUM + split_idx) * (FILTER_OUTPUT_FEATURE_NUM / OFM_PER_WORK_ITEM) * batch_num; 
#if X_PER_WORK_ITEM == 2
        uint global_id1 = ((get_group_id(0) * LOCAL_WORK_GROUP_SIZE) / batch_num) * batch_num + ( (1 + linear_id_xy) * FILTER_ARRAY_NUM + split_idx) * (FILTER_OUTPUT_FEATURE_NUM / OFM_PER_WORK_ITEM) * batch_num; 
#endif
        const uint out_batch_id = get_local_id(0) % INPUT_BATCH_NUM;
        const uint out_x = get_group_id(1) * X_PER_WORK_ITEM;
        const uint out_y = get_group_id(2);

        const uint out_id0 = (global_id0 / batch_num) * OFM_PER_WORK_ITEM * batch_num + out_batch_id;
#if X_PER_WORK_ITEM == 2
        const uint out_id1 = (global_id1 / batch_num) * OFM_PER_WORK_ITEM * batch_num + out_batch_id;
#endif

        const uint ofm_offset = (global_id0 * (OFM_PER_WORK_ITEM / batch_num)) % FILTER_OUTPUT_FEATURE_NUM;

        bool finish = out_x >= OUTPUT_LIMIT_SIZE_X || out_x < OUTPUT_OFFSET_SIZE_X 
                   || out_y >= OUTPUT_LIMIT_SIZE_Y || out_y < OUTPUT_OFFSET_SIZE_Y;

        const uint sub_group_id = get_local_id(0) % INPUT_BATCH_NUM;

        VECTOR_FLOAT _data0 = 0.f;
#if X_PER_WORK_ITEM == 2
        VECTOR_FLOAT _data1 = 0.f;
#endif

        if(!finish)
        {
            const int x = out_x * STRIDE_SIZE_X + INPUT_OFFSET_SIZE_X;
            const int y = out_y * STRIDE_SIZE_Y + INPUT_OFFSET_SIZE_Y;

            for (uint i = 0; i < FILTER_SIZE_Y; i++)
            {
                int input_offset_y = y + i;
                bool zero_y = input_offset_y >= INPUT_SIZE_Y || input_offset_y < 0;

                if(!zero_y)
                {
                    for (uint j = 0; j < FILTER_SIZE_X; j++)
                    {
                        int input_offset_x = x + j;
                    
                        bool zero_x0 = input_offset_x >= INPUT_SIZE_X || input_offset_x < 0;
#if X_PER_WORK_ITEM == 2
                        bool zero_x1 = input_offset_x + STRIDE_SIZE_X >= INPUT_SIZE_X || input_offset_x + STRIDE_SIZE_X < 0;
#endif
                        VECTOR_FLOAT _tmp0 = 0.f;
#if X_PER_WORK_ITEM == 2
                        VECTOR_FLOAT _tmp1 = 0.f;
#endif

                        uint input_idx = (input_offset_x + (input_offset_y * INPUT_SIZE_X)) * INPUT_FEATURE_NUM * INPUT_BATCH_NUM;
                        input_idx += split_idx * FILTER_INPUT_FEATURE_NUM * INPUT_BATCH_NUM;
                        input_idx += out_batch_id;

                        uint filter_idx = ofm_offset + sub_group_id + FILTER_INPUT_FEATURE_NUM * (FILTER_OUTPUT_FEATURE_NUM * (i * FILTER_SIZE_X + j));

#if INPUT_BATCH_NUM == 1
                        for(uint h = 0; h < FILTER_INPUT_FEATURE_NUM / 8; h++)
                        {
                            float _in0 = as_float(intel_sub_group_block_read((const __global uint*)input + input_idx));
#if X_PER_WORK_ITEM == 2
                            float _in1 = as_float(intel_sub_group_block_read((const __global uint*)input + (input_idx + INPUT_FEATURE_NUM * STRIDE_SIZE_X)));
#endif
                            float8 _input0 = TRANSPOSE_BLOCK_8(_in0);
#if X_PER_WORK_ITEM == 2
                            float8 _input1 = TRANSPOSE_BLOCK_8(_in1);
#endif
                            VECTOR_FLOAT _filter;
                            _filter = BLOCK_READ(filter + filter_idx); filter_idx += FILTER_OUTPUT_FEATURE_NUM;
                            _tmp0 = mad(_input0.s0, _filter, _tmp0);
#if X_PER_WORK_ITEM == 2
                            _tmp1 = mad(_input1.s0, _filter, _tmp1);
#endif
                            _filter = BLOCK_READ(filter + filter_idx); filter_idx += FILTER_OUTPUT_FEATURE_NUM;
                            _tmp0 = mad(_input0.s1, _filter, _tmp0);
#if X_PER_WORK_ITEM == 2
                            _tmp1 = mad(_input1.s1, _filter, _tmp1);
#endif

                            _filter = BLOCK_READ(filter + filter_idx); filter_idx += FILTER_OUTPUT_FEATURE_NUM;
                            _tmp0 = mad(_input0.s2, _filter, _tmp0);
#if X_PER_WORK_ITEM == 2
                            _tmp1 = mad(_input1.s2, _filter, _tmp1);
#endif

                            _filter = BLOCK_READ(filter + filter_idx); filter_idx += FILTER_OUTPUT_FEATURE_NUM;
                            _tmp0 = mad(_input0.s3, _filter, _tmp0);
#if X_PER_WORK_ITEM == 2
                            _tmp1 = mad(_input1.s3, _filter, _tmp1);
#endif

                            _filter = BLOCK_READ(filter + filter_idx); filter_idx += FILTER_OUTPUT_FEATURE_NUM;
                            _tmp0 = mad(_input0.s4, _filter, _tmp0);
#if X_PER_WORK_ITEM == 2
                            _tmp1 = mad(_input1.s4, _filter, _tmp1);
#endif

                            _filter = BLOCK_READ(filter + filter_idx); filter_idx += FILTER_OUTPUT_FEATURE_NUM;
                            _tmp0 = mad(_input0.s5, _filter, _tmp0);
#if X_PER_WORK_ITEM == 2
                            _tmp1 = mad(_input1.s5, _filter, _tmp1);
#endif

                            _filter = BLOCK_READ(filter + filter_idx); filter_idx += FILTER_OUTPUT_FEATURE_NUM;
                            _tmp0 = mad(_input0.s6, _filter, _tmp0);
#if X_PER_WORK_ITEM == 2
                            _tmp1 = mad(_input1.s6, _filter, _tmp1);
#endif

                            _filter = BLOCK_READ(filter + filter_idx); filter_idx += FILTER_OUTPUT_FEATURE_NUM;
                            _tmp0 = mad(_input0.s7, _filter, _tmp0);
#if X_PER_WORK_ITEM == 2
                            _tmp1 = mad(_input1.s7, _filter, _tmp1);
#endif

                            input_idx += 8 * INPUT_BATCH_NUM;
                        }
                        for (uint h = FILTER_INPUT_FEATURE_NUM - (FILTER_INPUT_FEATURE_NUM % 8); h < FILTER_INPUT_FEATURE_NUM; h++)
#else
                        for (uint h = 0; h < FILTER_INPUT_FEATURE_NUM; h++)
#endif
                        {
                            VECTOR_FLOAT _filter = BLOCK_READ(filter + filter_idx);
                            _tmp0 = mad(input[input_idx], _filter, _tmp0);
#if X_PER_WORK_ITEM == 2
                            _tmp1 = mad(input[input_idx + INPUT_FEATURE_NUM * STRIDE_SIZE_X], _filter, _tmp1);
#endif
                            filter_idx += FILTER_OUTPUT_FEATURE_NUM;
                            input_idx += INPUT_BATCH_NUM;
                        }
                        if(!zero_x0)
                            _data0 += _tmp0;
#if X_PER_WORK_ITEM == 2
                        if(!zero_x1)
                            _data1 += _tmp1;
#endif
                    }
                } 
            }
        }

        _data0 += BLOCK_READ(bias + ofm_offset);
#if X_PER_WORK_ITEM == 2
        _data1 += BLOCK_READ(bias + ofm_offset);
#endif

        ACTIVATION(_data0, _data0);
#if X_PER_WORK_ITEM == 2
        ACTIVATION(_data1, _data1);
#endif

        BLOCK_WRITE(output + out_id0, _data0);
#if X_PER_WORK_ITEM == 2
        if(!(out_x + 1 >= OUTPUT_LIMIT_SIZE_X || out_x + 1 < OUTPUT_OFFSET_SIZE_X))
            BLOCK_WRITE(output + out_id1, _data1);
#endif
        
#if defined(USE_VECTOR_8) || defined(USE_VECTOR_4) || defined(USE_VECTOR_2)
    #undef VECTOR_FLOAT
    #undef BLOCK_READ
    #undef BLOCK_WRITE
#endif
}

#undef ACTIVATION