#ifdef RELU
#define ACTIVATION(output, input) output = max(input, 0.0f) + NEGATIVE_SLOPE * min(input, 0.0f);
#else
#define ACTIVATION(output, input) output = input;
#endif

__attribute__((reqd_work_group_size(LOCAL_WORK_GROUP_SIZE, 1, 1)))
KERNEL(Convolution_GPU_YXFB_YXIO_B1_vload_memory)(
	const __global float* input,
	__global float* output,
	const __global float* filter,
	const __global float* bias,
	uint split_idx)
{
        const uint batch_num = INPUT_BATCH_NUM;

#ifdef USE_VECTOR_8
        #define VLOAD vload8
        #define VSTORE vstore8
        #define VECTOR_SIZE 8
        #define VECTOR_FLOAT float8
#endif
#ifdef USE_VECTOR_4
        #define VLOAD vload4
        #define VSTORE vstore4
        #define VECTOR_SIZE 4
        #define VECTOR_FLOAT float4
#endif
#ifdef USE_VECTOR_2
        #define VLOAD vload2
        #define VSTORE vstore2
        #define VECTOR_SIZE 2
        #define VECTOR_FLOAT float2
#endif

        const uint linear_id_xy = get_global_id(1) + get_global_size(1) * get_global_id(2);
        uint global_id = (get_global_id(0) / batch_num) * batch_num + (linear_id_xy * FILTER_ARRAY_NUM + split_idx) * (FILTER_OUTPUT_FEATURE_NUM / OFM_PER_WORK_ITEM) * batch_num; 

        const uint out_batch_id = get_local_id(0) % INPUT_BATCH_NUM;
        const uint out_x = get_global_id(1);
        const uint out_y = get_global_id(2);

        const uint out_id = (global_id / batch_num) * OFM_PER_WORK_ITEM * batch_num + out_batch_id;

        const uint ofm_offset = (global_id * (OFM_PER_WORK_ITEM / batch_num)) % FILTER_OUTPUT_FEATURE_NUM;

        bool finish = out_x >= OUTPUT_LIMIT_SIZE_X || out_x < OUTPUT_OFFSET_SIZE_X 
                   || out_y >= OUTPUT_LIMIT_SIZE_Y || out_y < OUTPUT_OFFSET_SIZE_Y;

        const uint sub_group_id = get_local_id(0) % INPUT_BATCH_NUM;

        VECTOR_FLOAT _data0 = 0.f;

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
                    
                        bool zero = input_offset_x >= INPUT_SIZE_X || input_offset_x < 0;
    
                        if(!zero)
                        {
                            uint input_idx = (input_offset_x + (input_offset_y * INPUT_SIZE_X)) * INPUT_FEATURE_NUM * INPUT_BATCH_NUM;
                            input_idx += split_idx * FILTER_INPUT_FEATURE_NUM * INPUT_BATCH_NUM;
                            input_idx += out_batch_id;
                        
                            //sub_group_id used as offset to make each workitem load different filter, and then shuffle it
                            uint filter_idx = ofm_offset + sub_group_id + FILTER_INPUT_FEATURE_NUM * (FILTER_OUTPUT_FEATURE_NUM * (i * FILTER_SIZE_X + j));

#if INPUT_BATCH_NUM == 1
                            uint _input_idx = input_idx / 8;
                            for(uint h = 0; h < FILTER_INPUT_FEATURE_NUM / 8; h++)
                            {
                                float8 _input = vload8(_input_idx, input);
                                VECTOR_FLOAT _filter;
                                _filter = VLOAD(filter_idx / VECTOR_SIZE, filter); filter_idx += FILTER_OUTPUT_FEATURE_NUM;
                                _data0 = mad(_input.s0, _filter, _data0);

                                _filter = VLOAD(filter_idx / VECTOR_SIZE, filter); filter_idx += FILTER_OUTPUT_FEATURE_NUM;
                                _data0 = mad(_input.s1, _filter, _data0);

                                _filter = VLOAD(filter_idx / VECTOR_SIZE, filter); filter_idx += FILTER_OUTPUT_FEATURE_NUM;
                                _data0 = mad(_input.s2, _filter, _data0);

                                _filter = VLOAD(filter_idx / VECTOR_SIZE, filter); filter_idx += FILTER_OUTPUT_FEATURE_NUM;
                                _data0 = mad(_input.s3, _filter, _data0);

                                _filter = VLOAD(filter_idx / VECTOR_SIZE, filter); filter_idx += FILTER_OUTPUT_FEATURE_NUM;
                                _data0 = mad(_input.s4, _filter, _data0);

                                _filter = VLOAD(filter_idx / VECTOR_SIZE, filter); filter_idx += FILTER_OUTPUT_FEATURE_NUM;
                                _data0 = mad(_input.s5, _filter, _data0);

                                _filter = VLOAD(filter_idx / VECTOR_SIZE, filter); filter_idx += FILTER_OUTPUT_FEATURE_NUM;
                                _data0 = mad(_input.s6, _filter, _data0);

                                _filter = VLOAD(filter_idx / VECTOR_SIZE, filter); filter_idx += FILTER_OUTPUT_FEATURE_NUM;
                                _data0 = mad(_input.s7, _filter, _data0);

                                _input_idx += INPUT_BATCH_NUM;
                            }
                            input_idx += (FILTER_INPUT_FEATURE_NUM / 8) * 8 * INPUT_FEATURE_NUM;
                            for (uint h = FILTER_INPUT_FEATURE_NUM - (FILTER_INPUT_FEATURE_NUM % 8); h < FILTER_INPUT_FEATURE_NUM; h++)
#else
                            for (uint h = 0; h < FILTER_INPUT_FEATURE_NUM; h++)
#endif
                            {
                                VECTOR_FLOAT _filter = VLOAD(filter_idx / VECTOR_SIZE, filter);
                                _data0 = mad(input[input_idx], _filter, _data0);
                                filter_idx += FILTER_OUTPUT_FEATURE_NUM;
                                input_idx += INPUT_BATCH_NUM;
                            }
                        }
                    } 
                }
            }
        }

        _data0 += VLOAD(ofm_offset / VECTOR_SIZE, bias);
        ACTIVATION(_data0, _data0);

        VSTORE(_data0, out_id / VECTOR_SIZE, output);

#if defined(USE_VECTOR_8) || defined(USE_VECTOR_4) || defined(USE_VECTOR_2)
    #undef VLOAD
    #undef VSTORE
    #undef VECTOR_SIZE
    #undef VECTOR_FLOAT
#endif
}

#undef ACTIVATION