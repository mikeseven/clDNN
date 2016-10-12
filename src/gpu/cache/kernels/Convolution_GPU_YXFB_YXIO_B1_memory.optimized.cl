#ifdef RELU
#define ACTIVATION(output, input) output = max(input, 0.0f) + NEGATIVE_SLOPE * min(input, 0.0f);
#else
#define ACTIVATION(output, input) output = input;
#endif

__attribute__((reqd_work_group_size(LOCAL_WORK_GROUP_SIZE, 1, 1)))
KERNEL(Convolution_GPU_YXFB_YXIO_B1_memory)(
	const __global float* input,
	__global float* output,
	const __global float* filter,
	const __global float* bias,
	uint split_idx)
{
        const uint batch_num = INPUT_BATCH_NUM;

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

        float _data0 = 0.f;

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
                            float8 _tmp_data0 = 0;
                            uint _input_idx = input_idx / 8;
                            for(uint h = 0; h < FILTER_INPUT_FEATURE_NUM / 8; h++)
                            {
                                float8 _input = vload8(_input_idx, input);
                                float8 _filter;
                                _filter.s0 = filter[filter_idx]; filter_idx += FILTER_OUTPUT_FEATURE_NUM;
                                _filter.s1 = filter[filter_idx]; filter_idx += FILTER_OUTPUT_FEATURE_NUM;
                                _filter.s2 = filter[filter_idx]; filter_idx += FILTER_OUTPUT_FEATURE_NUM;
                                _filter.s3 = filter[filter_idx]; filter_idx += FILTER_OUTPUT_FEATURE_NUM;
                                _filter.s4 = filter[filter_idx]; filter_idx += FILTER_OUTPUT_FEATURE_NUM;
                                _filter.s5 = filter[filter_idx]; filter_idx += FILTER_OUTPUT_FEATURE_NUM;
                                _filter.s6 = filter[filter_idx]; filter_idx += FILTER_OUTPUT_FEATURE_NUM;
                                _filter.s7 = filter[filter_idx]; filter_idx += FILTER_OUTPUT_FEATURE_NUM;
                                _tmp_data0 = mad(_input, _filter, _tmp_data0);
                                _input_idx += INPUT_BATCH_NUM;
                            }
                            input_idx += (FILTER_INPUT_FEATURE_NUM / 8) * 8 * INPUT_FEATURE_NUM;
                            _data0 += _tmp_data0.s0 + _tmp_data0.s1 + _tmp_data0.s2 + _tmp_data0.s3 +
                                      _tmp_data0.s4 + _tmp_data0.s5 + _tmp_data0.s6 + _tmp_data0.s7;
                            for (uint h = FILTER_INPUT_FEATURE_NUM - (FILTER_INPUT_FEATURE_NUM % 8); h < FILTER_INPUT_FEATURE_NUM; h++)
#else
                            for (uint h = 0; h < FILTER_INPUT_FEATURE_NUM; h++)
#endif
                            {
                                _data0 = mad(input[input_idx], filter[filter_idx], _data0);
                                filter_idx += FILTER_OUTPUT_FEATURE_NUM;
                                input_idx += INPUT_BATCH_NUM;
                            }
                        }
                    } 
                }
            }
        }

        _data0 += bias[ofm_offset + sub_group_id];
        ACTIVATION(_data0, _data0);

        output[out_id] = _data0;
}

#undef ACTIVATION