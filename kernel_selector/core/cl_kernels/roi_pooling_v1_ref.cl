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


#if FP16_SUPPORTED
    #pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#define ROI_NUM_ELEMENTS 5
#define POOLED_SIZE (POOLED_WIDTH * POOLED_HEIGHT)
#define POOLED_BATCH_SIZE (POOLED_SIZE * INPUT_FEATURE_NUM)

#define GET_ROI(x) round(x * SPATIAL_SCALE);

KERNEL(roi_pooling_gpu)(const __global UNIT_TYPE* input_data,
                        __global UNIT_TYPE* output,
                        const __global UNIT_TYPE* input_rois)
{
    const uint i = get_global_id(0);

    const uint pw    = (i % POOLED_WIDTH);
    const uint ph    = (i / POOLED_WIDTH) % POOLED_HEIGHT;
    const uint fm    = (i / POOLED_SIZE)  % INPUT_FEATURE_NUM;
    const uint roi   = (i / POOLED_BATCH_SIZE);

    const __global UNIT_TYPE* rois = input_rois + roi * ROI_NUM_ELEMENTS;
    const int roi_batch_ind = rois[0];
    const int roi_start_x = GET_ROI(rois[1]);
    const int roi_start_y = GET_ROI(rois[2]);
    const int roi_end_x   = GET_ROI(rois[3]);
    const int roi_end_y   = GET_ROI(rois[4]);

    // Force malformed ROIs to be 1x1
    const uint roi_width  = max(roi_end_x - roi_start_x + 1, 1);
    const uint roi_height = max(roi_end_y - roi_start_y + 1, 1);

    const uint ph_mult_height = ph * roi_height;
    const uint pw_mult_width  = pw * roi_width;
    
    int ystart = ph_mult_height / POOLED_HEIGHT;
    int xstart = pw_mult_width  / POOLED_WIDTH;
    int yend   = (ph_mult_height + roi_height) / POOLED_HEIGHT;
    if ( (yend * POOLED_HEIGHT) < (ph_mult_height + roi_height) ) {
        ++yend;
    }
    int xend   = (pw_mult_width + roi_width) / POOLED_WIDTH;
    if ( (xend * POOLED_WIDTH) < (pw_mult_width + roi_width) ) {
        ++xend;
    }

    ystart  = min(max(ystart + roi_start_y, 0), INPUT_SIZE_Y);
    yend    = min(max(yend   + roi_start_y, 0), INPUT_SIZE_Y);
    xstart  = min(max(xstart + roi_start_x, 0), INPUT_SIZE_X);
    xend    = min(max(xend   + roi_start_x, 0), INPUT_SIZE_X);

    UNIT_TYPE maxval = UNIT_VAL_MIN;
	
    if ( (yend == ystart) || (xend == xstart) )	// handle zero sized rect
    { 
    	maxval = 0;
    }

    const __global UNIT_TYPE* input = &input_data[INPUT_OFFSET + roi_batch_ind*INPUT_BATCH_PITCH + fm*INPUT_FEATURE_PITCH];

    for (int h = ystart; h < yend; ++h) {
        for (int w = xstart; w < xend; ++w) {
            int input_index = h*INPUT_Y_PITCH + w*INPUT_X_PITCH;
            UNIT_TYPE cur_val = input[input_index];
            maxval = fmax(maxval, cur_val);
        }
    }

    uint output_offset = OUTPUT_OFFSET + pw*OUTPUT_X_PITCH + ph*OUTPUT_Y_PITCH + fm*OUTPUT_FEATURE_PITCH + roi*OUTPUT_ROI_PITCH + roi_batch_ind*OUTPUT_BATCH_PITCH;
    output[output_offset] = maxval;
}