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


KERNEL(roi_pooling_gpu)(const __global UNIT_TYPE* input_data,
                        const __global UNIT_TYPE* input_rois,
                        __global UNIT_TYPE* output)
{
    int index = get_global_id(0);

    int pw = index % POOLED_WIDTH;
    int ph = (index / POOLED_WIDTH) % POOLED_HEIGHT;
    int c = (index / POOLED_WIDTH / POOLED_HEIGHT) % INPUT_FEATURE_NUM;
    int n = index / POOLED_WIDTH / POOLED_HEIGHT / INPUT_FEATURE_NUM;

    const __global UNIT_TYPE* rois = input_rois + n * 5;
    int roi_batch_ind = rois[0];
    int roi_start_x = round(rois[1] * SPATIAL_SCALE);
    int roi_start_y = round(rois[2] * SPATIAL_SCALE);
    int roi_end_x = round(rois[3] * SPATIAL_SCALE);
    int roi_end_y = round(rois[4] * SPATIAL_SCALE);

    // Force malformed ROIs to be 1x1
    int roi_width = max(roi_end_x - roi_start_x + 1, 1);
    int roi_height = max(roi_end_y - roi_start_y + 1, 1);

    // The following computation of ystart, xstart, yend, xend is
    // done with integers due to floating precision errors.
    // As the floating point computing on GPU is not identical to CPU,
    // integer computing is used as a workaround.
    // The following approach also works but requires a rigorous analysis:
    // int ystart = (int)(floor(((float)ph * (float)(roi_height)) /
    //                           (float)(POOLED_HEIGHT)));
    // int xstart = (int)(floor(((float)pw * (float)(roi_width)) /
    //                           (float)(POOLED_WIDTH)));
    // int yend = (int)(ceil(((float)(ph + 1) * (float)(roi_height)) /
    //                        (float)(POOLED_HEIGHT)));
    // int xend = (int)(ceil(((float)(pw + 1) * (float)(roi_width)) /
    //                        (float)(POOLED_WIDTH)));

    int ystart = (ph * roi_height) / POOLED_HEIGHT;
    if ( (ystart * POOLED_HEIGHT) > (ph * roi_height) ) {
        --ystart;
    }
    int xstart = (pw * roi_width) / POOLED_WIDTH;
    if ( (xstart * POOLED_WIDTH) > (pw * roi_width) ) {
        --xstart;
    }
    int yend = ((ph + 1) * roi_height) / POOLED_HEIGHT;
    if ( (yend * POOLED_HEIGHT) < ((ph + 1) * roi_height) ) {
        ++yend;
    }
    int xend = ((pw + 1) * roi_width) / POOLED_WIDTH;
    if ( (xend * POOLED_WIDTH) < ((pw + 1) * roi_width) ) {
        ++xend;
    }

    ystart = min(max(ystart + roi_start_y, 0), INPUT_SIZE_Y);
    yend = min(max(yend + roi_start_y, 0), INPUT_SIZE_Y);
    xstart = min(max(xstart + roi_start_x, 0), INPUT_SIZE_X);
    xend = min(max(xend + roi_start_x, 0), INPUT_SIZE_X);
    // rounding to integer can lead to a zero sized box
    bool is_empty = (yend == ystart) || (xend == xstart);

    UNIT_TYPE maxval = is_empty ? 0 : -UNIT_INIT_VAL_MAX;
    int offset = (roi_batch_ind * INPUT_FEATURE_NUM + c) * INPUT_SIZE_Y * INPUT_SIZE_X;
    const __global UNIT_TYPE* input = input_data + offset + (INPUT_PADDING_LOWER_SIZE_Y * INPUT_SIZE_X) + INPUT_PADDING_LOWER_SIZE_X;

    for (int h = ystart; h < yend; ++h) {
        for (int w = xstart; w < xend; ++w) {
            int bottom_index = h * INPUT_SIZE_X + w;
            if (input[bottom_index] > maxval) {
                maxval = input[bottom_index];
            }
        }
    }

    output[index] = maxval;

    // TODO: currently not used in clCaffe
    //argmax_data[index] = index of the maximum value;
}