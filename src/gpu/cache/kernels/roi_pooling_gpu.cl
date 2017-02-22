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
    int c = (index / POOLED_WIDTH / POOLED_HEIGHT) % CHANNELS;
    int n = index / POOLED_WIDTH / POOLED_HEIGHT / CHANNELS;

    __global UNIT_TYPE* rois = input_rois + n * 5;
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

    ystart = min(max(ystart + roi_start_y, 0), HEIGHT);
    yend = min(max(yend + roi_start_y, 0), HEIGHT);
    xstart = min(max(xstart + roi_start_x, 0), WIDTH);
    xend = min(max(xend + roi_start_x, 0), WIDTH);
    // rounding to integer can lead to a zero sized box
    bool is_empty = (yend == ystart) || (xend == xstart);

    UNIT_TYPE maxval = is_empty ? 0 : -UNIT_INIT_VAL_MAX;
    int offset = (roi_batch_ind * CHANNELS + c) * HEIGHT * WIDTH;
    __global UNIT_TYPE* input = input_data + offset + (INPUT_PADDING_SIZE_Y * WIDTH) + INPUT_PADDING_SIZE_X;
    
    for (int h = ystart; h < yend; ++h) {
        for (int w = xstart; w < xend; ++w) {
            int bottom_index = h * WIDTH + w;
            if (input[bottom_index] > maxval) {                
                maxval = input[bottom_index];
            }
        }
    }

    output[index] = maxval;
    
    // TODO: currently not used in clCaffe
    //argmax_data[index] = index of the maximum value;
}