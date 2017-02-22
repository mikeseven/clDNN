/*
//
//                 INTEL CORPORATION PROPRIETARY INFORMATION
//    This software is supplied under the terms of a license agreement or
//    nondisclosure agreement with Intel Corporation and may not be copied
//    or disclosed except in accordance with the terms of that agreement.
//        Copyright(c) 2017 Intel Corporation. All Rights Reserved.
*/

#if FP16_SUPPORTED
    #pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

__kernel void roi_pooling_gpu
(
    const __global UNIT_TYPE * src_data,
    const __global UNIT_TYPE * src_rois,
    __global UNIT_TYPE * dst_data
)
{
    const size_t i = get_global_id(0);

    const int x = i % DST_W;
    const int y = i / DST_W % DST_H;
    const int c = i / DST_W / DST_H % CHAN_NUM;
    const int r = i / DST_W / DST_H / CHAN_NUM % ROIS_NUM;
    const int b = i / DST_W / DST_H / CHAN_NUM / ROIS_NUM;

    const __global UNIT_TYPE * roi_ptr = &src_rois[PITCH_ROI_R * r + PITCH_ROI_B * b];

    // TODO: check issue with rounding mode
//    const int roi_x  = convert_int(roi_ptr[0]);
//    const int roi_y  = convert_int(roi_ptr[1]);
//    const int roi_x1 = convert_int(roi_ptr[2]);
//    const int roi_y1 = convert_int(roi_ptr[3]);
    const int roi_x  = (int)round(roi_ptr[0]);
    const int roi_y  = (int)round(roi_ptr[1]);
    const int roi_x1 = (int)round(roi_ptr[2]);
    const int roi_y1 = (int)round(roi_ptr[3]);

    // The final coordinate is within the ROI and malformed dimensions are treated as 1
    const int roi_w = max(roi_x1 - roi_x, 0) + 1;
    const int roi_h = max(roi_y1 - roi_y, 0) + 1;

    // Note that when the "after" is rounded rounded up else we get the last cell,
    // instead of the cell beyond.
    //
    // For ex. with src being a 6 cell row and dest being a 4 cell one:
    // >>> [((x + 0) * 6) // 4 for x in [0, 1, 2, 3]]   # "begin" values
    // [0, 1, 3, 4]                                     # as expected
    // >>> [((x + 1) * 6) // 4 for x in [0, 1, 2, 3]]   # "after" values
    // [1, 3, 4 ,6]                                     # [2, 3, 5, 6] expected!
    const int dx_begin = ((x + 0) * roi_w) / DST_W;
    const int dy_begin = ((y + 0) * roi_h) / DST_H;
    const int dx_after = ((x + 1) * roi_w + (DST_W - 1)) / DST_W;
    const int dy_after = ((y + 1) * roi_h + (DST_H - 1)) / DST_H;

    // clamp in case roi_x or roi_y were unreasonable
    const int x_begin = clamp(roi_x + dx_begin, 0, SRC_W);
    const int y_begin = clamp(roi_y + dy_begin, 0, SRC_H);
    const int x_after = clamp(roi_x + dx_after, 0, SRC_W);
    const int y_after = clamp(roi_y + dy_after, 0, SRC_H);

    const __global UNIT_TYPE * data = src_data + PITCH_SRC_C * c + PITCH_SRC_B * b;
 
    UNIT_TYPE res = (x_begin < x_after && y_begin < y_after) ? -INFINITY : 0;

    for (int yy = y_begin; yy < y_after; ++yy)
    for (int xx = x_begin; xx < x_after; ++xx)
        res = max(res, data[xx + PITCH_SRC_H * yy]);

    dst_data[x + PITCH_DST_H * y + PITCH_DST_C * c + PITCH_DST_R * r + PITCH_DST_B * b] = res;
}