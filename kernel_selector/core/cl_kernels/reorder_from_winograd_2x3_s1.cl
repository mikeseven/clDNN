/*/// -----------------------------------------------------------------------------------------------------------------------------
Copyright (c) 2016, Intel Corporation
/*/// -----------------------------------------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------------------------------------------
// Convert the results using the inverse F(2,3) Winograd transform.
// --------------------------------------------------------------------------------------------------------------------------------

void kernel KERNEL(reorder_from_winograd_2x3_s1)(global const UNIT_TYPE* input_winograd, global float* output)
{
    const int winograd_tile_width = 4;
    const int winograd_tile_height = 1;
    const int output_tile_width = 2;
    const int output_tile_height = 1;
    
    const int batch_idx = get_global_id(0) / INPUT_FEATURE_NUM;
    const int feature_idx = get_global_id(0) % INPUT_FEATURE_NUM;
    const int tile_idx_x = get_global_id(1);
    const int tile_idx_y = get_global_id(2);

    const int out_x_idx = (tile_idx_x * output_tile_width);
    
    //input is in bxyf -- no paddings allowed in winograd domain
    int input_idx = batch_idx * INPUT_PITCH_BATCH +
                    feature_idx * INPUT_PITCH_FEATURE +
                    tile_idx_y * winograd_tile_height * INPUT_PITCH_SIZE_Y +
                    tile_idx_x * winograd_tile_width * INPUT_PITCH_SIZE_X;

    //winograd tile is 4x1, during conversion to standard domain values should have already been multiplied so this tile is actually an 'm' tile from the original paper
    UNIT_TYPE winograd_tile[winograd_tile_width];
    winograd_tile[0] = input_winograd[input_idx]; input_idx += INPUT_PITCH_SIZE_X;
    winograd_tile[1] = input_winograd[input_idx]; input_idx += INPUT_PITCH_SIZE_X;
    winograd_tile[2] = input_winograd[input_idx]; input_idx += INPUT_PITCH_SIZE_X;
    winograd_tile[3] = input_winograd[input_idx];

    UNIT_TYPE out_tile[output_tile_width];

    //transform back
#ifndef ACTIVATION
    out_tile[0] = winograd_tile[0] + winograd_tile[1] + winograd_tile[2];
    out_tile[1] = winograd_tile[1] - winograd_tile[2] - winograd_tile[3];
#else
    out_tile[0] = ACTIVATION(winograd_tile[0] + winograd_tile[1] + winograd_tile[2]);
    out_tile[1] = ACTIVATION(winograd_tile[1] - winograd_tile[2] - winograd_tile[3]);
#endif

    int out_idx = (OUTPUT_PADDING_LOWER_BATCH + batch_idx) * OUTPUT_PITCH_BATCH +
                  (OUTPUT_PADDING_LOWER_FEATURE + feature_idx) * OUTPUT_PITCH_FEATURE +
                  (OUTPUT_PADDING_LOWER_SIZE_Y + (tile_idx_y * output_tile_height)) * OUTPUT_PITCH_SIZE_Y +
                  (OUTPUT_PADDING_LOWER_SIZE_X + (tile_idx_x * output_tile_width)) * OUTPUT_PITCH_SIZE_X;

    output[out_idx] = out_tile[0];
#ifdef LEFTOVERS
    if (out_x_idx + 1 < OUTPUT_SIZE_X)
#endif
    {
        out_idx += OUTPUT_PITCH_SIZE_X;
        output[out_idx] = out_tile[1];
    }
};
