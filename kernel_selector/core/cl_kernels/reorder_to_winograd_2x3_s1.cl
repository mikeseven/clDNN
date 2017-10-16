// Copyright (c) 2017 Intel Corporation
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

// --------------------------------------------------------------------------------------------------------------------------------
// Convert the signal using the forward F(2,3) Winograd transform.
// --------------------------------------------------------------------------------------------------------------------------------
void kernel KERNEL(reorder_to_winograd_2x3_s1)(global const UNIT_TYPE* input, global UNIT_TYPE* output_winograd)
{
    const uint input_tile_width = 4; //how much data is needed to produce one winograd tile (in x-dim)
    const uint input_tile_height = 1; //how much data is needed to produce on winograd tile (in y-dim)
    const uint input_tile_stride_x = 2; //how much do we need to proceed in input's x-dim to read data for new winograd tile
    const uint input_tile_stride_y = 1; //how much do we need to proceed in input's y-dim to read data for new winograd tile
    const uint winograd_tile_width = 4; //dimensions of resulting tile
    const uint winograd_tile_height = 1;
    
    const int batch_idx = get_global_id(0) / INPUT_FEATURE_NUM;
    const int feature_idx = get_global_id(0) % INPUT_FEATURE_NUM; //which feature do we process
    const int tile_idx_x = get_global_id(1); //which tile do we process (in x-dim)
    const int tile_idx_y = get_global_id(2); //which tile do we process (in y-dim)
    
    int in_idx = (INPUT_PADDING_LOWER_BATCH + batch_idx) * INPUT_PITCH_BATCH +
                  (INPUT_PADDING_LOWER_FEATURE + feature_idx) * INPUT_PITCH_FEATURE +
                  (INPUT_PADDING_LOWER_SIZE_Y + (tile_idx_y * input_tile_stride_y) + INPUT_OFFSET_SIZE_Y) * INPUT_PITCH_SIZE_Y+
                  (INPUT_PADDING_LOWER_SIZE_X + (tile_idx_x * input_tile_stride_x) + INPUT_OFFSET_SIZE_X) * INPUT_PITCH_SIZE_X;

    // storage for input tile
    UNIT_TYPE input_tile[input_tile_width * input_tile_height];
    
    // input tile is 4x1 so read 4 consecutive values in x-dim from input
    input_tile[0] = input[in_idx]; in_idx += INPUT_PITCH_SIZE_X;
    input_tile[1] = input[in_idx]; in_idx += INPUT_PITCH_SIZE_X;
    input_tile[2] = input[in_idx]; in_idx += INPUT_PITCH_SIZE_X;
    input_tile[3] = input[in_idx];
    
    // output is in byxf -- no paddings allowed in winograd domain
    int out_idx = batch_idx * OUTPUT_PITCH_BATCH +
                   feature_idx * OUTPUT_PITCH_FEATURE + 
                   (tile_idx_y * winograd_tile_height) * OUTPUT_PITCH_SIZE_Y +
                   (tile_idx_x * winograd_tile_width) * OUTPUT_PITCH_SIZE_X;

    //produce single 4x1 winograd tile ==> write 4 consecutive values in x-dim to output
    output_winograd[out_idx] = input_tile[0] - input_tile[2]; out_idx += OUTPUT_PITCH_SIZE_X;
    output_winograd[out_idx] = input_tile[1] + input_tile[2]; out_idx += OUTPUT_PITCH_SIZE_X;
    output_winograd[out_idx] = input_tile[2] - input_tile[1]; out_idx += OUTPUT_PITCH_SIZE_X;
    output_winograd[out_idx] = input_tile[1] - input_tile[3];
};
