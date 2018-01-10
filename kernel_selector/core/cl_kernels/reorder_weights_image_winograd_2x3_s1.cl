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

#include "include/include_all.cl"

KERNEL(reorder_weights_image_winograd_2x3_s1)(const __global INPUT0_TYPE* input, write_only image2d_t output)
{
    const uint input_tile_width = 1;
    const uint input_tile_height = 3;
    const uint in_tile_x_idx = get_global_id(1);
    const uint in_tile_y_idx = get_global_id(0);

    const uint output_tile_width = 4;
    const uint output_tile_height = 1;
    
    const uint tile_x_idx = get_global_id(0);
    const uint tile_y_idx = get_global_id(1);
    const uint feature_idx = get_global_id(2) % INPUT0_IFM_NUM;
    const uint batch_idx = get_global_id(2) / INPUT0_IFM_NUM;
    
    uint in_idx = batch_idx * INPUT0_OFM_PITCH
                 + feature_idx * INPUT0_IFM_PITCH
                 + in_tile_y_idx * input_tile_height * INPUT0_Y_PITCH
                 + in_tile_x_idx * input_tile_width * INPUT0_X_PITCH;

    MAKE_VECTOR_TYPE(INPUT0_TYPE, 4) tile;
    tile.x = input[in_idx]; in_idx += INPUT0_Y_PITCH;
    tile.y = input[in_idx]; in_idx += INPUT0_Y_PITCH;
    tile.z = input[in_idx];

    const uint weightsOSplit = 8;
    const uint oDivSplit = OUTPUT_OFM_NUM / 8;
    uint out_idx = batch_idx % 8 + tile_y_idx * output_tile_height * weightsOSplit +
        tile_x_idx * output_tile_width * weightsOSplit * OUTPUT_SIZE_Y +
        batch_idx / 8 * weightsOSplit * OUTPUT_SIZE_X * OUTPUT_SIZE_Y +
        feature_idx * weightsOSplit * OUTPUT_SIZE_X * OUTPUT_SIZE_Y * oDivSplit;

    output[out_idx] = TO_OUTPUT_TYPE(tile.x); out_idx += weightsOSplit * OUTPUT_SIZE_Y;
    output[out_idx] = TO_OUTPUT_TYPE((tile.x + tile.y + tile.z) / 2.0f); out_idx += weightsOSplit * OUTPUT_SIZE_Y;
    output[out_idx] = TO_OUTPUT_TYPE((tile.x - tile.y + tile.z) / 2.0f); out_idx += weightsOSplit * OUTPUT_SIZE_Y;
    output[out_idx] = TO_OUTPUT_TYPE(tile.z);
}
