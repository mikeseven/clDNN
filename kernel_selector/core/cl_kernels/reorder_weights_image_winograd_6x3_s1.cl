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

KERNEL(reorder_weights_image_winograd_6x3_s1)(const __global INPUT0_TYPE* input, write_only image2d_t output)
{
	const uint input_tile_width = 1;
	const uint input_tile_height = 3;
	const uint in_tile_x_idx = get_global_id(1);
	const uint in_tile_y_idx = get_global_id(0);

	const uint output_tile_width = 8;
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

	const uint weightsOSplit = 16;
	const uint oDivSplit = OUTPUT_OFM_NUM / 16;

	const uint ySize = OUTPUT_OFM_NUM * OUTPUT_SIZE_X * OUTPUT_SIZE_Y;
	uint idx = batch_idx % 16 + tile_y_idx * output_tile_height * weightsOSplit +
		tile_x_idx * output_tile_width * weightsOSplit * OUTPUT_SIZE_Y +
		batch_idx / 16 * weightsOSplit * OUTPUT_SIZE_X * OUTPUT_SIZE_Y +
	    feature_idx * ySize;
	uint idx_x = idx%ySize;
	uint idx_y = idx/ySize;

	write_imagef(output, (int2)(idx_x, idx_y), TO_OUTPUT_TYPE(+90.0 / 90 * tile.x)); idx_x += weightsOSplit * OUTPUT_SIZE_Y; //if (idx_x >= ySize) { idx_x = idx_x % ySize; idx_y++; }
	write_imagef(output, (int2)(idx_x, idx_y), TO_OUTPUT_TYPE(-20.0 / 90 * tile.x - 20.0 / 90 * tile.y - 20.0 / 90 * tile.z)); idx_x += weightsOSplit * OUTPUT_SIZE_Y;  //if (idx_x >= ySize) { idx_x = idx_x % ySize; idx_y++; }
	write_imagef(output, (int2)(idx_x, idx_y), TO_OUTPUT_TYPE(-20.0 / 90 * tile.x + 20.0 / 90 * tile.y - 20.0 / 90 * tile.z)); idx_x += weightsOSplit * OUTPUT_SIZE_Y;  //if (idx_x >= ySize) { idx_x = idx_x % ySize; idx_y++; }
	write_imagef(output, (int2)(idx_x, idx_y), TO_OUTPUT_TYPE(+1.0 / 90 * tile.x + 2.0 / 90 * tile.y + 4.0 / 90 * tile.z)); idx_x += weightsOSplit * OUTPUT_SIZE_Y;  //if (idx_x >= ySize) { idx_x = idx_x % ySize; idx_y++; }
	write_imagef(output, (int2)(idx_x, idx_y), TO_OUTPUT_TYPE(+1.0 / 90 * tile.x - 2.0 / 90 * tile.y + 4.0 / 90 * tile.z)); idx_x += weightsOSplit * OUTPUT_SIZE_Y;  //if (idx_x >= ySize) { idx_x = idx_x % ySize; idx_y++; }
	write_imagef(output, (int2)(idx_x, idx_y), TO_OUTPUT_TYPE(+64.0 / 90 * tile.x + 32.0 / 90 * tile.y + 16.0 / 90 * tile.z)); idx_x += weightsOSplit * OUTPUT_SIZE_Y;  //if (idx_x >= ySize) { idx_x = idx_x % ySize; idx_y++; }
	write_imagef(output, (int2)(idx_x, idx_y), TO_OUTPUT_TYPE(+64.0 / 90 * tile.x - 32.0 / 90 * tile.y + 16.0 / 90 * tile.z)); idx_x += weightsOSplit * OUTPUT_SIZE_Y;  //if (idx_x >= ySize) { idx_x = idx_x % ySize; idx_y++; }
	write_imagef(output, (int2)(idx_x, idx_y), TO_OUTPUT_TYPE(+90.0 / 90 * tile.z)); 
	
	/*output[out_idx] = TO_OUTPUT_TYPE(+90.0 / 90 * tile.x); out_idx += weightsOSplit * OUTPUT_SIZE_Y;
	output[out_idx] = TO_OUTPUT_TYPE(-20.0 / 90 * tile.x - 20.0 / 90 * tile.y - 20.0 / 90 * tile.z); out_idx += weightsOSplit * OUTPUT_SIZE_Y;
	output[out_idx] = TO_OUTPUT_TYPE(-20.0 / 90 * tile.x + 20.0 / 90 * tile.y - 20.0 / 90 * tile.z); out_idx += weightsOSplit * OUTPUT_SIZE_Y;
	output[out_idx] = TO_OUTPUT_TYPE(+1.0 / 90 * tile.x + 2.0 / 90 * tile.y + 4.0 / 90 * tile.z); out_idx += weightsOSplit * OUTPUT_SIZE_Y;
	output[out_idx] = TO_OUTPUT_TYPE(+1.0 / 90 * tile.x - 2.0 / 90 * tile.y + 4.0 / 90 * tile.z); out_idx += weightsOSplit * OUTPUT_SIZE_Y;
	output[out_idx] = TO_OUTPUT_TYPE(+64.0 / 90 * tile.x + 32.0 / 90 * tile.y + 16.0 / 90 * tile.z); out_idx += weightsOSplit * OUTPUT_SIZE_Y;
	output[out_idx] = TO_OUTPUT_TYPE(+64.0 / 90 * tile.x - 32.0 / 90 * tile.y + 16.0 / 90 * tile.z); out_idx += weightsOSplit * OUTPUT_SIZE_Y;
	output[out_idx] = TO_OUTPUT_TYPE(+90.0 / 90 * tile.z);*/
}
