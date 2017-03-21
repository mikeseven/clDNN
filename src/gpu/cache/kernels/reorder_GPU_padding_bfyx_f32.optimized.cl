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


KERNEL (reorder_gpu_padding_bfyx_f32)(const __global SRC_TYPE* input, __global DEST_TYPE* output)
{
    const uint pos_b = get_global_id(0);
    const uint pos_f = get_global_id(1);
    const uint pos_y = get_global_id(2);

    if(pos_y >= INPUT_SIZE_Y)
        return;

    uint input_pos = pos_b * INPUT_SIZE_X * INPUT_SIZE_Y * INPUT_FEATURE_NUM;
    input_pos += pos_f * INPUT_SIZE_X * INPUT_SIZE_Y;
    input_pos += pos_y * INPUT_SIZE_X;

    uint output_pos = (pos_b * OUTPUT_SIZE_X * OUTPUT_SIZE_Y * OUTPUT_FEATURE_NUM);
    output_pos += pos_f * OUTPUT_SIZE_X * OUTPUT_SIZE_Y;
    output_pos += (pos_y + LOWER_PADDING[3]) * OUTPUT_SIZE_X;
    output_pos += LOWER_PADDING[2];
    for(uint x = 0; x < INPUT_SIZE_X; x++)
    {
        output[output_pos++] = input[input_pos++];
    }
}
