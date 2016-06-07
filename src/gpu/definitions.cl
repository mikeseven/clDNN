/*
// Copyright (c) 2016 Intel Corporation
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
*/

#ifdef CODE_PREFIX
#define CODE_BEGIN CODE_PREFIX
#define CODE_END CODE_POSTFIX
#else
#define CODE_BEGIN
#define CODE_END
#endif

CODE_BEGIN
enum neural_memory_format {
    x_f32,
    xb_f32,     // 1D+batch, float32
    bx_f32,     // 1D+batch, float32
    yxfb_f32,   // 3D+batch, float32
    byxf_f32,   // for convolution_cpu_jit_batch1
    bfyx_f32,   // used in Caffe
    fyxb_f32,   // used in Caffe
    oiyx_f32,   // format used only for weights: o - output feature maps, i - input feature maps
    byxf_b24_f32,        // for convolution_cpu_generic
    yxoi_o4_f32,       // for convolution_cpu_generic
    os_yxi_sv16_f32,   // format used only for weights: os - output slice, i - input feature maps, sv16 - 16 values of single slice
    bs_yxf_bv24_f32,
    any=-1
};

#pragma pack(push, 1)
typedef struct _neural_memory_tag {
    uint format;
    uint feature_offset;
    uint spatial_offset;
    uint data_offset;
    uint data[1];
} neural_memory;
#pragma pack(pop)

__global uint* get_raw(__global neural_memory* mem) { return &(mem->data[0]); }
uint get_raw_size(__global neural_memory* mem) { return mem->data_offset; } 

__global uint* get_batch(__global neural_memory* mem) { return get_raw(mem); }
uint get_batch_size(__global neural_memory* mem) { return mem->feature_offset; }

__global uint* get_feature(__global neural_memory* mem) { return &(mem->data[mem->feature_offset]); }
uint get_feature_size(__global neural_memory* mem) { return mem->spatial_offset - mem->feature_offset; }

__global uint* get_spatial(__global neural_memory* mem) { return &(mem->data[mem->spatial_offset]); }
uint get_spatial_size(__global neural_memory* mem) { return mem->data_offset - mem->spatial_offset; } 

__global void* get_data(__global neural_memory* mem) { return &(mem->data[mem->data_offset]); }
size_t get_element_size(__global neural_memory* mem) { return sizeof(float); }

size_t get_data_size(__global neural_memory* mem) {
    size_t result = get_element_size(mem);

    __global uint* raw = get_raw(mem);
    uint raw_size = get_raw_size(mem);

    for(uint i = 0; i < raw_size; i++) {
        result *= raw[i];
    }
    return result;
}

CODE_END

CODE_BEGIN
__kernel void Fully_Connected_GPU(__global neural_memory* input_mem, __global neural_memory* weights_mem, __global neural_memory* bias_mem, __global neural_memory* dst_mem)
{
    __global uint* input_size = get_raw(input_mem);
    __global uint* weights_size = get_raw(weights_mem);
    __global float* input = (__global float*)get_data(input_mem);
    __global float* weights = (__global float*)get_data(weights_mem);
    __global float* bias = (__global float*)get_data(bias_mem);
    __global float* pDst = (__global float*)get_data(dst_mem);

    const int x = get_global_id(0);

    pDst[x] = 0;
    uint outXIdx = x / input_size[0];
    uint inputBatchIdx = x % input_size[0];
    uint weightYIdx = outXIdx * weights_size[0];
    for (uint i = 0; i < input_size[2]; i++)
    {
        pDst[x] += input[i * input_size[0] + inputBatchIdx] * weights[weightYIdx + i];
    }
    pDst[x] += bias[outXIdx];
}
CODE_END