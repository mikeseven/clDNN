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

///////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef CLDNN_H
#define CLDNN_H

// exporting symbols form dynamic library
#ifdef EXPORT_NEURAL_SYMBOLS
#   if defined(_MSC_VER)
//  Microsoft
#      define CLDNN_API __declspec(dllexport)
#   elif defined(__GNUC__)
//  GCC
#      define CLDNN_API __attribute__((visibility("default")))
#   else
#      define CLDNN_API
#      pragma warning Unknown dynamic link import/export semantics.
#   endif
#else //import dll
#   if defined(_MSC_VER)
//  Microsoft
#      define CLDNN_API __declspec(dllimport)
#   elif defined(__GNUC__)
//  GCC
#      define CLDNN_API
#   else
#      define CLDNN_API
#      pragma warning Unknown dynamic link import/export semantics.
#   endif
#endif

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CLDNN_SUCCESS  0
#define CLDNN_ERROR   -1
#define CLDNN_UNSUPPORTED -2
#define CLDNN_INVALID_ARG -3

typedef int32_t cldnn_status;

typedef struct cldnn_engine_impl* cldnn_engine;
typedef struct cldnn_event_impl* cldnn_event;
typedef struct cldnn_topology_impl* cldnn_topology;
typedef struct cldnn_network_impl* cldnn_network;
typedef struct cldnn_memory_impl* cldnn_memory;

typedef enum /*:int32_t*/ { cldnn_engine_ocl } cldnn_engine_type;

typedef struct
{
    uint32_t enable_profiling;
    uint32_t meaningful_kernels_names;
    const char* compiler_options;
}  cldnn_engine_configuration;

typedef struct
{
    uint32_t cores_count;
    uint32_t core_frequency;

    uint64_t max_work_group_size;
    uint64_t max_local_mem_size;

    // Flags (for layout compatibility fixed size types are used).
    uint8_t supports_fp16;
    uint8_t supports_fp16_denorms;
}  cldnn_engine_info;

typedef void(*cldnn_event_handler)(void*);

typedef struct
{
    const char* name;
    uint64_t nanoseconds;
} cldnn_profiling_interval;

typedef enum /*:int32_t*/
{
    cldnn_build_option_fusing,
    cldnn_build_option_profiling,
    cldnn_build_option_optimize_data,
    cldnn_build_option_debug,
    cldnn_build_option_outputs,
} cldnn_build_option_type;

typedef struct
{
    int32_t type; /*cldnn_build_option_type*/
    const void* data;
}  cldnn_build_option;

typedef struct
{
    cldnn_event event;
    cldnn_memory memory;
} cldnn_network_output;


typedef enum /*:int32_t*/
{
    cldnn_format_x,
    cldnn_format_yx,
    cldnn_format_xy,
    cldnn_format_xb,          // 1D+batch, float32
    cldnn_format_bx,          // 1D+batch, float32
    cldnn_format_yxfn,          // 3D + number of neurons - used in fully connected weights
    cldnn_format_yxfb,          // 3D+batch, float32
    cldnn_format_byxf,          // for convolution_cpu_jit_batch1
    cldnn_format_bfyx,          // used in Caffe
    cldnn_format_fyxb,          // used in Caffe
    cldnn_format_oiyx,          // format used only for weights: o - output feature maps, i - input feature maps
    cldnn_format_yxoi,          // format used only for weights: o - output feature maps, i - input feature maps
    cldnn_format_oyxi,          // format used only for weights: o - output feature maps, i - input feature maps
    cldnn_format_yxio,          // format used only for weights: o - output feature maps, i - input feature maps
    cldnn_format_os_iyx_osv16,  // format used only for weights: os - output feature maps slice, i - input feature maps, yx - spatials, sv16 - 16 values of single slice
    cldnn_format_bs_xs_xsv8_bsv8, // format used only for Fully connected: bs - batch slice, xs - x slice, bsv8 - 8 values of single slice
    cldnn_format_bs_x_bsv16,    // format used only for fully connected: bs - batch slice (responses slice), bsv16 - 16 values of single batch slice, x - flattened plane of (fyx)
    cldnn_format_format_num,
    cldnn_format_any = -1
} cldnn_format_type;

#define CLDNN_FLOAT_TYPE_MASK 0x80

#define CLDNN_TENSOR_DIM_MAX 8

typedef struct
{
    int32_t format; /*cldnn_format_type*/
    size_t batch_num;
    size_t feature_num;
    size_t spatial_num;
    int32_t sizes[CLDNN_TENSOR_DIM_MAX];
} cldnn_tensor;

typedef enum /*:size_t*/
{
    cldnn_i8  = sizeof(int8_t),
    cldnn_i16 = sizeof(int16_t),
    cldnn_i32 = sizeof(int32_t),
    cldnn_i64 = sizeof(int64_t),
    cldnn_f16 = sizeof(int16_t) | CLDNN_FLOAT_TYPE_MASK,
    cldnn_f32 = sizeof(float) | CLDNN_FLOAT_TYPE_MASK,
    cldnn_f64 = sizeof(double) | CLDNN_FLOAT_TYPE_MASK,
} cldnn_data_type;

typedef struct
{
    size_t data_type; /*cldnn_data_type*/
    cldnn_tensor size;
} cldnn_layout;

typedef enum /*:int32_t*/
{
    cldnn_padding_zero,
    cldnn_padding_one,
    cldnn_padding_two,
} cldnn_padding_type;

typedef struct
{
    cldnn_tensor size;
    int32_t type; /*cldnn_padding_type*/
} cldnn_padding;

typedef struct
{
    const float* data;
    size_t size;
} cldnn_float_arr;

typedef const struct cldnn_primitive_type* cldnn_primitive_type_id;
typedef const char* cldnn_primitive_id;

typedef struct
{
    const cldnn_primitive_id* data;
    size_t size;
} cldnn_primitive_id_arr;

#define CLDNN_BEGIN_PRIMITIVE_DESC(PType) struct cldnn_##PType##_desc {\
    cldnn_primitive_type_id type;\
    cldnn_primitive_id id;\
    cldnn_primitive_id_arr input;\
    cldnn_padding input_padding;\
    cldnn_padding output_padding;

#define CLDNN_END_PRIMITIVE_DESC(PType) };

#define CLDNN_PRIMITIVE_DESC(PType) cldnn_##PType##_desc

CLDNN_BEGIN_PRIMITIVE_DESC(primitive)
CLDNN_END_PRIMITIVE_DESC(primitive)

// topology
CLDNN_API cldnn_topology cldnn_create_topology(cldnn_status* status);
CLDNN_API void cldnn_add_primitive(cldnn_topology topology, const CLDNN_PRIMITIVE_DESC(primitive)* dto, cldnn_status* status);
CLDNN_API void cldnn_retain_topology(cldnn_topology topology, cldnn_status* status);
CLDNN_API void cldnn_release_topology(cldnn_topology topology, cldnn_status* status);


// engine
/**
 * \brief number of available engines of the particular type
 */
CLDNN_API uint32_t cldnn_get_engine_count(/*cldnn_engine_type*/ int32_t type, cldnn_status* status);
CLDNN_API cldnn_engine cldnn_create_engine(/*cldnn_engine_type*/ int32_t type, uint32_t engine_num, const cldnn_engine_configuration* configuration, cldnn_status* status);
CLDNN_API void cldnn_retain_engine(cldnn_engine engine, cldnn_status* status);
CLDNN_API void cldnn_release_engine(cldnn_engine engine, cldnn_status* status);
CLDNN_API cldnn_engine_info cldnn_get_engine_info(cldnn_engine engine, cldnn_status* status);
CLDNN_API /*cldnn_engine_type*/ int32_t cldnn_get_engine_type(cldnn_engine engine, cldnn_status* status);

// event
CLDNN_API cldnn_event cldnn_create_user_event(cldnn_engine engine, cldnn_status* status);
CLDNN_API void cldnn_retain_event(cldnn_event event, cldnn_status* status);
CLDNN_API void cldnn_release_event(cldnn_event event, cldnn_status* status);
CLDNN_API void cldnn_wait_for_event(cldnn_event event, cldnn_status* status);
CLDNN_API void cldnn_set_event(cldnn_event event, cldnn_status* status);
CLDNN_API void cldnn_add_event_handler(cldnn_event event, cldnn_event_handler handler, void* param, cldnn_status* status);
CLDNN_API void cldnn_get_event_profiling_info(cldnn_event event, cldnn_profiling_interval* profiling, size_t size, size_t* size_ret, cldnn_status* status);

// network
CLDNN_API        cldnn_network cldnn_build_network(cldnn_engine engine, cldnn_topology topology, cldnn_build_option* options, size_t options_num, cldnn_status* status);
CLDNN_API                 void cldnn_retain_network(cldnn_network network, cldnn_status* status);
CLDNN_API                 void cldnn_release_network(cldnn_network network, cldnn_status* status);
CLDNN_API                 void cldnn_set_network_input(cldnn_network network, cldnn_primitive_id id, cldnn_memory mem, cldnn_status* status);
CLDNN_API         cldnn_engine cldnn_get_network_engine(cldnn_network network, cldnn_status* status);
CLDNN_API       cldnn_topology cldnn_get_network_topology(cldnn_network network, cldnn_status* status);
CLDNN_API                 void cldnn_get_network_output_names(cldnn_network network, char* names, size_t size, size_t* size_ret, cldnn_status* status);
CLDNN_API                 void cldnn_execute_network(cldnn_network network, cldnn_event* dependencies, size_t deps_num, cldnn_status* status);
CLDNN_API cldnn_network_output cldnn_get_network_output(cldnn_network network, const char* name, cldnn_status* status);

// memory
CLDNN_API cldnn_memory cldnn_allocate_memory(cldnn_engine engine, cldnn_layout layout, cldnn_status* status);
CLDNN_API cldnn_memory cldnn_attach_memory(cldnn_layout layout, void* pointer, size_t size, cldnn_status* status);
CLDNN_API void cldnn_retain_memory(cldnn_memory memory, cldnn_status* status);
CLDNN_API void cldnn_release_memory(cldnn_memory memory, cldnn_status* status);
CLDNN_API void* cldnn_lock_memory(cldnn_memory memory, cldnn_status* status);
CLDNN_API void cldnn_unlock_memory(cldnn_memory memory, cldnn_status* status);
CLDNN_API cldnn_layout cldnn_get_memory_layout(cldnn_memory memory, cldnn_status* status);
CLDNN_API cldnn_engine cldnn_get_memory_engine(cldnn_memory memory, cldnn_status* status);


#ifdef __cplusplus
}
#endif

//primitives
#ifdef __cplusplus
#define CLDNN_DECLARE_PRIMITIVE_TYPE_ID(PType) extern "C" CLDNN_API cldnn_primitive_type_id cldnn_##PType##_type_id(cldnn_status* status)
#else
#define CLDNN_DECLARE_PRIMITIVE_TYPE_ID(PType) CLDNN_API cldnn_primitive_type_id cldnn_##PType##_type_id(cldnn_status* status)
#endif


#endif /* CLDNN_H */
