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
#pragma once
#include <stdint.h>


#ifdef __cplusplus
extern "C" {
#endif

#define CLDNN_API_ENTRY
#define CLDNN_API_CALL

typedef struct _cldnn_context_t* cldnn_context_t;
typedef struct _cldnn_engine_t* cldnn_engine_t;
typedef struct _cldnn_network_t* cldnn_network_t;
typedef struct _cldnn_primitive_t* cldnn_primitive_t;
typedef struct _cldnn_memory_t* cldnn_memory_t;
typedef struct _cldnn_event_t* cldnn_event_t;

typedef uint32_t cldnn_status_t;

enum cldnn_engine_type
{
    engine_type_ocl
};

extern CLDNN_API_ENTRY cldnn_context_t CLDNN_API_CALL cldnnCreateContext();
extern CLDNN_API_ENTRY cldnn_engine_t CLDNN_API_CALL cldnnCreateEngine(cldnn_engine_type type);
extern CLDNN_API_ENTRY cldnn_network_t CLDNN_API_CALL cldnnCreateNerwork(cldnn_context_t context);

extern CLDNN_API_ENTRY cldnn_primitive_t CLDNN_API_CALL cldnnAddPrimitive(cldnn_network_t network, cldnn_primitive_def_t primitive_definition);
extern CLDNN_API_ENTRY void CLDNN_API_CALL cldnnAddNetworkInput(cldnn_network_t network, cldnn_data_layout_t layout);
extern CLDNN_API_ENTRY cldnn_memory_t CLDNN_API_CALL cldnnGetOutput(cldnn_network_t network, cldnn_primitive_t primitive);

#ifdef __cplusplus
} //extern "C"
#endif
