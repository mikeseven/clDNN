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
#ifndef INPUT_LAYOUT_H
#define INPUT_LAYOUT_H

#include "api/cldnn.h"

#ifdef __cplusplus
extern "C" {
#endif

CLDNN_BEGIN_PRIMITIVE_DESC(input_layout)
cldnn_layout layout;
CLDNN_END_PRIMITIVE_DESC(input_layout)

CLDNN_DECLARE_PRIMITIVE_TYPE_ID(input_layout);

#ifdef __cplusplus
}
#endif

#endif /* INPUT_LAYOUT_H */

