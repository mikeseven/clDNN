/*
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
*/

#include <api\cldnn.h>
#include <api\cldnn_defs.h>
#include <api\engine.hpp>
#include <api\memory.hpp>



void main()
{
    // To create memory we have to create engine first. Engine is responsible for creation and handle allocations on choosen backend.
    // In current implementation only OCL backend is avaiable.
    cldnn::engine engine;
    // we have to choose data type (f32 or f16):
    auto data_type = cldnn::data_types::f32;
    // and format (order of dimensions in memory), bfyx is the most optimal and common:
    auto format = cldnn::format::byxf;

    // before memory alocation we have to create tensor that describes memory. We can do it 


}