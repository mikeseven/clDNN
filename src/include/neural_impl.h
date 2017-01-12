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
#include "api/engine.hpp"
#include "api/memory.hpp"
#include "implementation_map.h"
#include "primitive_arg.h"
#include "reorder_arg.h"
#include "convolution_arg.h"
#include "fully_connected_arg.h"
#include "activation_arg.h"
#include "depth_concatenate_arg.h"
#include "mean_substract_arg.h"
#include "eltwise_arg.h"
#include "normalization_arg.h"
#include "pooling_arg.h"
#include "softmax_arg.h"

namespace neural
{
    using memory = cldnn::neural_memory;
    template<typename primitive_kind> using implementation_map = cldnn::implementation_map<primitive_kind>;
    using is_an_implementation = cldnn::primitive_impl;
    using reorder = cldnn::reorder_arg;
    using convolution = cldnn::convolution_arg;
    using fully_connected = cldnn::fully_connected_arg;
    using relu = cldnn::activation_arg;
    using depth_concatenate = cldnn::depth_concatenate_arg;
    using mean_substract = cldnn::mean_substract_arg;
    using eltwise = cldnn::eltwise_arg;
    namespace normalization
    {
        using response = cldnn::normalization_arg;
        using softmax = cldnn::softmax_arg;
    }
    using pooling = cldnn::pooling_arg;

    template<typename T> using vector = cldnn::tensor;
}
