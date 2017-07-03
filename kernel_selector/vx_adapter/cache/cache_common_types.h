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
#pragma once

#include <string>

namespace clDNN
{
    namespace gpu 
    {
        struct context_device;
    }
}

namespace clDNN
{
    namespace gpu 
    {
        namespace cache 
        {
            using context_device = clDNN::gpu::context_device;
            using code = std::string;
            using compile_options = std::string;
            using primitive_id = std::string;
        }
    }
}