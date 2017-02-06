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
#include <utility>
#include <unordered_map>
#include "serialization.h"
#include "persistent_cache.h"
#include "gpu_compiler.h"

namespace neural { namespace gpu { namespace cache 
{
/// \brief Class that provides transparent cache/compiler interface for collecting compilation results 
///
class kernel_cache
{
public:
    kernel_cache();
    ~kernel_cache();

    binary_data get(context* context, const jit & jit, const code& kernel_str, const compile_options& options);

private:
    persistent_cache file_cache;
    binary_cache kernel_binaries;
    bool dirty = false;
};

} } }