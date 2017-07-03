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

#include "program_cache.h"
#include "vxa_ocl_toolkit.h"

namespace clDNN { namespace gpu { namespace cache {

using context_device = clDNN::gpu::context_device;

static const char* cache_file_name = "cl_dnn_cache.intel"; //TODO building name

program_cache::program_cache() : file_cache(cache_file_name), program_binaries(serialization::deserialize(file_cache.get())) { }

program_cache::~program_cache()
{
    if (dirty)
    {
        file_cache.set(serialization::serialize(program_binaries));
    }
}

binary_data program_cache::get(context_device context, const code& program_str, const compile_options& options)
{
    std::hash<std::string> h;
    size_t hash = h(program_str + options);
    auto it = program_binaries.find(hash);

    binary_data binary;

    if (it == program_binaries.end())
    {
        dirty = true;
        binary = gpu_compiler::compile(context, program_str, options);
        program_binaries[hash] = binary;

    }
    else
    {
        binary = it->second;
    }

    return binary;
}

} } }