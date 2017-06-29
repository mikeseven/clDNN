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
#include "persistent_cache.h"
#include <fstream>
#include <sstream>
#include <system_error>

namespace KernelSelector { namespace gpu { namespace cache {

persistent_cache::persistent_cache(const char* cache_file_name) : file(cache_file_name) { }

binary_data persistent_cache::get() { return file.read(); }

void persistent_cache::set(binary_data data) { file.write(data); }

persistent_cache::cache_file::cache_file(const char* file_name) : cache_file_name(file_name) { }


binary_data persistent_cache::cache_file::read()
{
#ifdef CLDNN_ENABLE_PERSISTENT_CACHE
    std::ifstream c_file(cache_file_name, std::ios::binary);
    if (c_file.is_open())
    {
        std::stringstream data;
        data << c_file.rdbuf();
        c_file.close();
        const auto& str = data.str();
        return std::vector<unsigned char>(str.begin(), str.end());
    }
#endif
    return binary_data();
}

#ifdef CLDNN_ENABLE_PERSISTENT_CACHE
void persistent_cache::cache_file::write(const binary_data& data)
{
    std::ofstream c_file(cache_file_name, std::ios::binary);
    if (c_file.is_open())
    {
        std::string str(data.begin(), data.end());
        c_file << str;
        c_file.close();
        return;
    }
    throw std::system_error(errno, std::system_category( ));
}
#else
void persistent_cache::cache_file::write(const binary_data&)
{
}
#endif
} } }