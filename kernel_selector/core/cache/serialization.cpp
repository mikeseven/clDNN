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
#include "serialization.h"

namespace KernelSelector { namespace gpu { namespace cache {

namespace {
#ifdef _WIN32
#include <windows.h>

uint64_t get_os_build_id()
{
    HMODULE hModule = NULL;
    if (!GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, (LPCSTR)get_os_build_id, &hModule) ||
        hModule == nullptr)
    {
        return 0;
    }

    PIMAGE_DOS_HEADER pDosHeader = (PIMAGE_DOS_HEADER)hModule;
    if (pDosHeader->e_magic != IMAGE_DOS_SIGNATURE)
    {
        return 0;
    }

    PIMAGE_NT_HEADERS pImageHeader = (PIMAGE_NT_HEADERS)((char*)pDosHeader + pDosHeader->e_lfanew);

    return uint64_t(pImageHeader->FileHeader.TimeDateStamp);
}

#else
// TODO Eli: in gcc we can use the elf build ID
uint64_t get_os_build_id()
{
    const uint64_t BUILD_NUMBER = 9999;

    // __TIME__ macro is always in form of HH:MM:SS.
    return ((__TIME__[7] - '0') +
            (__TIME__[6] - '0') * 10 +
            (__TIME__[4] - '0') * 60 +
            (__TIME__[3] - '0') * 600 +
            (__TIME__[1] - '0') * 3600) ^
            // moving value to the most significant WORD
            (uint64_t(BUILD_NUMBER) << (sizeof(uint64_t) - 2) * 8);
}
#endif

uint64_t get_build_id()
{
    // Unique value for current driver version
    static uint64_t s_build_id = 0;
    if (s_build_id == 0)
    {
        s_build_id = get_os_build_id();
    }
    return s_build_id;
}
}

namespace {

template <class T>
static void push_back_as_binary(binary_data& binary, const T& value)
{
    // we convert value to an array of binary data elements and insert that at the end of the binary vector
    binary.insert(binary.end(), reinterpret_cast<const binary_data::value_type *>(&value),
        reinterpret_cast<const binary_data::value_type *>((&value) + 1));
}

template <class It>
static void push_back_as_binary(binary_data& binary, It begin, It end)
{
    // we convert value to an array of binary data elements and insert that at the end of the binary vector
    binary.insert(binary.end(), reinterpret_cast<const binary_data::value_type *>(begin),
        reinterpret_cast<const binary_data::value_type *>(end));
}

}

binary_data serialization::serialize(const binary_cache& kernels)
{
    binary_data result;
    push_back_as_binary(result, get_build_id());
    auto count = kernels.size();
    // we store data in file in the following way: kernels count, (hash, binary_size, binary), (...)...
    push_back_as_binary(result, count);
    for (const auto& hk : kernels)
    {
        push_back_as_binary(result, hk.first);
        auto size = hk.second.size();
        push_back_as_binary(result, size);
        push_back_as_binary(result, &hk.second[0], &hk.second[0] + size);
    }
    return result;
}

namespace {

template <class T>
static T read_as_binary(const binary_data& data, size_t& offset)
{
    static_assert(sizeof(T) % sizeof(binary_data::value_type) == 0, "Size of T has to be divisible by size of binary data element");
    const size_t size_in_value_types = sizeof(T) / sizeof(binary_data::value_type);
    if (data.size() < (offset += size_in_value_types)) throw std::out_of_range("binary data ended unexpectedly");
    return *reinterpret_cast<const T*>(&data[offset - size_in_value_types]);
}

static binary_data read_as_binary(const binary_data& data, size_t& offset, size_t count)
{
    if (data.size() < offset + count) throw std::out_of_range("binary data ended unexpectedly");
    offset += count;
    return binary_data(data.begin() + offset - count, data.begin() + offset);
}

}

// we store data in file in the following way: kernels count, (hash, binary_size, binary), (...)...
binary_cache serialization::deserialize(const binary_data& data)
{
    binary_cache ret;

    if (data.size())
    {
        size_t offset = 0;
        uint64_t build_id = read_as_binary<uint64_t>(data, offset);

        if (build_id == get_build_id())
        {
            size_t count = read_as_binary<size_t>(data, offset);
            for (size_t i = 0; i < count; ++i)
            {
                size_t hash = read_as_binary<size_t>(data, offset);
                size_t size = read_as_binary<size_t>(data, offset);
                ret[hash] = read_as_binary(data, offset, size);
            }
        }
    }

    return ret;
}

} } }