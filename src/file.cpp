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

#include "api/neural.h"
#include <fstream>
#include <nmmintrin.h>
#include <array>

namespace neural {

namespace {
const primitive null_primitive(nullptr);

uint32_t crc32(const void *buffer, size_t count, uint32_t crc) {
    const uint8_t *ptr = static_cast<const uint8_t *>(buffer);
    for(; count>=4; count-=4, ptr+=4)
        crc = _mm_crc32_u32( crc,*reinterpret_cast<const uint32_t *>(ptr) );
    while(count--) crc = _mm_crc32_u32(crc, *(ptr++));
    return crc;
}

#pragma pack(push,1)   /* The data has been redefined (alignment 4), so the pragma pack is not necessary,
                          but who knows if in the future, the compiler does not align to 8?  */
struct file_header {
    std::array<uint8_t,3>           magic;
    uint8_t                         version;
    neural::memory::format::type    format;
    uint8_t                         rank_minus_1;
};

#pragma pack(pop)

// missing in C++11
template<typename T_type, typename... T_args> std::unique_ptr<T_type> make_unique(T_args&&... args) {
    return std::unique_ptr<T_type>(new T_type(std::forward<T_args>(args)...));
}

} //namespace {


file::arguments::arguments(neural::engine::type aengine, std::string aname, memory::format::type aformat, std::vector<uint32_t> &asize)
    : engine(aengine)
    , name(aname)
    , output{{memory::create({aengine, aformat, asize, true})}} {}

file::arguments::arguments(neural::engine::type aengine, std::string aname, memory::format::type aformat)
    : engine(aengine)
    , name(aname)
    , output{{memory::create({aengine, aformat, std::vector<uint32_t>()})}} {}

file::arguments::arguments(neural::engine::type aengine, std::string aname, primitive aoutput)
    : engine(aengine)
    , name(aname)
    , output{aoutput} {}

file::arguments::arguments(neural::engine::type aengine, std::string aname)
    : engine(aengine)
    , name(aname)
    , output({null_primitive}) {};


// creates primitive with memry buffer loaded from specified file
primitive file::create(file::arguments arg) {
    try {
        std::ifstream in;
        in.exceptions(std::ios::failbit | std::ios::badbit);
        in.open(arg.name, std::ios::in | std::ios::binary);

        auto read_crc = [&in]() -> uint32_t {
            uint32_t result;
            in.read(reinterpret_cast<char *>(&result), sizeof(uint32_t));
            return result;
        };

        // load header, verify 32-bit crc, validate
        const uint32_t crc_seed = 0xdeadf00d;
        file_header header;
        in.read(reinterpret_cast<char *>(&header), sizeof(header));
        if(read_crc()!=crc32(&header, sizeof(header), crc_seed))    throw std::runtime_error("file::create: header crc mismatch");
        if(header.magic!=std::array<uint8_t,3>{{'n','I','A'}})      throw std::runtime_error("file::create: bad format");
        if(header.version!=0)                                       throw std::runtime_error("file::create: bad format version");

        // load vector of size_t sizes (one per each dimension)
        auto size = make_unique<std::vector<uint32_t>>(header.rank_minus_1+1);
        in.read(reinterpret_cast<char *>(size->data()), size->size()*sizeof(uint32_t));
        if(read_crc()!=crc32(&header, sizeof(header), crc_seed))    throw std::runtime_error("file::create: sizes crc mismatch");

        // validation
        if(arg.output[0]!=null_primitive) {
            auto output = arg.output[0].as<const memory&>().argument;
            if(header.format!=output.format)                            throw std::runtime_error("file::create: format in file different than requested");
            if(0 != output.size.raw.size() && *size != output.size.raw) throw std::runtime_error("file::create: size/dimensionality of data is different than requested");
        }

        // crete result with output owning memory, load data into it
        auto result = std::unique_ptr<file>(new file({arg.engine, arg.name, memory::create({arg.engine, header.format, *size, true})}));

            //todo tmp solution
        auto &buffer = result->argument.output[0].as<const memory&>();
        //auto &buffer = result->output_memory(0);
        auto count = buffer.count()*memory::traits(buffer.argument.format).type->size;
        in.read(static_cast<char *>(buffer.pointer), count);

        return result.release();
    }
    catch(...) {
        throw std::runtime_error("file::create: error loading file");
    }
}

}