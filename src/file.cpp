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
#define CRC_INIT 0xbaba7007

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
    // Data header                   // Size [B]
    uint8_t    magic[3];             // 3          |
    uint8_t    data_type;            // 1           } aligment 2x4B
    uint8_t    version;              // 1          |
    uint8_t    dimension;            // 1
    uint8_t    sizeof_value;         // 1
};

struct nn_data{
#if defined __cplusplus
    nn_data() : buffer(nullptr), size(nullptr), dimension(0), sizeof_value(0) {};
#endif
void           *const buffer;       /* buffer containig data */
const size_t   *const size;         /* sizes of signal in each coordinate; unit is a value */
const uint8_t         dimension;    /* dimensionality of data, as in http://en.wikipedia.org/wiki/Dimension */
const uint8_t         sizeof_value; /* size of single value in buffer */
};

/* calculate size of buffer, sizes in array
examples:
size_t sizes[3] = {2, 3, 4};
nn_data_buffer_size_ptr(sizeof(float), sizeof(sizes)/sizeof(sizes[0]), sizes);
Buffer for 3-dimensional grid of floats with size [2,3,4] has size of 96 bytes.
size_t sizes[2] = {320, 240};
nn_data_buffer_size_ptr(sizeof(float), sizeof(sizes)/sizeof(sizes[0]), sizes);
Buffer for 2-dimensional grid of bytes with size [320,240] has size od 768000.
*/
static inline size_t nn_data_buffer_size_ptr(
    size_t  sizeof_value,   /* sizeof single value */
    uint8_t dimension,      /* dimensionality of data */
    const size_t *size_ptr  /* array of sizes - one per axis */
) {
    assert(size_ptr);
    if (!size_ptr) return 0;
    size_t buffer_size = sizeof_value;
    for (uint8_t at = 0; at<dimension; ++at)
        buffer_size *= size_ptr[at];
    return buffer_size;
}

#pragma pack(pop)

// missing in C++11
template<typename T_type, typename... T_args> std::unique_ptr<T_type> make_unique(T_args&&... args) {
    return std::unique_ptr<T_type>(new T_type(std::forward<T_args>(args)...));
}

} //namespace {

file::arguments::arguments(neural::engine::type aengine, std::string aname, memory::format::type aformat, std::vector<uint32_t> &asize)
    : engine(aengine)
    , name(aname)
    , weight_type(weights_type::convolution)
    , output{{memory::allocate({aengine, aformat, asize})}} {}

file::arguments::arguments(neural::engine::type aengine, std::string aname, primitive aoutput)
    : engine(aengine)
    , name(aname)
    , weight_type(weights_type::convolution)
    , output{aoutput} {}

file::arguments::arguments(neural::engine::type aengine, std::string aname, weights_type type)
    : engine(aengine)
    , name(aname)
    , output({null_primitive})
    , weight_type(type) {};

// creates primitive with memry buffer loaded from specified file
primitive file::create(file::arguments arg) {
    try {
        std::ifstream rfile;
        rfile.exceptions(std::ios::failbit | std::ios::badbit);
        rfile.open(arg.name, std::ios::in | std::ios::binary);

        auto read_crc = [&rfile]() -> uint32_t {
            uint32_t result;
            rfile.read(reinterpret_cast<char *>(&result), sizeof(uint32_t));
            return result;
        };

        // load header, verify 32-bit crc
        file_header file_head;
        rfile.read(reinterpret_cast<char *>(&file_head), sizeof(file_head));
        if (read_crc() != crc32(&file_head, sizeof(file_head), CRC_INIT)) throw std::runtime_error("nn_data_t header crc mismatch");
        if (file_head.sizeof_value != sizeof(float)
            || file_head.data_type != 'F') throw std::runtime_error("nn_data_t has invalid type");

        // load size array, verify 32-bit crc
        auto array = std::unique_ptr<size_t>(new size_t[file_head.dimension]);
        auto array_size = file_head.dimension * sizeof(size_t);
        rfile.read(reinterpret_cast<char *>(array.get()), array_size);
        if (read_crc() != crc32(array.get(), array_size, CRC_INIT)) throw std::runtime_error("nn_data_t size array crc mismatch");

        // create target nn::data & load data into it               
        
        memory::arguments* p_arg = nullptr;

        switch (file_head.dimension)
        {
        case 1:
        {
            p_arg = new memory::arguments({ engine::reference, memory::format::x_f32,{ 1,{{ static_cast<unsigned int>(array.get()[0]) }}, 1 } });
            break;
        }
        case 2:
        {
            p_arg = new memory::arguments(
            { engine::reference, memory::format::xb_f32,
            {
                static_cast<unsigned int>(array.get()[0]),
                { { static_cast<unsigned int>(array.get()[1]) } },
                1
            }
            });
            break;
        }
        case 4:
        {
            if (arg.weight_type == file::weights_type::convolution)
                p_arg = new memory::arguments({ engine::reference, memory::format::oiyx_f32,{ 1,
                { static_cast<unsigned int>(array.get()[0]), static_cast<unsigned int>(array.get()[1]) }, // kernel spatials x, y
                { static_cast<unsigned int>(array.get()[3]), static_cast<unsigned int>(array.get()[2]) } } }); // ofm, ifm
            else // fully connected
                p_arg = new memory::arguments(
                { engine::reference, memory::format::xb_f32,
                    {
                         static_cast<unsigned int>(array.get()[1]) * static_cast<unsigned int>(array.get()[0]) * static_cast<unsigned int>(array.get()[2]),
                         {{ static_cast<unsigned int>(array.get()[3]) }}, 
                         1 
                    }
                });
            break;
        }
        default:
        {
            throw std::runtime_error("dimension mismatch");
            break;
        }
        }
        if (!p_arg) throw std::runtime_error("memory arguments allocation failed");

        auto memory_primitive = memory::allocate(*p_arg); // ofm, ifm
        delete p_arg;
        auto &mem = (memory_primitive).as<const neural::memory&>();
        auto buf = mem.pointer<char>();
        std::istream_iterator<char> src_begin(rfile);
        std::copy_n(src_begin, buf.size(), std::begin(buf));

        return memory_primitive;
    }
    catch (...) {
        return nullptr;
    }
}

}