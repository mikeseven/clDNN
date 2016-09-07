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
#include <iostream>
#include <nmmintrin.h>
#include <array>
#include <boost/filesystem.hpp>

namespace neural {
#define CRC_INIT 0xbaba7007

namespace {
const primitive null_primitive(nullptr);

uint32_t crc32(const void *buffer, uint64_t count, uint32_t crc) {
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
    uint8_t    data_sizeof;          // 1           - size of a single data value we will be reading i.e for "float" this will be "4".
};
// if file header is in version 2+, right after header there is file_header_ext_* struct
struct file_header_ext_2
{
    memory::format::type    layout;
};

struct nn_data{
#if defined __cplusplus
    nn_data() : buffer(nullptr), size(nullptr), dimension(0), sizeof_value(0) {};
#endif
void             *const buffer;       /* buffer containig data */
const uint64_t   *const size;         /* sizes of signal in each coordinate; unit is a value */
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
static inline uint64_t nn_data_buffer_size_ptr(
    uint64_t  sizeof_value,   /* sizeof single value */
    uint8_t dimension,      /* dimensionality of data */
    const uint64_t *size_ptr  /* array of sizes - one per axis */
) {
    assert(size_ptr);
    if (!size_ptr) return 0;
    uint64_t buffer_size = sizeof_value;
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
    , weight_type(type)
    , output({null_primitive}) {}

primitive read_file_v1_v2(engine::type eng, std::ifstream &rfile, file_header &file_head, file::weights_type type)
{
    auto read_crc = [&rfile]() -> uint32_t {
        uint32_t result;
        rfile.read(reinterpret_cast<char *>(&result), sizeof(uint32_t));
        return result;
    };

    // load header, verify 32-bit crc
    if (read_crc() != crc32(&file_head, sizeof(file_head), CRC_INIT)) throw std::runtime_error("nn_data_t header crc mismatch");
    if (file_head.data_sizeof != sizeof(float)
        || file_head.data_type != 'F') throw std::runtime_error("nn_data_t has invalid type");
    // load size array, verify 32-bit crc
    auto array = std::vector<uint64_t>(file_head.dimension);
    //std::unique_ptr<uint64_t>(new uint64_t[file_head.dimension]);
    auto array_size = file_head.dimension * sizeof(uint64_t);
    rfile.read(reinterpret_cast<char *>(&array[0]), array_size);
    if (read_crc() != crc32(&array[0], array_size, CRC_INIT)) throw std::runtime_error("nn_data_t size array crc mismatch");

    // create target nn::data & load data into it               

    std::unique_ptr<memory::arguments> p_arg = nullptr;

    switch (file_head.dimension)
    {
    case 1: // biases 1D
    {
        p_arg = std::unique_ptr<memory::arguments>(new memory::arguments({ eng, memory::format::x_f32,{ 1,{ { static_cast<uint32_t>(array[0]) } }, 1 } }));
        break;
    }
    case 2: // 2D i.e. fully connected
    {
        p_arg = std::unique_ptr<memory::arguments>(new memory::arguments(
        { eng, memory::format::bx_f32,
        {
            static_cast<uint32_t>(array[0]),
            { { static_cast<uint32_t>(array[1]) } },
            1
        }
        }));
        break;
    }
    case 3: // 3D mean
    {
        auto a = array[0], b = array[1], c = array[2];
        p_arg = std::unique_ptr<memory::arguments>(new memory::arguments(
        {
            eng, memory::format::bfyx_f32,
            {
                { 1 },
                { static_cast<uint32_t>(a), static_cast<uint32_t>(b) },
                { static_cast<uint32_t>(c) }
            }
        }));
        break;
    }
    case 4: // 4D convolution or convolution to fc conversion
    {
        if (type == file::weights_type::convolution)
            p_arg = std::unique_ptr<memory::arguments>(new memory::arguments({ eng, memory::format::oiyx_f32,{ 1,
            { static_cast<uint32_t>(array[0]), static_cast<uint32_t>(array[1]) }, // kernel spatials x, y
            { static_cast<uint32_t>(array[3]), static_cast<uint32_t>(array[2]) } } })); // ofm, ifm
        else if (type == file::weights_type::fully_connected)
        {
            p_arg = std::unique_ptr<memory::arguments>( new memory::arguments(
            { eng, memory::format::bfyx_f32,
            {
                { static_cast<uint32_t>(array[3]) }, // batches
                { static_cast<uint32_t>(array[1]), static_cast<uint32_t>(array[0]) },
                { static_cast<uint32_t>(array[2]) }, // feature maps
            }
            }));
        }
        else
            throw std::runtime_error("Unsupported weights type");
        break;
    }
    default:
    {
        throw std::runtime_error("dimension mismatch");
        break;
    }
    }

    auto memory_primitive = memory::allocate(*p_arg); // ofm, ifm
    auto &mem = (memory_primitive).as<const neural::memory&>();
    auto buf = mem.pointer<char>();
    rfile.read(&buf[0], buf.size());

    return memory_primitive;
}

primitive read_file_v3(engine::type eng, std::ifstream &rfile, file_header &file_head)
{
    file_header_ext_2 fh2;
    rfile.read(reinterpret_cast<char *>(&fh2), sizeof(fh2));
    memory::format::type format = fh2.layout;

    /*auto read_crc = [&rfile]() -> uint32_t {
        uint32_t result;
        rfile.read(reinterpret_cast<char *>(&result), sizeof(uint32_t));
        return result;
    };*/

    // load header, verify 32-bit crc
    // TODO!!!! create CRC for files with version 2 and then compare it here!
    //if (read_crc() != crc32(&file_head, sizeof(file_head), CRC_INIT)) throw std::runtime_error("nn_data_t header crc mismatch");
    if (file_head.data_sizeof != sizeof(float)
        || file_head.data_type != 'F') throw std::runtime_error("nn_data_t has invalid type");
    // load size array, verify 32-bit crc
    auto array = std::vector<uint64_t>(file_head.dimension);
    //std::unique_ptr<uint64_t>(new uint64_t[file_head.dimension]);
    auto array_size = file_head.dimension * sizeof(uint64_t);
    rfile.read(reinterpret_cast<char *>(&array[0]), array_size);
    // TODO!!!! create CRC for files with version 2 and then compare it here!
    //if (read_crc() != crc32(&array[0], array_size, CRC_INIT)) throw std::runtime_error("nn_data_t size array crc mismatch");

    // create target nn::data & load data into it               

    std::unique_ptr<memory::arguments> p_arg = nullptr;

    switch (format)
    {
    case memory::format::oyxi_f32:
    case memory::format::yxoi_f32:
    case memory::format::yxio_f32:
    {
        p_arg = std::unique_ptr<memory::arguments>(new memory::arguments({ eng, format,{ 1,
        { static_cast<uint32_t>(array[2]), static_cast<uint32_t>(array[3]) }, // kernel spatials x, y
        { static_cast<uint32_t>(array[0]), static_cast<uint32_t>(array[1]) } } })); // ofm, ifm
        break;
    }
    case memory::format::byxf_f32:
    {
        p_arg = std::unique_ptr<memory::arguments>(new memory::arguments({ eng, format,{ static_cast<uint32_t>(array[0]),
        { static_cast<uint32_t>(array[2]), static_cast<uint32_t>(array[3]) }, // kernel spatials x, y
        { static_cast<uint32_t>(array[1]) } } })); // batch, fm
        break;
    }
    default:
    {
        throw std::runtime_error("unsupported format");
        break;
    }
    }

    auto memory_primitive = memory::allocate(*p_arg); // ofm, ifm
    auto &mem = (memory_primitive).as<const neural::memory&>();
    auto buf = mem.pointer<char>();
    rfile.read(&buf[0], buf.size());

    return memory_primitive;
}

// creates primitive with memry buffer loaded from specified file
primitive file::create(file::arguments arg) {
    std::ifstream rfile;
    rfile.exceptions(std::ios::failbit | std::ios::badbit);
    rfile.open(arg.name, std::ios::binary);

    file_header file_head;
    rfile.read(reinterpret_cast<char *>(&file_head), sizeof(file_head));

    switch (file_head.version)
    {
    case 1:
    case 2:
        return read_file_v1_v2(arg.engine, rfile, file_head, arg.weight_type);
    case 3:
        return read_file_v3(arg.engine, rfile, file_head);
    default:
        throw std::runtime_error("file version not supported");
    }
}

void file::serialize(const primitive& data, const std::string& name)
{
    // TODO: start using boost
    auto size = data.as<const memory&>().argument.size;
    auto format = data.as<const memory&>().argument.format;
    boost::filesystem::path dir_path(std::string("weights_format_num") + std::to_string((uint32_t)format));
    boost::filesystem::create_directories(dir_path);
    dir_path /= boost::filesystem::path(name).filename();
    std::ofstream fstream( dir_path.string(), std::ios::out | std::ios::binary );
    file_header fh;
    file_header_ext_2 fh_ext;
    fh.data_type = 'F';
    fh.data_sizeof = sizeof(float);
    fh.version = 3;
    fh.dimension = memory::traits(format).dimension;
    if (fh.dimension == 0 || fh.dimension > 4) throw std::runtime_error("dimensions mismatch");
    // TODO: add crc validation
    fh_ext.layout = format;
    fstream.write(reinterpret_cast<const char*>(&fh),sizeof(fh));
    fstream.write(reinterpret_cast<const char*>(&fh_ext), sizeof(fh_ext));
    std::vector<uint64_t> array(fh.dimension);
    const auto dimension_offset = size.raw.size() - 4; // TODO!!! do it better way, this is needed because weights can have 5 dimensions with batch dimension always equal 1!
    for (auto ar = 0; ar < fh.dimension; ar++)
    {
        array[ar] = size.raw[dimension_offset + ar];
    }
    fstream.write(reinterpret_cast<const char*>(&array[0]), array.size()*sizeof(uint64_t));
    auto ptr = data.as<const memory&>().pointer<char>();
    fstream.write(&ptr[0], ptr.size());
}

}