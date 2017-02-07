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

#include "file.h"
#include <fstream>
#include <iostream>
#include <nmmintrin.h>
#include <array>
#include <boost/filesystem.hpp>
#include <api/primitives/data.hpp>

using memory = cldnn::neural_memory;
#define CRC_INIT 0xbaba7007

namespace {
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

file::arguments::arguments(const cldnn::engine& eng, const std::string& aname, weights_type type)
    : engine(eng)
    , name(aname)
    , weight_type(type)
{}

cldnn::memory read_file_v1_v2(std::ifstream &rfile, file_header &file_head, file::weights_type type, const cldnn::engine& engine)
{
    auto read_crc = [&rfile]() -> uint32_t {
        uint32_t result;
        rfile.read(reinterpret_cast<char *>(&result), sizeof(uint32_t));
        return result;
    };

    // load header, verify 32-bit crc
    if (read_crc() != crc32(&file_head, sizeof(file_head), CRC_INIT)) throw std::runtime_error("nn_data_t header crc mismatch");
    if ((file_head.data_sizeof != sizeof(float) || file_head.data_type != 'F') &&
        (file_head.data_sizeof != sizeof(half_t) || file_head.data_type != 'H')) throw std::runtime_error("nn_data_t has invalid type");
    auto use_fp16_data = file_head.data_type == 'H';
    // load size array, verify 32-bit crc
    auto array = std::vector<uint64_t>(file_head.dimension);
    //std::unique_ptr<uint64_t>(new uint64_t[file_head.dimension]);
    auto array_size = file_head.dimension * sizeof(uint64_t);
    rfile.read(reinterpret_cast<char *>(&array[0]), array_size);
    if (read_crc() != crc32(&array[0], array_size, CRC_INIT)) 
        throw std::runtime_error("nn_data_t size array crc mismatch");

    // create target nn::data & load data into it               

    std::unique_ptr<cldnn::layout> p_arg = nullptr;
    auto data_type = use_fp16_data ? cldnn::data_types::f16 : cldnn::data_types::f32;

    switch (file_head.dimension)
    {
    case 1: // biases 1D
    {
        p_arg = std::unique_ptr<cldnn::layout>(new cldnn::layout(data_type, { cldnn::format::x, {static_cast<cldnn::tensor::value_type>(array[0])} }));
        break;
    }
    case 2: // 2D i.e. fully connected
    {
        p_arg = std::unique_ptr<cldnn::layout>(new cldnn::layout(data_type, 
        { cldnn::format::bx,
        {
            static_cast<cldnn::tensor::value_type>(array[1]),
            static_cast<cldnn::tensor::value_type>(array[0])
        }
        }));
        break;
    }
    case 3: // 3D mean
    {
        auto a = array[0], b = array[1], c = array[2];
        p_arg = std::unique_ptr<cldnn::layout>(new cldnn::layout(data_type,
        { cldnn::format::bfyx,
        {
            static_cast<cldnn::tensor::value_type>(1),
            static_cast<cldnn::tensor::value_type>(c),
            static_cast<cldnn::tensor::value_type>(b),
            static_cast<cldnn::tensor::value_type>(a)
        }
        }));
        break;
    }
    case 4: // 4D convolution or convolution to fc conversion
    {
        if (type == file::weights_type::convolution)
        {
            p_arg = std::unique_ptr<cldnn::layout>(new cldnn::layout(data_type,
            { cldnn::format::oiyx,
            {
                static_cast<cldnn::tensor::value_type>(array[3]), static_cast<cldnn::tensor::value_type>(array[2]), // ofm, ifm
                static_cast<cldnn::tensor::value_type>(array[1]), static_cast<cldnn::tensor::value_type>(array[0])  // kernel spatials y, x
            }
            }));
        }
        else if (type == file::weights_type::fully_connected)
        {
            p_arg = std::unique_ptr<cldnn::layout>(new cldnn::layout(data_type,
            { cldnn::format::bfyx,
            {
                static_cast<cldnn::tensor::value_type>(array[3]), // batches
                static_cast<cldnn::tensor::value_type>(array[2]), // feature maps
                static_cast<cldnn::tensor::value_type>(array[1]), static_cast<cldnn::tensor::value_type>(array[0])  // kernel spatials y, x
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

    auto mem = cldnn::memory::allocate(engine, *p_arg); // ofm, ifm
    auto buf = mem.pointer<char>();
    rfile.read(buf.data(), buf.size());

    return mem;
}

cldnn::memory read_file_v3(std::ifstream &rfile, file_header &file_head, const cldnn::engine& engine)
{
    file_header_ext_2 fh2;
    rfile.read(reinterpret_cast<char *>(&fh2), sizeof(fh2));
    auto format = memory::to_tensor_format(fh2.layout);
    auto data_type = memory::to_data_type(fh2.layout);

    /*auto read_crc = [&rfile]() -> uint32_t {
        uint32_t result;
        rfile.read(reinterpret_cast<char *>(&result), sizeof(uint32_t));
        return result;
    };*/

    // load header, verify 32-bit crc
    // TODO!!!! create CRC for files with version 2 and then compare it here!
    //if (read_crc() != crc32(&file_head, sizeof(file_head), CRC_INIT)) throw std::runtime_error("nn_data_t header crc mismatch");
    if ((file_head.data_sizeof != sizeof(float) || file_head.data_type != 'F') &&
        (file_head.data_sizeof != sizeof(half_t) || file_head.data_type != 'H')) throw std::runtime_error("nn_data_t has invalid type");
    // load size array, verify 32-bit crc
    auto array = std::vector<uint64_t>(file_head.dimension);
    //std::unique_ptr<uint64_t>(new uint64_t[file_head.dimension]);
    auto array_size = file_head.dimension * sizeof(uint64_t);
    rfile.read(reinterpret_cast<char *>(&array[0]), array_size);
    // TODO!!!! create CRC for files with version 2 and then compare it here!
    //if (read_crc() != crc32(&array[0], array_size, CRC_INIT)) throw std::runtime_error("nn_data_t size array crc mismatch");

    // create target nn::data & load data into it               

    std::unique_ptr<cldnn::layout> p_arg = nullptr;

    switch (format)
    {
    case cldnn::format::oiyx: //CONV 
    {
        p_arg = std::unique_ptr<cldnn::layout>(new cldnn::layout(data_type,
        { cldnn::format::oiyx,
        {
            static_cast<cldnn::tensor::value_type>(array[3]), static_cast<cldnn::tensor::value_type>(array[2]), // ofm, ifm
            static_cast<cldnn::tensor::value_type>(array[1]), static_cast<cldnn::tensor::value_type>(array[0])  // kernel spatials y, x
        }
        }));
        break;
    }

    case cldnn::format::bfyx: //FC
    {
        p_arg = std::unique_ptr<cldnn::layout>(new cldnn::layout(data_type,
        { cldnn::format::bfyx,
        {
            static_cast<cldnn::tensor::value_type>(array[3]), // batches
            static_cast<cldnn::tensor::value_type>(array[2]), // feature maps
            static_cast<cldnn::tensor::value_type>(array[1]), static_cast<cldnn::tensor::value_type>(array[0])  // kernel spatials y, x
        }
        }));
        break;
    }

    case cldnn::format::bx: // 2D
    {
        p_arg = std::unique_ptr<cldnn::layout>(new cldnn::layout(data_type,
        { cldnn::format::bx,
        {
            static_cast<cldnn::tensor::value_type>(array[1]),
            static_cast<cldnn::tensor::value_type>(array[0])
        }
        }));
        break;
    }

    case cldnn::format::x: // 1D
    {
        p_arg = std::unique_ptr<cldnn::layout>(new cldnn::layout(data_type, { cldnn::format::x,{ static_cast<cldnn::tensor::value_type>(array[0]) } }));
        break;

    }

    // FP32
    case cldnn::format::oyxi:
    case cldnn::format::yxoi:
    case cldnn::format::yxio:
    {
        auto size = cldnn::tensor(cldnn::format::oiyx,
        {
            static_cast<cldnn::tensor::value_type>(array[0]), static_cast<cldnn::tensor::value_type>(array[1]), // ofm, ifm
            static_cast<cldnn::tensor::value_type>(array[3]), static_cast<cldnn::tensor::value_type>(array[2])  // kernel spatials y, x
        }
        ).transform(format, 1);
        p_arg = std::unique_ptr<cldnn::layout>(new cldnn::layout(data_type, size));
        break;
    }

    // FP32
    case cldnn::format::byxf:
    case cldnn::format::yxfb:
    {
        auto size = cldnn::tensor(cldnn::format::byxf,
        {
            static_cast<cldnn::tensor::value_type>(array[0]), // batch
            static_cast<cldnn::tensor::value_type>(array[3]), static_cast<cldnn::tensor::value_type>(array[2]),  // kernel spatials y, x
            static_cast<cldnn::tensor::value_type>(array[1]), // fm
        }
        ).transform(format, 1);
        p_arg = std::unique_ptr<cldnn::layout>(new cldnn::layout(data_type, size));
        break;
    }

    // FP32
    case cldnn::format::xb:
    {
        auto size = cldnn::tensor(cldnn::format::xb,
        {
            static_cast<cldnn::tensor::value_type>(array[1]), // x
            static_cast<cldnn::tensor::value_type>(array[0]), // batch
        }
        );
        p_arg = std::unique_ptr<cldnn::layout>(new cldnn::layout(data_type, size));
        break;
    }

    default:
        throw std::runtime_error("unsupported format");
    }

    auto mem = cldnn::memory::allocate(engine, *p_arg); // ofm, ifm
    auto buf = mem.pointer<char>();
    rfile.read(&buf[0], buf.size());

    return mem;
}

// creates primitive with memry buffer loaded from specified file
cldnn::memory file::read(file::arguments arg) {
    std::ifstream rfile;
    rfile.exceptions(std::ios::failbit | std::ios::badbit);
    rfile.open(arg.name, std::ios::binary);

    file_header file_head;
    rfile.read(reinterpret_cast<char *>(&file_head), sizeof(file_head));

    switch (file_head.version)
    {
    case 1:
    case 2:
        return read_file_v1_v2(rfile, file_head, arg.weight_type, arg.engine);
    case 3:
        return read_file_v3(rfile, file_head, arg.engine);
    default:
        throw std::runtime_error("file version not supported");
    }
}

cldnn::data::dto* file::create(file::arguments arg) {
    auto data_id = boost::filesystem::path(arg.name).filename().string();
    auto ret = cldnn::data(data_id, file::read(arg));
    return cldnn::as_dto<cldnn::data>(ret.get_dto());
}

void file::serialize(const cldnn::memory& data, const std::string& name)
{
    // TODO: start using boost
    auto size = data.argument().size;
    auto format = data.argument().format;
    boost::filesystem::path dir_path(std::string("weights_format_num") + std::to_string((uint32_t)format));
    boost::filesystem::create_directories(dir_path);
    dir_path /= boost::filesystem::path(name).filename();
    std::ofstream fstream( dir_path.string(), std::ios::out | std::ios::binary );
    file_header fh;
    file_header_ext_2 fh_ext;
    if (data.get_layout().data_type == cldnn::data_types::f16)
    {
        fh.data_type = 'H';
        fh.data_sizeof = sizeof(half_t);
    }
    else
    {
        fh.data_type = 'F';
        fh.data_sizeof = sizeof(float);
    }
    fh.version = 3;
    fh.dimension = static_cast<uint8_t>(memory::traits(data.get_layout()).dimension);
    if (fh.dimension == 0 || fh.dimension > 4) throw std::runtime_error("dimensions mismatch");
    // TODO: add crc validation
    fh_ext.layout = format;
    fstream.write(reinterpret_cast<const char*>(&fh),sizeof(fh));
    fstream.write(reinterpret_cast<const char*>(&fh_ext), sizeof(fh_ext));
    std::vector<uint64_t> array(fh.dimension);
    if (format == memory::format::type::xb_f32 || format == memory::format::type::bx_f32 ||
        format == memory::format::type::xb_f16 || format == memory::format::type::bx_f16)
    {
        array[0] = size.batch[0];
        array[1] = size.spatial[0];
    }
    else
    {
        const auto dimension_offset = size.raw.size() - fh.dimension; // TODO!!! do it better way, this is needed because weights can have 5 dimensions with batch dimension always equal 1!
        for (auto ar = 0; ar < fh.dimension; ar++)
        {
            array[ar] = size.raw[dimension_offset + ar];
        }
    }
    fstream.write(reinterpret_cast<const char*>(&array[0]), array.size()*sizeof(uint64_t));
    auto ptr = data.pointer<char>();
    fstream.write(&ptr[0], ptr.size());
}

