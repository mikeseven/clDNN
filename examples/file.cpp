// Copyright (c) 2016, 2018 Intel Corporation
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


#include "file.h"
#include "neural_memory.h"

#include <boost/filesystem.hpp>
#include <api/CPP/data.hpp>

#include <cstdint>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <utility>


using memory = cldnn::backward_comp::neural_memory;
using memory_traits = memory::memory_traits;


namespace {

#pragma pack(push, 1)

struct file_header {
    // Data header                        // Size [B]
    std::uint8_t    magic[3];             // 3
    std::uint8_t    data_type;            // 1
    std::uint8_t    version;              // 1
    std::uint8_t    dimension;            // 1
    std::uint8_t    data_sizeof;          // 1 - Size of a single data value we will
                                          //     be reading i.e for "float" this will be "4".
};
// If file header is in version 2+, right after header there is file_header_ext_* struct
struct file_header_ext_2
{
    memory::nnd_layout_format::type    layout;
};

#pragma pack(pop)

} //namespace {

/// @brief Converts .nnd type-code (code which identifies data type used in .nnd) into proper identifier of
///        data type used in clDNN.
///
/// @param nnd_layout_type Type-code which determines layout type (compatible with old form).
/// @param nnd_data_type   Type-code which identifies data type used in .nnd file.
/// @param old_mode        Indicates that old mode of determining data type from old layout format should
///                        be used.
/// @return                Corresponding enum value which identifies data type used in clDNN.
static cldnn::data_types to_cldnn_data_type(memory::nnd_layout_format::type nnd_layout_type,
                                            std::uint8_t nnd_data_type,
                                            bool old_mode = false)
{
    switch (nnd_data_type)
    {
    case 'F': return old_mode ? memory::to_cldnn_data_type_old(nnd_layout_type)                  : cldnn::data_types::f32;
    case 'H': return old_mode ? memory::to_cldnn_data_type_old(nnd_layout_type)                  : cldnn::data_types::f16;
    case 's': throw std::runtime_error("INT16 data type is currently not supported by clDNN.");
    case 'S': throw std::runtime_error("UINT16 data type is currently not supported by clDNN.");
    case 'b': return old_mode ? memory::to_cldnn_data_type_old(nnd_layout_type)                  : cldnn::data_types::i8;
    case 'B': return old_mode ? memory::to_cldnn_data_type_old(nnd_layout_type)                  : cldnn::data_types::u8;

    default:  throw std::runtime_error("Unsupported data type encountered in .nnd file.");
    }
}

/// @brief Reads extended header (for version 3) and all data of .nnd file into clDNN memory object.
///.
/// @param rfile           Read .nnd file stream (on position of extended header). 
/// @param file_head       Main .nnd file header.
/// @param engine          clDNN engine on which memory object will allocated.
/// @param old_layout_mode Indicates that old mode of handling data type from layout should be used.
/// @return                Memory object with contents of the .nnd file.
static cldnn::memory read_file(std::ifstream &rfile, file_header &file_head, const cldnn::engine& engine,
                               bool old_layout_mode = false)
{
    file_header_ext_2 fh2{};
    rfile.read(reinterpret_cast<char *>(&fh2), sizeof(fh2));
    auto format = memory::to_cldnn_format(fh2.layout);
    auto data_type = to_cldnn_data_type(fh2.layout, file_head.data_type, old_layout_mode);

    if     ((file_head.data_sizeof != sizeof(float)         || file_head.data_type != 'F') &&
            (file_head.data_sizeof != sizeof(half_t)        || file_head.data_type != 'H') &&
            (file_head.data_sizeof != sizeof(signed char)   || file_head.data_type != 'b') &&
            (file_head.data_sizeof != sizeof(unsigned char) || file_head.data_type != 'B'))
        throw std::runtime_error("Type-code specified as data type in .nnd file has incorrect byte size "
                                 "specified in .nnd header.");

    auto array = std::vector<std::uint64_t>(file_head.dimension);
    const auto array_size = array.size() * sizeof(decltype(array)::value_type);
    rfile.read(reinterpret_cast<char*>(array.data()), array_size);

    // Create target clnn::memory and load data into it.
    std::unique_ptr<cldnn::layout> layout_ptr;

    if (file_head.dimension == 3) // Lets be sure we load imagenet.
    {
        layout_ptr = std::make_unique<cldnn::layout>(
            data_type,
            cldnn::format::bfyx,
            cldnn::tensor{
                static_cast<cldnn::tensor::value_type>(1),
                static_cast<cldnn::tensor::value_type>(array[2]),
                static_cast<cldnn::tensor::value_type>(array[0]),
                static_cast<cldnn::tensor::value_type>(array[1])
            });
    }
    else
    {
        switch (format)
        {
        case cldnn::format::bfyx:
        {
            if (array.size() == 1)
            {
                layout_ptr = std::make_unique<cldnn::layout>(
                    data_type,
                    format,
                    cldnn::tensor{
                        static_cast<cldnn::tensor::value_type>(1),
                        static_cast<cldnn::tensor::value_type>(1),
                        static_cast<cldnn::tensor::value_type>(array[0]),
                        static_cast<cldnn::tensor::value_type>(1)
                    });
            }
            else if (array.size() == 2)
            {
                layout_ptr = std::make_unique<cldnn::layout>(
                    data_type,
                    format,
                    cldnn::tensor{
                        static_cast<cldnn::tensor::value_type>(array[1]),
                        static_cast<cldnn::tensor::value_type>(1),
                        static_cast<cldnn::tensor::value_type>(array[0]),
                        static_cast<cldnn::tensor::value_type>(1)
                    });
            }
            else
            {
                layout_ptr = std::make_unique<cldnn::layout>(
                    data_type,
                    format,
                    cldnn::tensor{
                        static_cast<cldnn::tensor::value_type>(array[3]), // batches
                        static_cast<cldnn::tensor::value_type>(array[2]), // feature maps
                        static_cast<cldnn::tensor::value_type>(array[0]), // spatial x
                        static_cast<cldnn::tensor::value_type>(array[1])  // spatial y
                    });
            }
            break;
        }

        case cldnn::format::byxf: // TODO: Explain why this has tha same handling as below?
        case cldnn::format::yxfb:
        {
            if (array.size() == 2)
            {
                layout_ptr = std::make_unique<cldnn::layout>(
                    data_type,
                    format,
                    cldnn::tensor{
                        static_cast<cldnn::tensor::value_type>(array[0]), // batch
                        static_cast<cldnn::tensor::value_type>(1),
                        static_cast<cldnn::tensor::value_type>(array[1]),
                        static_cast<cldnn::tensor::value_type>(1)
                    });
            }
            else
            {
                layout_ptr = std::make_unique<cldnn::layout>(
                    data_type,
                    format,
                    cldnn::tensor{
                        static_cast<cldnn::tensor::value_type>(array[0]), // batch
                        static_cast<cldnn::tensor::value_type>(array[1]),
                        static_cast<cldnn::tensor::value_type>(array[2]),
                        static_cast<cldnn::tensor::value_type>(array[3])  // kernel spatials y, x
                    });
            }
            break;
        }

        default:
            throw std::runtime_error("unsupported format");
        }
    }

    auto mem = cldnn::memory::allocate(engine, *layout_ptr);
    auto buf = mem.pointer<char>();
    rfile.read(&buf[0], buf.size());

    return mem;
}


file::arguments::arguments(const cldnn::engine& eng, std::string file_name)
    : engine(eng), name(std::move(file_name)) {}

cldnn::memory file::read(file::arguments arg, bool validate_magic) {
    std::ifstream rfile;
    rfile.exceptions(std::ios::failbit | std::ios::badbit);
    rfile.open(arg.name, std::ios::binary);

    file_header file_head{};
    rfile.read(reinterpret_cast<char*>(&file_head), sizeof(file_head));

    if (validate_magic)
    {
        if (file_head.magic[0] != 'n' || file_head.magic[1] != 'n' || file_head.magic[2] != 'd')
            throw std::runtime_error("Header of .nnd file is invalid (\"magic\" check failed).");
    }

    if(file_head.version == 3)
        return read_file(rfile, file_head, arg.engine);

    throw std::runtime_error("Version of .nnd file is not supported.");
}

cldnn::data file::create(file::arguments arg, bool validate_magic) {
    auto data_prim_id = boost::filesystem::path(arg.name).filename().string();

    return cldnn::data(data_prim_id, file::read(arg, validate_magic));
}

cldnn::mutable_data file::create_mutable(file::arguments arg, bool initialize, cldnn::layout layout,cldnn::mutable_data::filler_type filler_type) {
    auto data_id = boost::filesystem::path(arg.name).filename().string();
    
    if (initialize)
    {
        auto mem = cldnn::memory::allocate(arg.engine, layout);
        return cldnn::mutable_data(data_id, mem, filler_type);
    }

    return cldnn::mutable_data(data_id, file::read(arg));
}

void file::serialize(const cldnn::memory& data, const std::string& file_name, bool old_layout_mode)
{
    auto size = memory_traits(data, old_layout_mode).size;
    auto format = memory_traits(data, old_layout_mode).format;

    boost::filesystem::path dir_path("weights_format_num" + std::to_string(static_cast<std::uint32_t>(format)));
    create_directories(dir_path);
    dir_path /= boost::filesystem::path(file_name).filename();

    std::ofstream fstream(dir_path.string(), std::ios::out | std::ios::binary);

    file_header fh{};
    file_header_ext_2 fh_ext{};

    fh.magic[0] = 'n';
    fh.magic[1] = 'n';
    fh.magic[2] = 'd';

    switch (data.get_layout().data_type)
    {
    case cldnn::data_types::f32: fh.data_type = 'F'; fh.data_sizeof = sizeof(float); break;
    case cldnn::data_types::f16: fh.data_type = 'H'; fh.data_sizeof = sizeof(half_t); break;
    case cldnn::data_types::i8:  fh.data_type = 'b'; fh.data_sizeof = sizeof(signed char); break;
    case cldnn::data_types::u8:  fh.data_type = 'B'; fh.data_sizeof = sizeof(unsigned char); break;

    default: throw std::logic_error("Encountered unhandled clDNN data type.");
    }

    fh.version = 3;

    fh.dimension = static_cast<uint8_t>(memory::layout_traits(data.get_layout()).dimension);
    if (fh.dimension == 0 || fh.dimension > 4)
        throw std::runtime_error("Number of dimensions is out of supported range.");

    fh_ext.layout = format;

    fstream.write(reinterpret_cast<const char*>(&fh), sizeof(fh));
    fstream.write(reinterpret_cast<const char*>(&fh_ext), sizeof(fh_ext));

    std::vector<std::uint64_t> array(fh.dimension);
    const auto dimension_offset = size.raw.size() - fh.dimension; // TODO!!! do it better way, this is needed because weights can have 5 dimensions with batch dimension always equal 1!
    for (auto ar = 0; ar < fh.dimension; ar++)
    {
        array[ar] = size.raw[dimension_offset + ar];
    }

    fstream.write(reinterpret_cast<const char*>(&array[0]), array.size() * sizeof(uint64_t));
    auto ptr = data.pointer<char>();
    fstream.write(&ptr[0], ptr.size());
}

void file::serialize_train(const cldnn::memory& data, const std::string& file_name)
{
    // This function is used for weights updates in network training
    auto size = memory_traits(data).size;
    auto format = memory_traits(data).format;
    boost::filesystem::path dir_path(std::string("weights_format_num") + std::to_string((uint32_t)format));
    boost::filesystem::create_directories(dir_path);
    dir_path /= boost::filesystem::path(file_name).filename();
    std::ofstream fstream(file_name, std::ios::out | std::ios::binary);
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

    auto data_layout_size = data.get_layout().size;
    if (data_layout_size.batch[0] == 1 && data_layout_size.feature[0] == 1 && data_layout_size.spatial[1] == 1)
        fh.dimension = 1;
    else if (data_layout_size.feature[0] == 1 && data_layout_size.spatial[1] == 1)
        fh.dimension = 2;
    else
        fh.dimension = 4;

    if (fh.dimension == 0 || fh.dimension > 4) throw std::runtime_error("dimensions mismatch");

    if (fh.dimension == 1)
        fh_ext.layout = cldnn::backward_comp::neural_memory::nnd_layout_format::type::bias_x_f32;
    else if (fh.dimension == 2)
        fh_ext.layout = cldnn::backward_comp::neural_memory::nnd_layout_format::type::fc_bx_f32;
    else
        fh_ext.layout = cldnn::backward_comp::neural_memory::nnd_layout_format::type::weights_bfyx_f32;
    if (fh.dimension == 0 || fh.dimension > 4) throw std::runtime_error("dimensions mismatch");

    fstream.write(reinterpret_cast<const char*>(&fh), sizeof(fh));
    fstream.write(reinterpret_cast<const char*>(&fh_ext), sizeof(fh_ext));

    std::vector<std::uint64_t> array(fh.dimension);

    if (fh.dimension == 1)
    {
        array[0] = size.raw[2];
    }
    else if (fh.dimension == 2)
    {
        array[1] = size.raw[0];
        array[0] = size.raw[2];
    }
    else
    {
        array[3] = size.raw[0];
        array[2] = size.raw[1];
        array[1] = size.raw[2];
        array[0] = size.raw[3];
    }

    fstream.write(reinterpret_cast<const char*>(&array[0]), array.size() * sizeof(uint64_t));
    auto ptr = data.pointer<char>();
    fstream.write(&ptr[0], ptr.size());
}