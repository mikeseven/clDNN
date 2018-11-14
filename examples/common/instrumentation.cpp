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


#include "instrumentation.h"
#include "neural_memory.h"

#include <fstream>
#include <iomanip>
#include <string>
#include <vector>
#include <iostream>
#include <boost/filesystem.hpp>


using memory_traits = cldnn::backward_comp::neural_memory::memory_traits;


namespace instrumentation {
    // initalize dumping directory for whole run
    const std::string logger::dump_dir = "cldnn_dumps/" + std::to_string(std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()));

    static float convert_half_to_float(half_t val, bool flush_denorm_to_zero = false)
    {
#if defined HALF_HALF_HPP
        return val;
#else
        // FP32 parts extracted from FP16.
        uint32_t sign = (static_cast<uint16_t>(val) & 0x8000U) << 16;
        uint32_t mantissa = (static_cast<uint16_t>(val) & 0x3FFU) << 13;

        uint32_t exp_val_f16 = (static_cast<uint16_t>(val) & 0x7C00U) >> 10;
        uint32_t exp;
        if (exp_val_f16 == 0)
        {
            // Handling +/-0 and denormals.
            if (mantissa == 0)
            {
                exp = 0;
            }
            else if (flush_denorm_to_zero)
            {
                sign = 0;
                exp = 0;
                mantissa = 0;
            }
            else
            {
                // Denorms conversion to normal numbers.
                exp = 127 - 15;
                while (!(mantissa & 0x400000U))
                {
                    mantissa <<= 1;
                    --exp;
                }
                mantissa = (mantissa << 1) & 0x7FFFFFU;
                exp <<= 23;
            }
        }
        else
        {
            // Handling +/-infinity, NaN and normal numbers.
            exp = (exp_val_f16 == 0x1FU ? 0xFFU : exp_val_f16 + 127 - 15) << 23;
        }

        float ret;
        reinterpret_cast<uint32_t&>(ret) = sign | exp | mantissa;

        return ret;
#endif
    }

    inline float convert_element(float f)
    {
        return f;
    }

    inline float convert_element(half_t h)
    {
        return convert_half_to_float(h);
    }

    inline float convert_element(char c)
    {
        return static_cast<float>(c);
    }

    inline float convert_element(signed char c)
    {
        return static_cast<float>(c);
    }

    inline float convert_element(unsigned char c)
    {
        return static_cast<float>(c);
    }

    template<typename elemType>
    void dump_byxf(const cldnn::memory& mem, bool single_batch, cldnn::tensor::value_type batch_id, bool single_feature, cldnn::tensor::value_type feature_id, std::vector<std::vector<std::stringstream>> & streams)
    {
        auto mem_arg = memory_traits(mem);
        auto mem_ptr = mem.pointer<elemType>();

        unsigned int input_it = 0;
        for (cldnn::tensor::value_type b = 0; b < mem_arg.size.batch[0]; b++)
        {
            for (cldnn::tensor::value_type y = 0; y < mem_arg.size.spatial[1]; y++)
            {
                for (cldnn::tensor::value_type x = 0; x < mem_arg.size.spatial[0]; x++)
                {
                    for (cldnn::tensor::value_type f = 0; f < mem_arg.size.feature[0]; f++)
                    {
                        if ((!single_batch || b == batch_id) && (!single_feature || f == feature_id))
                        {
                            streams[b][f] << convert_element(mem_ptr[input_it]) << " ";
                            if (x == mem_arg.size.spatial[0] - 1)
                            {
                                streams[b][f] << std::endl;
                            }
                        }
                        input_it++;
                    }
                }
            }
        }
    }

    template<typename elemType>
    void dump_bfyx(const cldnn::memory& mem, bool single_batch, cldnn::tensor::value_type batch_id, bool single_feature, cldnn::tensor::value_type feature_id, std::vector<std::vector<std::stringstream>> & streams)
    {
        auto mem_arg = memory_traits(mem);
        auto mem_ptr = mem.pointer<elemType>();

        unsigned int input_it = 0;
        for (cldnn::tensor::value_type b = 0; b < mem_arg.size.batch[0]; b++)
        {
            for (cldnn::tensor::value_type f = 0; f < mem_arg.size.feature[0]; f++)
            {
                for (cldnn::tensor::value_type y = 0; y < mem_arg.size.spatial[1]; y++)
                {
                    for (cldnn::tensor::value_type x = 0; x < mem_arg.size.spatial[0]; x++)
                    {
                        if ((!single_batch || b == batch_id) && (!single_feature || f == feature_id))
                        {
                            streams[b][f] << convert_element(mem_ptr[input_it]) << " ";
                            if (x == mem_arg.size.spatial[0] - 1)
                            {
                                streams[b][f] << std::endl;
                            }
                        }
                        input_it++;
                    }
                }
            }
        }
    }

    template<typename elemType>
    void dump_yxfb(const cldnn::memory& mem, bool single_batch, cldnn::tensor::value_type batch_id, bool single_feature, cldnn::tensor::value_type feature_id, std::vector<std::vector<std::stringstream>> & streams)
    {
        auto mem_arg = memory_traits(mem);
        auto mem_ptr = mem.pointer<elemType>();

        unsigned int input_it = 0;
        for (cldnn::tensor::value_type y = 0; y < mem_arg.size.spatial[1]; y++)
        {
            for (cldnn::tensor::value_type x = 0; x < mem_arg.size.spatial[0]; x++)
            {
                for (cldnn::tensor::value_type f = 0; f < mem_arg.size.feature[0]; f++)
                {
                    for (cldnn::tensor::value_type b = 0; b < mem_arg.size.batch[0]; b++)
                    {
                        if ((!single_batch || b == batch_id) && (!single_feature || f == feature_id))
                        {
                            streams[b][f] << convert_element(mem_ptr[input_it]) << " ";
                            if (x == mem_arg.size.spatial[0] - 1)
                            {
                                streams[b][f] << std::endl;
                            }
                        }
                        input_it++;
                    }
                }
            }
        }
    }

    template<typename elemType>
    void dump_byx8_f4(const cldnn::memory& mem, bool single_batch, cldnn::tensor::value_type batch_id, bool single_feature, cldnn::tensor::value_type feature_id, std::vector<std::vector<std::stringstream>> & streams)
    {
        auto mem_ptr = mem.pointer<elemType>();
        const auto& layout = mem.get_layout();
        auto size = layout.size;

        size.feature[0] = ((size.feature[0] + 3) / 4) * 4;
        size.spatial[0] = ((size.spatial[0] + 7) / 8) * 8;

        const auto b_size = size.batch[0];
        const auto f_size = size.feature[0];
        const auto x_size = size.spatial[0] + layout.data_padding.lower_size().spatial[0] + layout.data_padding.upper_size().spatial[0];
        const auto y_size = size.spatial[1] + layout.data_padding.lower_size().spatial[1] + layout.data_padding.upper_size().spatial[1];

        unsigned int input_it = 0;
        for (cldnn::tensor::value_type b = 0; b < b_size; b++)
        {
            for (cldnn::tensor::value_type y = 0; y < y_size; y++)
            {
                for (cldnn::tensor::value_type x = 0; x < x_size; x++)
                {
                    for (cldnn::tensor::value_type f = 0; f < f_size; f++)
                    {
                        if ((!single_batch || b == batch_id) && (!single_feature || f == feature_id))
                        {
                            streams[b][f] << convert_element(mem_ptr[input_it]) << " ";
                            if (x == x_size - 1)
                            {
                                streams[b][f] << std::endl;
                            }
                        }
                        input_it++;
                    }
                }
            }
        }
    }

    template<typename elemType>
    void dump_fs_bs_yx_bsv4_fsv32(const cldnn::memory& mem, bool single_batch, cldnn::tensor::value_type batch_id, bool single_feature, cldnn::tensor::value_type feature_id, std::vector<std::vector<std::stringstream>> & streams)
    {
        auto mem_ptr = mem.pointer<elemType>();

        const auto& layout = mem.get_layout();
        auto size = layout.size;
        size.feature[0] = ((size.feature[0] + 31) / 32) * 32;
        size.batch[0] = ((size.batch[0] + 3) / 4) * 4;

        const auto bs_size = size.batch[0] / 4;
        const auto fs_size = size.feature[0] / 32;
        const auto x_size = size.spatial[0] + layout.data_padding.lower_size().spatial[0] + layout.data_padding.upper_size().spatial[0];
        const auto y_size = size.spatial[1] + layout.data_padding.lower_size().spatial[1] + layout.data_padding.upper_size().spatial[1];

        unsigned int input_it = 0;
        if (single_batch && single_feature)
        {
            const size_t x_pitch = 32 * 4;
            const size_t y_pitch = 32 * 4 * x_size;
            const size_t b_block_pitch = y_pitch * y_size;
            const size_t f_block_pitch = b_block_pitch * bs_size;
            input_it += (unsigned int)(batch_id % 4 + (batch_id / 4) * b_block_pitch);
            input_it += (unsigned int)(feature_id % 32 + (feature_id / 32) * f_block_pitch);
            for (cldnn::tensor::value_type y = 0; y < y_size; y++)
            {
                for (cldnn::tensor::value_type x = 0; x < x_size; x++)
                {
                    streams[batch_id][feature_id] << convert_element(mem_ptr[input_it]) << " ";
                    if (x == x_size - 1)
                    {
                        streams[batch_id][feature_id] << std::endl;
                    }
                    input_it++;
                }
            }
        }
        else
        {
            for (cldnn::tensor::value_type fs = 0; fs < fs_size; fs++)
            {
                for (cldnn::tensor::value_type bs = 0; bs < bs_size; bs++)
                {
                    for (cldnn::tensor::value_type y = 0; y < y_size; y++)
                    {
                        for (cldnn::tensor::value_type x = 0; x < x_size; x++)
                        {
                            for (cldnn::tensor::value_type bv = 0; bv < 4; bv++)
                            {
                                for (cldnn::tensor::value_type fv = 0; fv < 32; fv++)
                                {
                                    const auto b = bs * 4 + bv;
                                    const auto f = fs * 32 + fv;
                                    if ((!single_batch || b == batch_id) && (!single_feature || f == feature_id))
                                    {
                                        streams[b][f] << convert_element(mem_ptr[input_it]) << " ";
                                        if (x == x_size - 1)
                                        {
                                            streams[b][f] << std::endl;
                                        }
                                    }
                                    input_it++;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    template<typename elemType>
    void dump_xb(const cldnn::memory& mem, bool single_batch, cldnn::tensor::value_type batch_id, std::vector<std::vector<std::stringstream>> & streams)
    {
        auto mem_arg = memory_traits(mem);
        auto mem_ptr = mem.pointer<elemType>();

        unsigned int input_it = 0;
        for (cldnn::tensor::value_type x = 0; x < mem_arg.size.spatial[0]; x++)
        {
            for (cldnn::tensor::value_type b = 0; b < mem_arg.size.batch[0]; b++)
            {
                if (!single_batch || b == batch_id)
                {
                    streams[b][0] << convert_element(mem_ptr[input_it]) << std::endl;
                }
                input_it++;
            }
        }
    }

    template<typename elemType>
    void dump_bx(const cldnn::memory& mem, bool single_batch, cldnn::tensor::value_type batch_id, std::vector<std::vector<std::stringstream>> & streams)
    {
        auto mem_arg = memory_traits(mem);
        auto mem_ptr = mem.pointer<elemType>();

        unsigned int input_it = 0;
        for (cldnn::tensor::value_type b = 0; b < mem_arg.size.batch[0]; b++)
        {
            for (cldnn::tensor::value_type x = 0; x < mem_arg.size.spatial[0]; x++)
            {
                if (!single_batch || b == batch_id)
                {
                    streams[b][0] << convert_element(mem_ptr[input_it]) << std::endl;
                }
                input_it++;
            }
        }
    }

    template<typename elemType>
    void dump_yxio(const cldnn::memory& mem, std::stringstream & stream)
    {
        auto mem_arg = memory_traits(mem);
        auto mem_ptr = mem.pointer<elemType>();

        auto i_size = mem_arg.size.batch[0];
        auto o_size = mem_arg.size.feature[0];
        auto x_size = mem_arg.size.spatial[0];
        auto y_size = mem_arg.size.spatial[1];
        unsigned int input_it = 0;
        for (cldnn::tensor::value_type o = 0; o < o_size; o++)
        {
            for (cldnn::tensor::value_type i = 0; i < i_size; i++)
            {
                for (cldnn::tensor::value_type x = 0; x < x_size; x++)
                {
                    for (cldnn::tensor::value_type y = 0; y < y_size; y++)
                    {
                        stream<< convert_element(mem_ptr[input_it]) << " ";
                        input_it++;
                    }
                    stream<< std::endl;
                }
            }
        }
    }

    template<typename elemType>
    void dump_oiyx(const cldnn::memory& mem, std::stringstream & stream)
    {
        auto mem_arg = memory_traits(mem);
        auto mem_ptr = mem.pointer<elemType>();

        auto i_size = mem_arg.size.batch[0];
        auto o_size = mem_arg.size.feature[0];
        auto x_size = mem_arg.size.spatial[0];
        auto y_size = mem_arg.size.spatial[1];
        unsigned int input_it = 0;
        for (cldnn::tensor::value_type x = 0; x < x_size; x++)
        {
            for (cldnn::tensor::value_type y = 0; y < y_size; y++)
            {
                for (cldnn::tensor::value_type i = 0; i < i_size; i++)
                {
                    for (cldnn::tensor::value_type o = 0; o < o_size; o++)
                    {
                        stream << convert_element(mem_ptr[input_it]) << " ";
                        input_it++;
                    }
                    stream << std::endl;
                }
            }
        }
    }

    template<typename elemType>
    void dump_os_iyx_osv16(const cldnn::memory& mem, std::stringstream & stream)
    {
        auto mem_arg = memory_traits(mem);
        auto mem_ptr = mem.pointer<elemType>();

        auto i_size = mem_arg.size.batch[0];
        auto o_size = mem_arg.size.feature[0];
        auto x_size = mem_arg.size.spatial[0];
        auto y_size = mem_arg.size.spatial[1];
        auto weights_size = i_size * o_size * x_size * y_size; //count() also counts feature[1]
        int slice_value = 16;
        cldnn::tensor::value_type it = 0;
        while (it < weights_size)
        {
            stream << convert_element(mem_ptr[it]) << " ";
            it++;
            if (it % slice_value == 0) //separate every bsv with a new line
                stream << std::endl;
        };
    }

    template<typename elemType>
    void dump_bs_xs_xsv8_bsv8(const cldnn::memory& mem, std::stringstream & stream)
    {
        auto mem_arg = memory_traits(mem);
        auto mem_ptr = mem.pointer<elemType>();

        auto i_size = mem_arg.size.batch[0]; //batch = input feature map
        auto x_size = mem_arg.size.spatial[0]; // spatial_x = output feature map
        auto weights_size = mem_arg.size.count();
        int xsv = 8, bsv = 8; 
        unsigned int input_it = 0, input_i_it= 0 , input_o_it = 0;
        for (cldnn::tensor::value_type it = 0; it < weights_size; it++)
        {
                stream << convert_element(mem_ptr[input_it]) << " ";
                input_i_it++;
                if (input_i_it % bsv == 0) //separete every input slice with a new line
                {
                    stream << std::endl;
                    input_o_it++;
                    input_i_it = 0;
                }
                input_it = input_o_it*bsv + input_i_it;

                if (input_it % (xsv*bsv) == 0) // seperate every block (8x8) with a new line
                    stream << std::endl;
        }
    }

    template<typename elemType>
    void dump_bs_x_bsv16(const cldnn::memory& mem, std::stringstream & stream)
    {
        auto mem_arg = memory_traits(mem);
        auto mem_ptr = mem.pointer<elemType>();

        auto i_size = mem_arg.size.batch[0]; //batch = input feature map
        auto x_size = mem_arg.size.spatial[0]; // spatial_x = output feature map
        auto weights_size = mem_arg.size.count();
        int bsv = 16;
        cldnn::tensor::value_type it = 0;
        while (it < weights_size)
        {
            stream << convert_element(mem_ptr[it]) << " ";
            it++;
            if (it % bsv == 0) //separate every bsv with a new line
                stream << std::endl;
        }
    }

    template <class T>
    void dump(const cldnn::memory& mem, std::stringstream& stream)
    {
        auto mem_ptr = mem.pointer<T>();

        auto&& pitches = mem.get_layout().get_pitches();
        auto&& size = mem.get_layout().size;
        for (cldnn::tensor::value_type b = 0; b < size.batch[0]; ++b)
        {
            stream << "============= BATCH " << b << " ============\n\n";
            for (cldnn::tensor::value_type f = 0; f < size.feature[0]; ++f)
            {
                stream << "feature " << f << ":\n";
                for (cldnn::tensor::value_type y = 0; y < size.spatial[1]; ++y)
                {
                    for (cldnn::tensor::value_type x = 0; x < size.spatial[0]; ++x)
                    {
                        unsigned int input_it = b*pitches.batch[0] + f*pitches.feature[0] + y*pitches.spatial[1] + x*pitches.spatial[0];
                        stream << convert_element(mem_ptr[input_it]) << " ";
                        input_it++;
                    }
                    stream << '\n';
                }
                stream << std::endl;
            }
        }
    }

    template <class T>
    void dump(const cldnn::memory& mem, std::vector<std::vector<std::stringstream>>& streams)
    {
        auto mem_ptr = mem.pointer<T>();

        const auto& layout = mem.get_layout();

        auto pitches = layout.get_pitches();
        auto size = layout.size;
        if (layout.format == cldnn::format::byxf_af32)
        {
            size.feature[0] = ((size.feature[0] + 31) / 32) * 32;
        }
        if (layout.format == cldnn::format::fs_bs_yx_bsv4_fsv32)
        {
            size.feature[0] = ((size.feature[0] + 31) / 32) * 32;
            size.batch[0] = ((size.batch[0] + 3) / 4) * 4;
        }
        const unsigned int bp = pitches.batch[0];
        const unsigned int fp = pitches.feature[0];
        const unsigned int xp = pitches.spatial[0];
        const unsigned int yp = pitches.spatial[1];

        const auto bs = size.batch[0];
        const auto fs = size.feature[0];
        const auto xs = size.spatial[0] + layout.data_padding.lower_size().spatial[0] + layout.data_padding.upper_size().spatial[0]; // dump padded
        const auto ys = size.spatial[1] + layout.data_padding.lower_size().spatial[1] + layout.data_padding.upper_size().spatial[1]; // dump padded

        for (cldnn::tensor::value_type b = 0; b < bs; ++b)
        {
            for (cldnn::tensor::value_type f = 0; f < fs; ++f)
            {
                for (cldnn::tensor::value_type y = 0; y < ys; ++y)
                {
                    for (cldnn::tensor::value_type x = 0; x < xs; ++x)
                    {
                        unsigned int input_it = b*bp + f*fp + y*yp + x*xp;
                        streams[b][f] << convert_element(mem_ptr[input_it]) << " ";
                    }
                    streams[b][f] << std::endl;
                }
            }
        }
    }

    void logger::log_memory_to_file(const cldnn::memory& mem, std::string prefix, bool single_batch, cldnn::tensor::value_type batch_id, bool single_feature, cldnn::tensor::value_type feature_id)
    {        
        boost::filesystem::create_directories(dump_dir);
        const auto layout = mem.get_layout();
        auto batch = layout.size.batch[0];
        auto x = layout.size.spatial[0];
        auto feature = layout.size.feature[0];
        if (layout.format == cldnn::format::byxf_af32)
        {
            feature = ((feature + 31) / 32) * 32;
        }
        if (layout.format == cldnn::format::byx8_f4)
        {
            feature = ((feature + 3) / 4) * 4;
            x = ((x + 7) / 8) * 8;
        }
        if (layout.format == cldnn::format::fs_bs_yx_bsv4_fsv32)
        {
            feature = ((feature + 31) / 32) * 32;
            batch = ((batch + 3) / 4) * 4;
        }
        auto eng_type =  "gpu" ;
        std::vector<std::vector<std::stringstream>> streams(batch);
        for(cldnn::tensor::value_type b = 0; b < batch; b++)
        {
            streams[b].resize(feature);
        }

        if (layout.format == cldnn::format::fs_bs_yx_bsv4_fsv32)
        {
            if (layout.data_type == cldnn::data_types::f32)
                dump_fs_bs_yx_bsv4_fsv32<float>(mem, single_batch, batch_id, single_feature, feature_id, streams);
            else if (layout.data_type == cldnn::data_types::f16)
                dump_fs_bs_yx_bsv4_fsv32<half_t>(mem, single_batch, batch_id, single_feature, feature_id, streams);
            else if (layout.data_type == cldnn::data_types::i8)
                dump_fs_bs_yx_bsv4_fsv32<signed char>(mem, single_batch, batch_id, single_feature, feature_id, streams);
            else if (layout.data_type == cldnn::data_types::u8)
                dump_fs_bs_yx_bsv4_fsv32<unsigned char>(mem, single_batch, batch_id, single_feature, feature_id, streams);
            else
                dump_fs_bs_yx_bsv4_fsv32<char>(mem, single_batch, batch_id, single_feature, feature_id, streams);
        }
        else if (layout.format == cldnn::format::byx8_f4)
        {
            if (layout.data_type == cldnn::data_types::f32)
                dump_byx8_f4<float>(mem, single_batch, batch_id, single_feature, feature_id, streams);
            else if (layout.data_type == cldnn::data_types::f16)
                dump_byx8_f4<half_t>(mem, single_batch, batch_id, single_feature, feature_id, streams);
            else if (layout.data_type == cldnn::data_types::i8)
                dump_byx8_f4<signed char>(mem, single_batch, batch_id, single_feature, feature_id, streams);
            else if (layout.data_type == cldnn::data_types::u8)
                dump_byx8_f4<unsigned char>(mem, single_batch, batch_id, single_feature, feature_id, streams);
            else
                dump_byx8_f4<char>(mem, single_batch, batch_id, single_feature, feature_id, streams);
        }
        else
        {
            if (layout.data_type == cldnn::data_types::f32)
                dump<float>(mem, streams);
            else if (layout.data_type == cldnn::data_types::f16)
                dump<half_t>(mem, streams);
            else if (layout.data_type == cldnn::data_types::i8)
                dump<signed char>(mem, streams);
            else if (layout.data_type == cldnn::data_types::u8)
                dump<unsigned char>(mem, streams);
            else
                dump<char>(mem, streams);
        }

        if (layout.size.spatial[0] == 1 && layout.size.spatial[1] == 1)
        {
            for (cldnn::tensor::value_type b = 0; b < batch; b++)
            {
                if (!single_batch || b == batch_id)
                {
                    replace_forbidden_txt_characters(prefix);
                    std::string filename((dump_dir + "/" + prefix + "_" + eng_type + "_b" + std::to_string(b) + ".txt"));
                    std::ofstream file_stream = std::ofstream(filename, std::ios::out);
                    for (cldnn::tensor::value_type f = 0; f < feature; f++)
                    {
                        file_stream << streams[b][f].str();
                    }
                    file_stream.close();
                }

            }
        }
        else
        {
            for (cldnn::tensor::value_type b = 0; b < batch; b++)
            {
                for (cldnn::tensor::value_type f = 0; f < feature; f++)
                {
                    if ((!single_batch || b == batch_id) && (!single_feature || f == feature_id))
                    {
                        replace_forbidden_txt_characters(prefix);
                        std::string filename((dump_dir + "/" + prefix + "_" + eng_type + "_b" + std::to_string(b) + "_f" + std::to_string(f) + ".txt"));
                        std::ofstream file_stream = std::ofstream(filename, std::ios::out);
                        file_stream << streams[b][f].str();
                        file_stream.close();
                    }
                }
            }
        }
    }

    void logger::log_weights_to_file(const cldnn::memory& mem, std::string prefix)
    {
        boost::filesystem::create_directories(dump_dir);
        std::stringstream stream;

        if (mem.get_layout().data_type == cldnn::data_types::f32)
            dump<float>(mem, stream);
        else if (mem.get_layout().data_type == cldnn::data_types::f16)
            dump<half_t>(mem, stream);
        else if (mem.get_layout().data_type == cldnn::data_types::i8)
            dump<signed char>(mem, stream);
        else if (mem.get_layout().data_type == cldnn::data_types::u8)
            dump<unsigned char>(mem, stream);
        else
            dump<char>(mem, stream);

        std::string filename((dump_dir + "/" + prefix + ".txt"));
        std::ofstream file_stream = std::ofstream(filename, std::ios::out);
        file_stream << stream.str();
        file_stream.close();
    }
    std::string logger::create_graphs_dumps_dir(std::string& err_str)
    {
        auto dir = dump_dir + "/graphs";
        try
        {
            boost::filesystem::create_directories(dir);
            err_str.clear();
        }
        catch (boost::filesystem::filesystem_error const& err)
        {
           err_str = err.what();
        }

        return dir;
    }
    std::string logger::get_dumps_dir()
    {
        boost::filesystem::create_directories(dump_dir);
        return dump_dir;
    }
    std::string logger::create_sources_dumps_dir(std::string& err_str)
    {
        auto dir = dump_dir + "/sources";
        try
        {
            boost::filesystem::create_directories(dir);
            err_str.clear();
        }
        catch (boost::filesystem::filesystem_error const& err)
        {
            err_str = err.what();
        }

        return dir;
    }
    void logger::replace_forbidden_txt_characters(std::string& prefix)
    {
        std::string forbidden_characters = R"(<>/\?*|:")";
        std::size_t found = prefix.find_first_of(forbidden_characters);
        while (found != std::string::npos)
        {
            prefix[found] = '_';
            found = prefix.find_first_of(forbidden_characters, found + 1);
        }
    }
}
