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
#include "api/instrumentation.h"

#include <fstream>
#include <iomanip>
#include <string>
#include <vector>

#include <boost/filesystem.hpp>


namespace neural {
    namespace instrumentation {
        // initalize dumping directory for whole run
        const std::string logger::dump_dir = std::to_string(std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()));

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
                    exp = (exp - 1U) << 23;
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

        float convert_element(float f)
        {
            return f;
        }

        float convert_element(half_t h)
        {
            return convert_half_to_float(h);
        }

        template<typename elemType>
        void dump_byxf(const memory& mem, bool single_batch, uint32_t batch_id, bool single_feature, uint32_t feature_id, std::vector<std::vector<std::stringstream>> & streams)
        {
            auto mem_arg = mem.argument;
            auto mem_ptr = mem.pointer<elemType>();

            unsigned int input_it = 0;
            for (uint32_t b = 0; b < mem_arg.size.batch[0]; b++)
            {
                for (uint32_t y = 0; y < mem_arg.size.spatial[1]; y++)
                {
                    for (uint32_t x = 0; x < mem_arg.size.spatial[0]; x++)
                    {
                        for (uint32_t f = 0; f < mem_arg.size.feature[0]; f++)
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
        void dump_bfyx(const memory& mem, bool single_batch, uint32_t batch_id, bool single_feature, uint32_t feature_id, std::vector<std::vector<std::stringstream>> & streams)
        {
            auto mem_arg = mem.argument;
            auto mem_ptr = mem.pointer<elemType>();

            unsigned int input_it = 0;
            for (uint32_t b = 0; b < mem_arg.size.batch[0]; b++)
            {
                for (uint32_t f = 0; f < mem_arg.size.feature[0]; f++)
                {
                    for (uint32_t y = 0; y < mem_arg.size.spatial[1]; y++)
                    {
                        for (uint32_t x = 0; x < mem_arg.size.spatial[0]; x++)
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
        void dump_yxfb(const memory& mem, bool single_batch, uint32_t batch_id, bool single_feature, uint32_t feature_id, std::vector<std::vector<std::stringstream>> & streams)
        {
            auto mem_arg = mem.argument;
            auto mem_ptr = mem.pointer<elemType>();

            unsigned int input_it = 0;
            for (uint32_t y = 0; y < mem_arg.size.spatial[1]; y++)
            {
                for (uint32_t x = 0; x < mem_arg.size.spatial[0]; x++)
                {
                    for (uint32_t f = 0; f < mem_arg.size.feature[0]; f++)
                    {
                        for (uint32_t b = 0; b < mem_arg.size.batch[0]; b++)
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
        void dump_xb(const memory& mem, bool single_batch, uint32_t batch_id, std::vector<std::vector<std::stringstream>> & streams)
        {
            auto mem_arg = mem.argument;
            auto mem_ptr = mem.pointer<elemType>();

            unsigned int input_it = 0;
            for (uint32_t x = 0; x < mem_arg.size.spatial[0]; x++)
            {
                for (uint32_t b = 0; b < mem_arg.size.batch[0]; b++)
                {
                    if (!single_batch || b == batch_id)
                    {
                        streams[b][0] << convert_element(mem_ptr[input_it]) << std::endl;
                    }
                    input_it++;
                }
            }
        }

        void logger::log_memory_to_file(const primitive& mem, std::string prefix, bool single_batch, uint32_t batch_id, bool single_feature, uint32_t feature_id)
        {
            const auto& mem_prim = get_memory_primitive(mem);
            auto mem_arg = mem_prim.argument;

            boost::filesystem::create_directories(dump_dir);
            auto batch = mem_arg.size.batch[0];
            auto feature = mem_arg.size.feature[0];
            auto eng_type =  "gpu" ;
            std::vector<std::vector<std::stringstream>> streams(batch);
            for(uint32_t b = 0; b < batch; b++)
            {
                streams[b].resize(feature);
            }

            switch (mem_arg.format)
            {
            // FP32 (float)
            case memory::format::byxf_f32:
                {
                    dump_byxf<float>(mem_prim, single_batch, batch_id, single_feature, feature_id, streams);
                }
                break;
            case memory::format::bfyx_f32:
                {
                    dump_bfyx<float>(mem_prim, single_batch, batch_id, single_feature, feature_id, streams);
                }
                break;
            case memory::format::yxfb_f32:
                {
                    dump_yxfb<float>(mem_prim, single_batch, batch_id, single_feature, feature_id, streams);
                }
                break;
            case memory::format::xb_f32:
                {
                    dump_xb<float>(mem_prim, single_batch, batch_id, streams);
                }
                break;

            // FP16 (half)
            case memory::format::byxf_f16:
                {
                    dump_byxf<half_t>(mem_prim, single_batch, batch_id, single_feature, feature_id, streams);
                }
                break;
            case memory::format::bfyx_f16:
                {
                    dump_bfyx<half_t>(mem_prim, single_batch, batch_id, single_feature, feature_id, streams);
                }
                break;
            case memory::format::yxfb_f16:
                {
                    dump_yxfb<half_t>(mem_prim, single_batch, batch_id, single_feature, feature_id, streams);
                }
                break;
            case memory::format::xb_f16:
                {
                    dump_xb<half_t>(mem_prim, single_batch, batch_id, streams);
                }
            break;

            default:
                throw std::runtime_error("format not implemented yet");
            }

            for (uint32_t b = 0; b < batch; b++)
                for (uint32_t f = 0; f < feature; f++)
                {
                    if ((!single_batch || b == batch_id) && (!single_feature || f == feature_id))
                    {
                        std::string filename((dump_dir + "/" + prefix + "_" + eng_type + "_b" + std::to_string(b) + "_f" + std::to_string(f) + ".txt"));
                        std::ofstream file_stream = std::ofstream(filename, std::ios::out);
                        file_stream << streams[b][f].str();
                        file_stream.close();
                    }
                }
        }
    }
}
