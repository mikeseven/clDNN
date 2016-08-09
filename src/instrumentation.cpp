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
#include <sstream>
#include <iomanip>
#include <nmmintrin.h>
#include <array>
#include <vector>
#include <direct.h>
#include <ctime>
#include <string>
namespace neural {
    namespace instrumentation {
        void logger::log_memory_to_file(const primitive& mem, std::string prefix)
        {
            auto mem_arg = mem.id() == type_id<const memory>()->id ? mem.as<const memory&>().argument : mem.output[0].as<const memory&>().argument;
            auto mem_ptr = mem.id() == type_id<const memory>()->id ? mem.as<const memory&>().pointer<float>() : mem.output[0].as<const memory&>().pointer<float>();
            time_t rawtime;
            char buf[85];
            time(&rawtime);
            struct tm timebuf;
            localtime_s(&timebuf, &rawtime);
            strftime(buf, 80, "./dumps%d_%m_%Y_%I_%M_%S", &timebuf);
            _mkdir(buf);
            auto batch = mem_arg.size.batch[0];
            auto feature = mem_arg.size.feature[0];
            auto sizex = mem_arg.size.spatial[0];
            auto sizey = mem_arg.size.spatial[1];
            std::vector<std::vector<std::ofstream>> files_handels(batch);
            std::string dirpath(buf);

            for (uint32_t i = 0; i < batch; i++)
                for (uint32_t j = 0; j < feature;j++)
                {
                    std::string filename((dirpath + "/" + prefix + "_b" + std::to_string(i) + "_f" + std::to_string(j) + ".txt"));
                    files_handels[i].push_back(std::ofstream(filename, std::ios::out));
                }
            int input_it = 0;
            switch (mem_arg.format)
            {
            case memory::format::yxfb_f32:
                for (uint32_t y = 0; y < sizey;y++)
                {
                    for (uint32_t x = 0; x < sizex;x++)
                        for (uint32_t feature_it = 0; feature_it < feature; feature_it++)
                            for (uint32_t batch_it = 0; batch_it < batch; batch_it++)
                            {
                                files_handels[batch_it][feature_it] << mem_ptr[input_it++] << " ";
                                if (x == sizex - 1)
                                    files_handels[batch_it][feature_it] << std::endl;
                            }
                }
                break;
            case memory::format::xb_f32:
                for (uint32_t x = 0; x < sizex;x++)
                    for (uint32_t batch_it = 0; batch_it < batch; batch_it++)
                        files_handels[batch_it][0] << mem_ptr[input_it++] << std::endl;
                break;
            default:
                throw std::runtime_error("format not implemented yet");
                break;
            }


            for (uint32_t i = 0; i < batch; i++)
                for (uint32_t j = 0; j < feature;j++)
                    files_handels[i][j].close();
        }

        std::string timer::time_diff_string(uint64_t t_diff) {
            std::ostringstream temp;
            if (t_diff > 0) {
                double t_d = static_cast<double>(t_diff);
                static const std::string units[] = { "ns", "us", "ms", "s" };
                uint8_t     index = 0;

                while (t_d > 1000 && index < 3) {
                    t_d /= 1000;
                    ++index;
                };
                temp << std::setprecision(3) << std::fixed << t_d << " " + units[index];
            }
            else
            {
                temp << "Error: time interval is less than zero";
            }
            return temp.str();
        }
    }
}