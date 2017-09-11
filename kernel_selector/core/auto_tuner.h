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

#pragma once

#include <atomic>
#include <mutex>
#include "kernel_selector_common.h"

namespace KernelSelector 
{
    class AutoTuner
    {
        struct TuningData
        {
            std::map<std::string, std::tuple<std::string, int>> hashToKernelConfig;
        };

    public:
        AutoTuner() = default;
        std::tuple<std::string, int> LoadKernel(const TuningMode tuningMode, const std::string& tuningFilePath, const std::string& deviceID, const std::string& driverVersion, const std::string& hostVersion, const std::string& hash);
        void StoreKernel(const std::string& tuningFilePath, const std::string& hash, const std::string& implementationName, const int tuneIndex);

    private:    
        std::map<std::string, TuningData> tuningCache; // Tuning file name -> kernel/config per hash (hash -> [implementation name, tuning index])
        std::mutex mutex; // Mutex to synchronize cache updates
    };
}