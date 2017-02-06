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

#include <algorithm>
#include <string>
#include "vxa_common.h"

#ifdef WIN32
#include <stdlib.h>
#endif

// #define ENABLE_SUPPORTED_ENV

namespace clDNN
{
#ifdef ENABLE_SUPPORTED_ENV
    bool GetBoolEnv(const char* varName, bool defaultVal)
    {
        std::string str;
#ifdef WIN32
        char* env = nullptr;
        std::size_t len = 0;
        errno_t err = _dupenv_s(&env, &len, varName);
        if (err == 0)
        {
            if (env != nullptr)
            {
                str = std::string(env);
            }
            free(env);
        }
#else
        const char *env = getenv(varName);
        if (env)
        {
            str = std::string(env);
        }
#endif

        bool res = defaultVal;

        std::transform(str.begin(), str.end(), str.begin(), ::tolower);
        if (str == "y" || str == "yes")
        {
            res = true;
        }
        else if (str == "n" || str == "no")
        {
            res = false;
        }

        return res;
    }
#endif

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // IsSupported
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    bool IsSupported(const Params& params)
    {
        static bool bEnvSupported = true;
        static bool bInitialized = false;

        if (!bInitialized)
        {
            bInitialized = true;
#ifdef ENABLE_SUPPORTED_ENV
            bEnvSupported = GetBoolEnv("USE_CL_DNN", bEnvSupported);
#endif
        }

        bool bSupportCurrentSettings = true;

        const BaseParams& baseParams = static_cast<const BaseParams&>(params);

        if (baseParams.inputType != Datatype::F16 &&
            baseParams.inputType != Datatype::F32)
        {
            bSupportCurrentSettings = false;
        }

        return bEnvSupported && bSupportCurrentSettings;
    }
}