#include <algorithm>
#include <string>
#include "vxa_common.h"

#ifdef WIN32
#include <stdlib.h>
#endif

namespace clDNN
{
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
            bEnvSupported = GetBoolEnv("USE_CL_DNN", bEnvSupported);
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

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // UseReferenceKernel
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    bool UseReferenceKernel()
    {
        static bool bUseReferenceKernels = false;
        static bool bInitialized = false;

        if (!bInitialized)
        {
            bInitialized = true;
            bUseReferenceKernels = GetBoolEnv("USE_CL_DNN_REFERENCE_KERNELS", bUseReferenceKernels);
        }

        return bUseReferenceKernels;
    }
}