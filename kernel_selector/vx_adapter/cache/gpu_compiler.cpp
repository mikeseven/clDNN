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
#include "gpu_compiler.h"
#include "vxa_ocl_toolkit.h"
#include <iostream>
#include <sstream>
#include <assert.h>

namespace clDNN { namespace gpu { namespace cache {

binary_data gpu_compiler::compile(context_device& context, const code& program_str, const compile_options& options) // throws cl::BuildError
{
    auto& clContext = context.context;
    auto& clDevice = context.device;

    cl_int status = CL_SUCCESS;
    cl::Program program = cl::Program(clContext, program_str, false, &status);

    if (status == CL_SUCCESS)
    {
        try
        {
            status = program.build(cl::vector<cl::Device>(1, clDevice), options.c_str());
        }
        catch (const std::exception &)
        {
            status = CL_BUILD_PROGRAM_FAILURE;
        }

        if (CL_SUCCESS != status)
        {
#ifndef NDEBUG
            if (CL_BUILD_PROGRAM_FAILURE == status)
            {
                cl_int getLogStatus;
                std::string buildLog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(clDevice, &getLogStatus);
                std::cout << buildLog + "\n";

                std::istringstream stream(program_str);
                std::string line;
                unsigned int lineNumber = 1;
                while (std::getline(stream, line))
                {
                    std:: cout << lineNumber << ": " << line << "\n";
                    lineNumber++;
                }

                std::cout << buildLog + "\n";
            }
#endif
        }
    }

    auto binaries = program.getInfo<CL_PROGRAM_BINARIES>();
	assert(binaries.size() == 1 && "There should be only one binary");
	return binary_data(binaries[0].begin(), binaries[0].end());
}

} } }
