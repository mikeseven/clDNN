#include "gpu_compiler.h"
#include "../ocl_toolkit.h"
#include <iostream>
#include <sstream>

namespace neural { namespace gpu { namespace cache {

namespace {
    
code inject_jit(const jit& compile_options, const code& code)
{
    return compile_options + code; //TODO temporary untill we merge proper mechanism
}

}

binary_data gpu_compiler::compile(context* context, const jit& compile_options, const code& code_src)
{
#if 0
    context;
    return binary_code(inject_jit(compile_options, code)); //TODO temporary untill we merge proper mechanism
#else
    auto& clContext = context->context();
    auto& clDevice = context->device();
    auto& program = context->program();
    code source = inject_jit(compile_options, code_src);

    cl_int status = CL_SUCCESS;
    program = cl::Program(clContext, source, false, &status);

    if (status == CL_SUCCESS)
    {
        try
        {
            status = program.build(cl::vector<cl::Device>(1, clDevice), "");
        }
        catch (const std::exception &)
        {
            status = CL_BUILD_PROGRAM_FAILURE;
        }

        if (CL_SUCCESS != status)
        {
            if (CL_BUILD_PROGRAM_FAILURE == status)
            {
                cl_int getLogStatus;
                std::string buildLog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(clDevice, &getLogStatus);
                std::cout << buildLog + "\n";

                std::istringstream stream(source);
                std::string line;
                unsigned int lineNumber = 1;
                while (std::getline(stream, line))
                {
                    std:: cout << lineNumber << ": " << line << "\n";
                    lineNumber++;
                }
            }
        }
    }

    size_t programSize = 0;

    if (status == CL_SUCCESS)
    {
        status = clGetProgramInfo(program.get(), CL_PROGRAM_BINARY_SIZES, sizeof(programSize), &programSize, nullptr);
    }

    binary_data binary;

    if (status == CL_SUCCESS)
    {
        binary.resize(programSize);
        char* pBinaries = const_cast<char*>(binary.data());
        status = clGetProgramInfo(program.get(), CL_PROGRAM_BINARIES, sizeof(char *), &pBinaries, nullptr);
    }

    if (status == CL_SUCCESS)
    {
        return binary;
    }

    throw std::runtime_error("Error during compilation");
#endif
}

} } }
