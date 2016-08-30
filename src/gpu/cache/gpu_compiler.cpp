#include "gpu_compiler.h"
#include "../ocl_toolkit.h"
#include <iostream>
#include <sstream>
#include <assert.h>

namespace neural { namespace gpu { namespace cache {

namespace {
    
code inject_jit(const jit& compile_options, const code& code)
{
    return compile_options + code; //TODO temporary untill we merge proper mechanism
}

}

binary_data gpu_compiler::compile(context* context, const jit& compile_options, const code& code_src) // throws cl::BuildError
{
    auto& clContext = context->context();
    auto& program = context->program();
    code source = inject_jit(compile_options, code_src);
    program = cl::Program(clContext, source, false);
	program.compile();
    auto binaries = program.getInfo<CL_PROGRAM_BINARIES>();
	assert(binaries.size() == 1 && "There should be only one binary");
	return binary_data(binaries[0].begin(), binaries[0].end());
}

} } }
