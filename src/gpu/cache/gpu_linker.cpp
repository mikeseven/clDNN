#include "gpu_linker.h"

namespace neural { namespace gpu { namespace manager {

gpu_program gpu_linker::link(context * context, const std::vector<cache::binary_data>& kernels)
{
	auto& clContext = context->context();
	auto& device = context->device();
	auto& program = context->program();
	cl::vector<cl::vector<unsigned char>> binaries;
	for (const auto & k : kernels)
	{
		binaries.emplace_back(k.begin(), k.end());
	}
	program = cl::Program(clContext, cl::vector<cl::Device>(1, device), binaries);
	program.build();
    return program;
}

} } }
