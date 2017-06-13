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

#include <stdio.h>
#include <iostream>
#include <sstream>
#include "kernel_base.h"
#include "kernel_selector.h"
#include "kernel_selector_params.h"
#include "actual_kernels/convolution/convolution_kernel_selector.h"

using namespace KernelSelctor;

#define  RUN_MODE 1

#define CL_CHECK(status, message)             \
do                                            \
{                                             \
   if (CL_SUCCESS != status)                  \
   {                                          \
      printf("%s\n", message);                \
      return 0.f;                             \
   }                                          \
}                                             \
while (0)


#if defined(_MSC_VER) || defined(__MINGW32__)
  #define ST_LLX "%Ix"
#else
  #define ST_LLX "%zx"
#endif

std::vector<ConvolutionParams> params_vec;
void InitConvParams();

class CLRunner
{
    kernel_cache m_binaryManager;
    std::shared_ptr<gpu_toolkit> gpu_context;

public:

    CLRunner()
    {
        neural::gpu::configuration cfg;
        cfg.enable_profiling = true;
        gpu_context = std::make_shared<gpu_toolkit>(cfg);
    }

    float run(const KernelData& kernelData)
    {
        cl_int status = CL_SUCCESS;
        const clKernelData& clData = kernelData.kernels[0];
        binary_data binary = clData.GetBinary(gpu_context.get(), m_binaryManager);
        cl::Program::Binaries binaries;
        binaries.push_back(binary);
        auto devices = std::vector<cl::Device>(1, gpu_context->device());
        auto& clContext = gpu_context->context();
        cl::Program program(gpu_context->context(), devices, binaries, NULL, &status);

        CL_CHECK(status, "Error: cannot create program");

        cl::Kernel clKernel(program, clData.kernel_string.entry_point.c_str(), &status);
        CL_CHECK(status, "Error: cannot create kernel");

        auto& newParams = *static_cast<ConvolutionParams*>(kernelData.params.get());
        std::size_t weightsSize =
            kernelData.weights_reorder_params.engine != WeightsReorderParams::Engine::NONE ?
            kernelData.weights_reorder_params.new_buffer_size :
            newParams.convParams.filterSize.x * newParams.convParams.filterSize.y * newParams.inDims.z * newParams.outDims.z;
        cl::Buffer input(clContext, CL_MEM_READ_WRITE, newParams.inDesc.Size(), nullptr, &status);
        cl::Buffer output(clContext, CL_MEM_READ_WRITE, newParams.outDesc.Size(), nullptr, &status);
        cl::Buffer weights(clContext, CL_MEM_READ_WRITE, weightsSize, nullptr, &status);
        cl::Buffer bias(clContext, CL_MEM_READ_WRITE, newParams.outDims.z, nullptr, &status);

        if (!clData.args_desc.SetArguments(clKernel, { &input }, &output, &weights, &bias, nullptr))
        {
            printf("Error: setting args\n");
            return 0.f;
        }

        const uint warm_up = 3;
        const uint iteration = 100;
        std::vector<cl::Event> events;
        events.resize(iteration);

        for (uint i = 0; i < warm_up; i++)
        {
            status = gpu_context->queue().enqueueNDRangeKernel(
                clKernel,
                cl::NullRange,
                clData.work_groups.global,
                clData.work_groups.local,
                nullptr);
            CL_CHECK(status, "Error: enqueue failed");
        }

        for (auto& event : events)
        {
            status = gpu_context->queue().enqueueNDRangeKernel(
                clKernel,
                cl::NullRange,
                clData.work_groups.global,
                clData.work_groups.local,
                nullptr,
                &event);
            CL_CHECK(status, "Error: enqueue failed");
        }
        status = gpu_context->queue().finish();
        CL_CHECK(status, "Error: finish failed");

        double avg_time = 0.0;

        for (auto& e : events)
        {
            cl_int status = e.wait();
            CL_CHECK(status, "clWaitForEvents failed");
            cl_double start = (cl_double)e.getProfilingInfo<CL_PROFILING_COMMAND_START>(&status);
            CL_CHECK(status, "getProfilingInfo<CL_PROFILING_COMMAND_START> failed");
            cl_double end = (cl_double)e.getProfilingInfo<CL_PROFILING_COMMAND_END>(&status);
            CL_CHECK(status, "getProfilingInfo<CL_PROFILING_COMMAND_END> failed");
            cl_double total_time = (end - start) / 1000000.0;
            avg_time += total_time;
        }

        avg_time /= iteration;

        return (float)avg_time;
    }
};

class ConvolutionCostModel : public ConvolutionKernelSelctor
{
    CLRunner cl_runner;

    std::vector<uint> GetKernels()
    {
        std::vector<uint> optional;
        std::vector<uint> force;

        for (std::size_t i = 0; i < implementations.size(); i++)
        {
            const auto& implementation = implementations[i];
            const auto& it = force_kernels.find(implementation->GetName());
            if (it != force_kernels.end())
            {
                if (it->second == true)
                {
                    //std::cout << "Force: " << it->first << std::endl;
                    force.push_back((uint)i);
                }
                else
                {
                    //std::cout << "Deny: " << it->first << std::endl;
                }
            }
            else
            {
                optional.push_back((uint)i);
            }
        }

        if (force.size())
        {
            return force;
        }

        return optional;
    }

public:
    void run()
    {
#if 0
        ConvolutionParams params;
        params.inputType = Datatype::F16;
        params.inputLayout = DataLayout::bfyx;
        params.outputLayout = DataLayout::bfyx;
        params.activationFunc = ActivationFunction::NONE;
#if 1
        params.inDims = { 227, 227, 3, 1 };
        params.inDesc = { 0, params.inDims };
        params.outDims = { 55, 55, 96, 1 };
        params.outDesc = { 0, params.outDims };
        params.convParams.filterSize = { 11,11 };
        params.convParams.padding = { 0, 0 };
        params.convParams.stride = { 4, 4 };
#else
        params.inDims = { 27, 27, 48, 1 };
        params.inDesc = { 0, params.inDims };
        params.outDims = { 27, 27, 128, 1 };
        params.outDesc = { 0, params.outDims };
        params.convParams.filterSize = { 5, 5 };
        params.convParams.padding = { 2, 2 };
        params.convParams.stride = { 1, 1 };
#endif
#endif
        InitConvParams();

        for (uint i : GetKernels())
        {
            const auto& impl = implementations[i];
            const auto& kernel = *impl.get();

#if RUN_MODE == 1
            printf("%s\n", kernel.GetName().c_str());
            printf("static std::map<std::size_t, float> known_params_time = {\n");
#endif

            for (const auto& params : params_vec)
            {
                std::string param_str = params.to_string();
                //printf("%s\n", param_str.c_str());

                float avg_time = NOT_SUPPORTED;

                if (kernel.GetSupportedKey().Support(params.GetParamsKey()))
                {
                    ConvolutionOptionalParams optParams;
                    optParams.allow_padding = true;

                    KernelsData kernelsData = kernel.GetKernelsData(params, optParams);
                    if (kernelsData.size() && kernelsData[0].kernels.size())
                    {
                        avg_time = cl_runner.run(kernelsData[0]);
#if RUN_MODE == 2
                        float diff = abs((float)(avg_time - kernelsData[0].estimated_time));
                        if (diff > 0.1)
                        {
                            printf("ERROR: bad value (%f, %f, %f) - %s\n", avg_time, kernelsData[0].estimated_time, diff, param_str.c_str());
                        }
#endif
                    }
                    //printf("%s - avg time = %lf\n", kernel.GetName().c_str(), avg_time);
                }
#if RUN_MODE == 1
                if (avg_time == NOT_SUPPORTED)
                {
                    printf("{0x" ST_LLX ", NOT_SUPPORTED}, // %s\n", std::hash<std::string>()(param_str), param_str.c_str());
                }
                else
                {
                    printf("{0x" ST_LLX ", %ff}, // %s\n", std::hash<std::string>()(param_str), avg_time, param_str.c_str());
                }
#endif
            }
#if RUN_MODE == 1
            printf("};\n");
#endif
        }
    }
};

int main( int argc, char* argv[ ] )
{
    ConvolutionCostModel().run();
    return 0;
}

void InitConvParams()
{

    ConvolutionParams params0;
    params0.inputType = Datatype::F16;
    params0.inputLayout = DataLayout::bfyx;
    params0.outputLayout = DataLayout::bfyx;
    params0.activationFunc = ActivationFunction::RELU;
    params0.nlParams = { 0, 0 };
    params0.inDims = { 227, 227, 3, 1 };
    params0.inDesc = { 0,{ 228, 51756, 155268, 155268 }, false };
    params0.outDims = { 55, 55, 96, 1 };
    params0.outDesc = { 0,{ 56, 3080, 295680, 295680 }, false };
    params0.convParams.filterSize = { 11, 11 };
    params0.convParams.padding = { 0, 0 };
    params0.convParams.stride = { 4, 4 };

    ConvolutionParams params1;
    params1.inputType = Datatype::F16;
    params1.inputLayout = DataLayout::bfyx;
    params1.outputLayout = DataLayout::bfyx;
    params1.activationFunc = ActivationFunction::RELU;
    params1.nlParams = { 0, 0 };
    params1.inDims = { 27, 27, 48, 1 };
    params1.inDesc = { 0,{ 28, 756, 36288, 36288 }, false };
    params1.outDims = { 27, 27, 128, 1 };
    params1.outDesc = { 0,{ 28, 756, 96768, 96768 }, false };
    params1.convParams.filterSize = { 5, 5 };
    params1.convParams.padding = { 2, 2 };
    params1.convParams.stride = { 1, 1 };

    ConvolutionParams params2;
    params2.inputType = Datatype::F16;
    params2.inputLayout = DataLayout::bfyx;
    params2.outputLayout = DataLayout::bfyx;
    params2.activationFunc = ActivationFunction::RELU;
    params2.nlParams = { 0, 0 };
    params2.inDims = { 27, 27, 48, 1 };
    params2.inDesc = { 66,{ 32, 992, 47616, 47616 }, false };
    params2.outDims = { 27, 27, 128, 1 };
    params2.outDesc = { 0,{ 28, 756, 96768, 96768 }, false };
    params2.convParams.filterSize = { 5, 5 };
    params2.convParams.padding = { 2, 2 };
    params2.convParams.stride = { 1, 1 };

    ConvolutionParams params3;
    params3.inputType = Datatype::F16;
    params3.inputLayout = DataLayout::bfyx;
    params3.outputLayout = DataLayout::bfyx;
    params3.activationFunc = ActivationFunction::RELU;
    params3.nlParams = { 0, 0 };
    params3.inDims = { 13, 13, 256, 1 };
    params3.inDesc = { 0,{ 14, 182, 46592, 46592 }, false };
    params3.outDims = { 13, 13, 384, 1 };
    params3.outDesc = { 0,{ 14, 182, 69888, 69888 }, false };
    params3.convParams.filterSize = { 3, 3 };
    params3.convParams.padding = { 1, 1 };
    params3.convParams.stride = { 1, 1 };

    ConvolutionParams params4;
    params4.inputType = Datatype::F16;
    params4.inputLayout = DataLayout::bfyx;
    params4.outputLayout = DataLayout::bfyx;
    params4.activationFunc = ActivationFunction::RELU;
    params4.nlParams = { 0, 0 };
    params4.inDims = { 13, 13, 256, 1 };
    params4.inDesc = { 17,{ 16, 240, 61440, 61440 }, false };
    params4.outDims = { 13, 13, 384, 1 };
    params4.outDesc = { 0,{ 14, 182, 69888, 69888 }, false };
    params4.convParams.filterSize = { 3, 3 };
    params4.convParams.padding = { 1, 1 };
    params4.convParams.stride = { 1, 1 };

    ConvolutionParams params5;
    params5.inputType = Datatype::F16;
    params5.inputLayout = DataLayout::bfyx;
    params5.outputLayout = DataLayout::bfyx;
    params5.activationFunc = ActivationFunction::RELU;
    params5.nlParams = { 0, 0 };
    params5.inDims = { 13, 13, 192, 1 };
    params5.inDesc = { 0,{ 14, 182, 34944, 34944 }, false };
    params5.outDims = { 13, 13, 192, 1 };
    params5.outDesc = { 0,{ 14, 182, 34944, 34944 }, false };
    params5.convParams.filterSize = { 3, 3 };
    params5.convParams.padding = { 1, 1 };
    params5.convParams.stride = { 1, 1 };

    ConvolutionParams params6;
    params6.inputType = Datatype::F16;
    params6.inputLayout = DataLayout::bfyx;
    params6.outputLayout = DataLayout::bfyx;
    params6.activationFunc = ActivationFunction::RELU;
    params6.nlParams = { 0, 0 };
    params6.inDims = { 13, 13, 192, 1 };
    params6.inDesc = { 17,{ 16, 240, 46080, 46080 }, false };
    params6.outDims = { 13, 13, 192, 1 };
    params6.outDesc = { 0,{ 14, 182, 34944, 34944 }, false };
    params6.convParams.filterSize = { 3, 3 };
    params6.convParams.padding = { 1, 1 };
    params6.convParams.stride = { 1, 1 };

    ConvolutionParams params7;
    params7.inputType = Datatype::F16;
    params7.inputLayout = DataLayout::bfyx;
    params7.outputLayout = DataLayout::bfyx;
    params7.activationFunc = ActivationFunction::RELU;
    params7.nlParams = { 0, 0 };
    params7.inDims = { 13, 13, 192, 1 };
    params7.inDesc = { 0,{ 14, 182, 34944, 34944 }, false };
    params7.outDims = { 13, 13, 128, 1 };
    params7.outDesc = { 0,{ 14, 182, 23296, 23296 }, false };
    params7.convParams.filterSize = { 3, 3 };
    params7.convParams.padding = { 1, 1 };
    params7.convParams.stride = { 1, 1 };

    ConvolutionParams params8;
    params8.inputType = Datatype::F16;
    params8.inputLayout = DataLayout::bfyx;
    params8.outputLayout = DataLayout::bfyx;
    params8.activationFunc = ActivationFunction::RELU;
    params8.nlParams = { 0, 0 };
    params8.inDims = { 13, 13, 192, 1 };
    params8.inDesc = { 17,{ 16, 240, 46080, 46080 }, false };
    params8.outDims = { 13, 13, 128, 1 };
    params8.outDesc = { 0,{ 14, 182, 23296, 23296 }, false };
    params8.convParams.filterSize = { 3, 3 };
    params8.convParams.padding = { 1, 1 };
    params8.convParams.stride = { 1, 1 };

    ConvolutionParams params9;
    params9.inputType = Datatype::F32;
    params9.inputLayout = DataLayout::bfyx;
    params9.outputLayout = DataLayout::bfyx;
    params9.activationFunc = ActivationFunction::RELU;
    params9.nlParams = { 0, 0 };
    params9.inDims = { 227, 227, 3, 1 };
    params9.inDesc = { 0,{ 227, 51529, 154587, 154587 }, false };
    params9.outDims = { 55, 55, 96, 1 };
    params9.outDesc = { 0,{ 55, 3025, 290400, 290400 }, false };
    params9.convParams.filterSize = { 11, 11 };
    params9.convParams.padding = { 0, 0 };
    params9.convParams.stride = { 4, 4 };

    ConvolutionParams params10;
    params10.inputType = Datatype::F32;
    params10.inputLayout = DataLayout::bfyx;
    params10.outputLayout = DataLayout::bfyx;
    params10.activationFunc = ActivationFunction::RELU;
    params10.nlParams = { 0, 0 };
    params10.inDims = { 27, 27, 48, 1 };
    params10.inDesc = { 0,{ 27, 729, 34992, 34992 }, false };
    params10.outDims = { 27, 27, 128, 1 };
    params10.outDesc = { 0,{ 27, 729, 93312, 93312 }, false };
    params10.convParams.filterSize = { 5, 5 };
    params10.convParams.padding = { 2, 2 };
    params10.convParams.stride = { 1, 1 };

    ConvolutionParams params11;
    params11.inputType = Datatype::F32;
    params11.inputLayout = DataLayout::bfyx;
    params11.outputLayout = DataLayout::bfyx;
    params11.activationFunc = ActivationFunction::RELU;
    params11.nlParams = { 0, 0 };
    params11.inDims = { 27, 27, 48, 1 };
    params11.inDesc = { 64,{ 31, 961, 46128, 46128 }, false };
    params11.outDims = { 27, 27, 128, 1 };
    params11.outDesc = { 0,{ 27, 729, 93312, 93312 }, false };
    params11.convParams.filterSize = { 5, 5 };
    params11.convParams.padding = { 2, 2 };
    params11.convParams.stride = { 1, 1 };

    ConvolutionParams params12;
    params12.inputType = Datatype::F32;
    params12.inputLayout = DataLayout::bfyx;
    params12.outputLayout = DataLayout::bfyx;
    params12.activationFunc = ActivationFunction::RELU;
    params12.nlParams = { 0, 0 };
    params12.inDims = { 13, 13, 256, 1 };
    params12.inDesc = { 0,{ 13, 169, 43264, 43264 }, false };
    params12.outDims = { 13, 13, 384, 1 };
    params12.outDesc = { 0,{ 13, 169, 64896, 64896 }, false };
    params12.convParams.filterSize = { 3, 3 };
    params12.convParams.padding = { 1, 1 };
    params12.convParams.stride = { 1, 1 };

    ConvolutionParams params13;
    params13.inputType = Datatype::F32;
    params13.inputLayout = DataLayout::bfyx;
    params13.outputLayout = DataLayout::bfyx;
    params13.activationFunc = ActivationFunction::RELU;
    params13.nlParams = { 0, 0 };
    params13.inDims = { 13, 13, 256, 1 };
    params13.inDesc = { 16,{ 15, 225, 57600, 57600 }, false };
    params13.outDims = { 13, 13, 384, 1 };
    params13.outDesc = { 0,{ 13, 169, 64896, 64896 }, false };
    params13.convParams.filterSize = { 3, 3 };
    params13.convParams.padding = { 1, 1 };
    params13.convParams.stride = { 1, 1 };

    ConvolutionParams params14;
    params14.inputType = Datatype::F32;
    params14.inputLayout = DataLayout::bfyx;
    params14.outputLayout = DataLayout::bfyx;
    params14.activationFunc = ActivationFunction::RELU;
    params14.nlParams = { 0, 0 };
    params14.inDims = { 13, 13, 192, 1 };
    params14.inDesc = { 0,{ 13, 169, 32448, 32448 }, false };
    params14.outDims = { 13, 13, 192, 1 };
    params14.outDesc = { 0,{ 13, 169, 32448, 32448 }, false };
    params14.convParams.filterSize = { 3, 3 };
    params14.convParams.padding = { 1, 1 };
    params14.convParams.stride = { 1, 1 };

    ConvolutionParams params15;
    params15.inputType = Datatype::F32;
    params15.inputLayout = DataLayout::bfyx;
    params15.outputLayout = DataLayout::bfyx;
    params15.activationFunc = ActivationFunction::RELU;
    params15.nlParams = { 0, 0 };
    params15.inDims = { 13, 13, 192, 1 };
    params15.inDesc = { 16,{ 15, 225, 43200, 43200 }, false };
    params15.outDims = { 13, 13, 192, 1 };
    params15.outDesc = { 0,{ 13, 169, 32448, 32448 }, false };
    params15.convParams.filterSize = { 3, 3 };
    params15.convParams.padding = { 1, 1 };
    params15.convParams.stride = { 1, 1 };

    ConvolutionParams params16;
    params16.inputType = Datatype::F32;
    params16.inputLayout = DataLayout::bfyx;
    params16.outputLayout = DataLayout::bfyx;
    params16.activationFunc = ActivationFunction::RELU;
    params16.nlParams = { 0, 0 };
    params16.inDims = { 13, 13, 192, 1 };
    params16.inDesc = { 0,{ 13, 169, 32448, 32448 }, false };
    params16.outDims = { 13, 13, 128, 1 };
    params16.outDesc = { 0,{ 13, 169, 21632, 21632 }, false };
    params16.convParams.filterSize = { 3, 3 };
    params16.convParams.padding = { 1, 1 };
    params16.convParams.stride = { 1, 1 };

    ConvolutionParams params17;
    params17.inputType = Datatype::F32;
    params17.inputLayout = DataLayout::bfyx;
    params17.outputLayout = DataLayout::bfyx;
    params17.activationFunc = ActivationFunction::RELU;
    params17.nlParams = { 0, 0 };
    params17.inDims = { 13, 13, 192, 1 };
    params17.inDesc = { 16,{ 15, 225, 43200, 43200 }, false };
    params17.outDims = { 13, 13, 128, 1 };
    params17.outDesc = { 0,{ 13, 169, 21632, 21632 }, false };
    params17.convParams.filterSize = { 3, 3 };
    params17.convParams.padding = { 1, 1 };
    params17.convParams.stride = { 1, 1 };

    ConvolutionParams params18;
    params18.inputType = Datatype::F16;
    params18.inputLayout = DataLayout::bfyx;
    params18.outputLayout = DataLayout::bfyx;
    params18.activationFunc = ActivationFunction::RELU;
    params18.nlParams = { 0, 0 };
    params18.inDims = { 224, 224, 3, 1 };
    params18.inDesc = { 0,{ 224, 50176, 150528, 150528 }, false };
    params18.outDims = { 224, 224, 64, 1 };
    params18.outDesc = { 0,{ 224, 50176, 3211264, 3211264 }, false };
    params18.convParams.filterSize = { 3, 3 };
    params18.convParams.padding = { 1, 1 };
    params18.convParams.stride = { 1, 1 };

    ConvolutionParams params19;
    params19.inputType = Datatype::F16;
    params19.inputLayout = DataLayout::bfyx;
    params19.outputLayout = DataLayout::bfyx;
    params19.activationFunc = ActivationFunction::RELU;
    params19.nlParams = { 0, 0 };
    params19.inDims = { 224, 224, 3, 1 };
    params19.inDesc = { 227,{ 226, 51076, 153228, 153228 }, false };
    params19.outDims = { 224, 224, 64, 1 };
    params19.outDesc = { 0,{ 224, 50176, 3211264, 3211264 }, false };
    params19.convParams.filterSize = { 3, 3 };
    params19.convParams.padding = { 1, 1 };
    params19.convParams.stride = { 1, 1 };

    ConvolutionParams params20;
    params20.inputType = Datatype::F16;
    params20.inputLayout = DataLayout::bfyx;
    params20.outputLayout = DataLayout::bfyx;
    params20.activationFunc = ActivationFunction::RELU;
    params20.nlParams = { 0, 0 };
    params20.inDims = { 224, 224, 64, 1 };
    params20.inDesc = { 0,{ 224, 50176, 3211264, 3211264 }, false };
    params20.outDims = { 224, 224, 64, 1 };
    params20.outDesc = { 0,{ 224, 50176, 3211264, 3211264 }, false };
    params20.convParams.filterSize = { 3, 3 };
    params20.convParams.padding = { 1, 1 };
    params20.convParams.stride = { 1, 1 };

    ConvolutionParams params21;
    params21.inputType = Datatype::F16;
    params21.inputLayout = DataLayout::bfyx;
    params21.outputLayout = DataLayout::bfyx;
    params21.activationFunc = ActivationFunction::RELU;
    params21.nlParams = { 0, 0 };
    params21.inDims = { 224, 224, 64, 1 };
    params21.inDesc = { 227,{ 226, 51076, 3268864, 3268864 }, false };
    params21.outDims = { 224, 224, 64, 1 };
    params21.outDesc = { 0,{ 224, 50176, 3211264, 3211264 }, false };
    params21.convParams.filterSize = { 3, 3 };
    params21.convParams.padding = { 1, 1 };
    params21.convParams.stride = { 1, 1 };

    ConvolutionParams params22;
    params22.inputType = Datatype::F16;
    params22.inputLayout = DataLayout::bfyx;
    params22.outputLayout = DataLayout::bfyx;
    params22.activationFunc = ActivationFunction::RELU;
    params22.nlParams = { 0, 0 };
    params22.inDims = { 112, 112, 64, 1 };
    params22.inDesc = { 0,{ 112, 12544, 802816, 802816 }, false };
    params22.outDims = { 112, 112, 128, 1 };
    params22.outDesc = { 0,{ 112, 12544, 1605632, 1605632 }, false };
    params22.convParams.filterSize = { 3, 3 };
    params22.convParams.padding = { 1, 1 };
    params22.convParams.stride = { 1, 1 };

    ConvolutionParams params23;
    params23.inputType = Datatype::F16;
    params23.inputLayout = DataLayout::bfyx;
    params23.outputLayout = DataLayout::bfyx;
    params23.activationFunc = ActivationFunction::RELU;
    params23.nlParams = { 0, 0 };
    params23.inDims = { 112, 112, 64, 1 };
    params23.inDesc = { 115,{ 114, 12996, 831744, 831744 }, false };
    params23.outDims = { 112, 112, 128, 1 };
    params23.outDesc = { 0,{ 112, 12544, 1605632, 1605632 }, false };
    params23.convParams.filterSize = { 3, 3 };
    params23.convParams.padding = { 1, 1 };
    params23.convParams.stride = { 1, 1 };

    ConvolutionParams params24;
    params24.inputType = Datatype::F16;
    params24.inputLayout = DataLayout::bfyx;
    params24.outputLayout = DataLayout::bfyx;
    params24.activationFunc = ActivationFunction::RELU;
    params24.nlParams = { 0, 0 };
    params24.inDims = { 112, 112, 128, 1 };
    params24.inDesc = { 0,{ 112, 12544, 1605632, 1605632 }, false };
    params24.outDims = { 112, 112, 128, 1 };
    params24.outDesc = { 0,{ 112, 12544, 1605632, 1605632 }, false };
    params24.convParams.filterSize = { 3, 3 };
    params24.convParams.padding = { 1, 1 };
    params24.convParams.stride = { 1, 1 };

    ConvolutionParams params25;
    params25.inputType = Datatype::F16;
    params25.inputLayout = DataLayout::bfyx;
    params25.outputLayout = DataLayout::bfyx;
    params25.activationFunc = ActivationFunction::RELU;
    params25.nlParams = { 0, 0 };
    params25.inDims = { 112, 112, 128, 1 };
    params25.inDesc = { 115,{ 114, 12996, 1663488, 1663488 }, false };
    params25.outDims = { 112, 112, 128, 1 };
    params25.outDesc = { 0,{ 112, 12544, 1605632, 1605632 }, false };
    params25.convParams.filterSize = { 3, 3 };
    params25.convParams.padding = { 1, 1 };
    params25.convParams.stride = { 1, 1 };

    ConvolutionParams params26;
    params26.inputType = Datatype::F16;
    params26.inputLayout = DataLayout::bfyx;
    params26.outputLayout = DataLayout::bfyx;
    params26.activationFunc = ActivationFunction::RELU;
    params26.nlParams = { 0, 0 };
    params26.inDims = { 56, 56, 128, 1 };
    params26.inDesc = { 0,{ 56, 3136, 401408, 401408 }, false };
    params26.outDims = { 56, 56, 256, 1 };
    params26.outDesc = { 0,{ 56, 3136, 802816, 802816 }, false };
    params26.convParams.filterSize = { 3, 3 };
    params26.convParams.padding = { 1, 1 };
    params26.convParams.stride = { 1, 1 };

    ConvolutionParams params27;
    params27.inputType = Datatype::F16;
    params27.inputLayout = DataLayout::bfyx;
    params27.outputLayout = DataLayout::bfyx;
    params27.activationFunc = ActivationFunction::RELU;
    params27.nlParams = { 0, 0 };
    params27.inDims = { 56, 56, 128, 1 };
    params27.inDesc = { 59,{ 58, 3364, 430592, 430592 }, false };
    params27.outDims = { 56, 56, 256, 1 };
    params27.outDesc = { 0,{ 56, 3136, 802816, 802816 }, false };
    params27.convParams.filterSize = { 3, 3 };
    params27.convParams.padding = { 1, 1 };
    params27.convParams.stride = { 1, 1 };

    ConvolutionParams params28;
    params28.inputType = Datatype::F16;
    params28.inputLayout = DataLayout::bfyx;
    params28.outputLayout = DataLayout::bfyx;
    params28.activationFunc = ActivationFunction::RELU;
    params28.nlParams = { 0, 0 };
    params28.inDims = { 56, 56, 256, 1 };
    params28.inDesc = { 0,{ 56, 3136, 802816, 802816 }, false };
    params28.outDims = { 56, 56, 256, 1 };
    params28.outDesc = { 0,{ 56, 3136, 802816, 802816 }, false };
    params28.convParams.filterSize = { 3, 3 };
    params28.convParams.padding = { 1, 1 };
    params28.convParams.stride = { 1, 1 };

    ConvolutionParams params29;
    params29.inputType = Datatype::F16;
    params29.inputLayout = DataLayout::bfyx;
    params29.outputLayout = DataLayout::bfyx;
    params29.activationFunc = ActivationFunction::RELU;
    params29.nlParams = { 0, 0 };
    params29.inDims = { 56, 56, 256, 1 };
    params29.inDesc = { 59,{ 58, 3364, 861184, 861184 }, false };
    params29.outDims = { 56, 56, 256, 1 };
    params29.outDesc = { 0,{ 56, 3136, 802816, 802816 }, false };
    params29.convParams.filterSize = { 3, 3 };
    params29.convParams.padding = { 1, 1 };
    params29.convParams.stride = { 1, 1 };

    ConvolutionParams params30;
    params30.inputType = Datatype::F16;
    params30.inputLayout = DataLayout::bfyx;
    params30.outputLayout = DataLayout::bfyx;
    params30.activationFunc = ActivationFunction::RELU;
    params30.nlParams = { 0, 0 };
    params30.inDims = { 28, 28, 256, 1 };
    params30.inDesc = { 0,{ 28, 784, 200704, 200704 }, false };
    params30.outDims = { 28, 28, 512, 1 };
    params30.outDesc = { 0,{ 28, 784, 401408, 401408 }, false };
    params30.convParams.filterSize = { 3, 3 };
    params30.convParams.padding = { 1, 1 };
    params30.convParams.stride = { 1, 1 };

    ConvolutionParams params31;
    params31.inputType = Datatype::F16;
    params31.inputLayout = DataLayout::bfyx;
    params31.outputLayout = DataLayout::bfyx;
    params31.activationFunc = ActivationFunction::RELU;
    params31.nlParams = { 0, 0 };
    params31.inDims = { 28, 28, 256, 1 };
    params31.inDesc = { 31,{ 30, 900, 230400, 230400 }, false };
    params31.outDims = { 28, 28, 512, 1 };
    params31.outDesc = { 0,{ 28, 784, 401408, 401408 }, false };
    params31.convParams.filterSize = { 3, 3 };
    params31.convParams.padding = { 1, 1 };
    params31.convParams.stride = { 1, 1 };

    ConvolutionParams params32;
    params32.inputType = Datatype::F16;
    params32.inputLayout = DataLayout::bfyx;
    params32.outputLayout = DataLayout::bfyx;
    params32.activationFunc = ActivationFunction::RELU;
    params32.nlParams = { 0, 0 };
    params32.inDims = { 28, 28, 512, 1 };
    params32.inDesc = { 0,{ 28, 784, 401408, 401408 }, false };
    params32.outDims = { 28, 28, 512, 1 };
    params32.outDesc = { 0,{ 28, 784, 401408, 401408 }, false };
    params32.convParams.filterSize = { 3, 3 };
    params32.convParams.padding = { 1, 1 };
    params32.convParams.stride = { 1, 1 };

    ConvolutionParams params33;
    params33.inputType = Datatype::F16;
    params33.inputLayout = DataLayout::bfyx;
    params33.outputLayout = DataLayout::bfyx;
    params33.activationFunc = ActivationFunction::RELU;
    params33.nlParams = { 0, 0 };
    params33.inDims = { 28, 28, 512, 1 };
    params33.inDesc = { 31,{ 30, 900, 460800, 460800 }, false };
    params33.outDims = { 28, 28, 512, 1 };
    params33.outDesc = { 0,{ 28, 784, 401408, 401408 }, false };
    params33.convParams.filterSize = { 3, 3 };
    params33.convParams.padding = { 1, 1 };
    params33.convParams.stride = { 1, 1 };

    ConvolutionParams params34;
    params34.inputType = Datatype::F16;
    params34.inputLayout = DataLayout::bfyx;
    params34.outputLayout = DataLayout::bfyx;
    params34.activationFunc = ActivationFunction::RELU;
    params34.nlParams = { 0, 0 };
    params34.inDims = { 14, 14, 512, 1 };
    params34.inDesc = { 0,{ 14, 196, 100352, 100352 }, false };
    params34.outDims = { 14, 14, 512, 1 };
    params34.outDesc = { 0,{ 14, 196, 100352, 100352 }, false };
    params34.convParams.filterSize = { 3, 3 };
    params34.convParams.padding = { 1, 1 };
    params34.convParams.stride = { 1, 1 };

    ConvolutionParams params35;
    params35.inputType = Datatype::F16;
    params35.inputLayout = DataLayout::bfyx;
    params35.outputLayout = DataLayout::bfyx;
    params35.activationFunc = ActivationFunction::RELU;
    params35.nlParams = { 0, 0 };
    params35.inDims = { 14, 14, 512, 1 };
    params35.inDesc = { 17,{ 16, 256, 131072, 131072 }, false };
    params35.outDims = { 14, 14, 512, 1 };
    params35.outDesc = { 0,{ 14, 196, 100352, 100352 }, false };
    params35.convParams.filterSize = { 3, 3 };
    params35.convParams.padding = { 1, 1 };
    params35.convParams.stride = { 1, 1 };

    ConvolutionParams params36;
    params36.inputType = Datatype::F16;
    params36.inputLayout = DataLayout::bfyx;
    params36.outputLayout = DataLayout::bfyx;
    params36.activationFunc = ActivationFunction::RELU;
    params36.nlParams = { 0, 0 };
    params36.inDims = { 224, 224, 3, 1 };
    params36.inDesc = { 0,{ 224, 50176, 150528, 150528 }, false };
    params36.outDims = { 112, 112, 64, 1 };
    params36.outDesc = { 0,{ 112, 12544, 802816, 802816 }, false };
    params36.convParams.filterSize = { 7, 7 };
    params36.convParams.padding = { 3, 3 };
    params36.convParams.stride = { 2, 2 };

    ConvolutionParams params37;
    params37.inputType = Datatype::F16;
    params37.inputLayout = DataLayout::bfyx;
    params37.outputLayout = DataLayout::bfyx;
    params37.activationFunc = ActivationFunction::RELU;
    params37.nlParams = { 0, 0 };
    params37.inDims = { 224, 224, 3, 1 };
    params37.inDesc = { 693,{ 230, 52900, 158700, 158700 }, false };
    params37.outDims = { 112, 112, 64, 1 };
    params37.outDesc = { 0,{ 112, 12544, 802816, 802816 }, false };
    params37.convParams.filterSize = { 7, 7 };
    params37.convParams.padding = { 3, 3 };
    params37.convParams.stride = { 2, 2 };

    ConvolutionParams params38;
    params38.inputType = Datatype::F16;
    params38.inputLayout = DataLayout::bfyx;
    params38.outputLayout = DataLayout::bfyx;
    params38.activationFunc = ActivationFunction::RELU;
    params38.nlParams = { 0, 0 };
    params38.inDims = { 56, 56, 64, 1 };
    params38.inDesc = { 0,{ 56, 3136, 200704, 200704 }, false };
    params38.outDims = { 56, 56, 64, 1 };
    params38.outDesc = { 0,{ 56, 3136, 200704, 200704 }, false };
    params38.convParams.filterSize = { 1, 1 };
    params38.convParams.padding = { 0, 0 };
    params38.convParams.stride = { 1, 1 };

    ConvolutionParams params39;
    params39.inputType = Datatype::F16;
    params39.inputLayout = DataLayout::bfyx;
    params39.outputLayout = DataLayout::bfyx;
    params39.activationFunc = ActivationFunction::RELU;
    params39.nlParams = { 0, 0 };
    params39.inDims = { 56, 56, 64, 1 };
    params39.inDesc = { 0,{ 56, 3136, 200704, 200704 }, false };
    params39.outDims = { 56, 56, 192, 1 };
    params39.outDesc = { 0,{ 56, 3136, 602112, 602112 }, false };
    params39.convParams.filterSize = { 3, 3 };
    params39.convParams.padding = { 1, 1 };
    params39.convParams.stride = { 1, 1 };

    ConvolutionParams params40;
    params40.inputType = Datatype::F16;
    params40.inputLayout = DataLayout::bfyx;
    params40.outputLayout = DataLayout::bfyx;
    params40.activationFunc = ActivationFunction::RELU;
    params40.nlParams = { 0, 0 };
    params40.inDims = { 56, 56, 64, 1 };
    params40.inDesc = { 59,{ 58, 3364, 215296, 215296 }, false };
    params40.outDims = { 56, 56, 192, 1 };
    params40.outDesc = { 0,{ 56, 3136, 602112, 602112 }, false };
    params40.convParams.filterSize = { 3, 3 };
    params40.convParams.padding = { 1, 1 };
    params40.convParams.stride = { 1, 1 };

    ConvolutionParams params41;
    params41.inputType = Datatype::F16;
    params41.inputLayout = DataLayout::bfyx;
    params41.outputLayout = DataLayout::bfyx;
    params41.activationFunc = ActivationFunction::RELU;
    params41.nlParams = { 0, 0 };
    params41.inDims = { 28, 28, 192, 1 };
    params41.inDesc = { 0,{ 28, 784, 150528, 150528 }, false };
    params41.outDims = { 28, 28, 64, 1 };
    params41.outDesc = { 0,{ 28, 784, 50176, 50176 }, false };
    params41.convParams.filterSize = { 1, 1 };
    params41.convParams.padding = { 0, 0 };
    params41.convParams.stride = { 1, 1 };

    ConvolutionParams params42;
    params42.inputType = Datatype::F16;
    params42.inputLayout = DataLayout::bfyx;
    params42.outputLayout = DataLayout::bfyx;
    params42.activationFunc = ActivationFunction::RELU;
    params42.nlParams = { 0, 0 };
    params42.inDims = { 28, 28, 192, 1 };
    params42.inDesc = { 0,{ 28, 784, 150528, 150528 }, false };
    params42.outDims = { 28, 28, 96, 1 };
    params42.outDesc = { 0,{ 28, 784, 75264, 75264 }, false };
    params42.convParams.filterSize = { 1, 1 };
    params42.convParams.padding = { 0, 0 };
    params42.convParams.stride = { 1, 1 };

    ConvolutionParams params43;
    params43.inputType = Datatype::F16;
    params43.inputLayout = DataLayout::bfyx;
    params43.outputLayout = DataLayout::bfyx;
    params43.activationFunc = ActivationFunction::RELU;
    params43.nlParams = { 0, 0 };
    params43.inDims = { 28, 28, 192, 1 };
    params43.inDesc = { 0,{ 28, 784, 150528, 150528 }, false };
    params43.outDims = { 28, 28, 16, 1 };
    params43.outDesc = { 0,{ 28, 784, 12544, 12544 }, false };
    params43.convParams.filterSize = { 1, 1 };
    params43.convParams.padding = { 0, 0 };
    params43.convParams.stride = { 1, 1 };

    ConvolutionParams params44;
    params44.inputType = Datatype::F16;
    params44.inputLayout = DataLayout::bfyx;
    params44.outputLayout = DataLayout::bfyx;
    params44.activationFunc = ActivationFunction::RELU;
    params44.nlParams = { 0, 0 };
    params44.inDims = { 28, 28, 96, 1 };
    params44.inDesc = { 0,{ 28, 784, 75264, 75264 }, false };
    params44.outDims = { 28, 28, 128, 1 };
    params44.outDesc = { 0,{ 28, 784, 100352, 100352 }, false };
    params44.convParams.filterSize = { 3, 3 };
    params44.convParams.padding = { 1, 1 };
    params44.convParams.stride = { 1, 1 };

    ConvolutionParams params45;
    params45.inputType = Datatype::F16;
    params45.inputLayout = DataLayout::bfyx;
    params45.outputLayout = DataLayout::bfyx;
    params45.activationFunc = ActivationFunction::RELU;
    params45.nlParams = { 0, 0 };
    params45.inDims = { 28, 28, 96, 1 };
    params45.inDesc = { 31,{ 30, 900, 86400, 86400 }, false };
    params45.outDims = { 28, 28, 128, 1 };
    params45.outDesc = { 0,{ 28, 784, 100352, 100352 }, false };
    params45.convParams.filterSize = { 3, 3 };
    params45.convParams.padding = { 1, 1 };
    params45.convParams.stride = { 1, 1 };

    ConvolutionParams params46;
    params46.inputType = Datatype::F16;
    params46.inputLayout = DataLayout::bfyx;
    params46.outputLayout = DataLayout::bfyx;
    params46.activationFunc = ActivationFunction::RELU;
    params46.nlParams = { 0, 0 };
    params46.inDims = { 28, 28, 16, 1 };
    params46.inDesc = { 0,{ 28, 784, 12544, 12544 }, false };
    params46.outDims = { 28, 28, 32, 1 };
    params46.outDesc = { 0,{ 28, 784, 25088, 25088 }, false };
    params46.convParams.filterSize = { 5, 5 };
    params46.convParams.padding = { 2, 2 };
    params46.convParams.stride = { 1, 1 };

    ConvolutionParams params47;
    params47.inputType = Datatype::F16;
    params47.inputLayout = DataLayout::bfyx;
    params47.outputLayout = DataLayout::bfyx;
    params47.activationFunc = ActivationFunction::RELU;
    params47.nlParams = { 0, 0 };
    params47.inDims = { 28, 28, 16, 1 };
    params47.inDesc = { 66,{ 32, 1024, 16384, 16384 }, false };
    params47.outDims = { 28, 28, 32, 1 };
    params47.outDesc = { 0,{ 28, 784, 25088, 25088 }, false };
    params47.convParams.filterSize = { 5, 5 };
    params47.convParams.padding = { 2, 2 };
    params47.convParams.stride = { 1, 1 };

    ConvolutionParams params48;
    params48.inputType = Datatype::F16;
    params48.inputLayout = DataLayout::bfyx;
    params48.outputLayout = DataLayout::bfyx;
    params48.activationFunc = ActivationFunction::RELU;
    params48.nlParams = { 0, 0 };
    params48.inDims = { 28, 28, 192, 1 };
    params48.inDesc = { 0,{ 28, 784, 150528, 150528 }, false };
    params48.outDims = { 28, 28, 32, 1 };
    params48.outDesc = { 0,{ 28, 784, 25088, 25088 }, false };
    params48.convParams.filterSize = { 1, 1 };
    params48.convParams.padding = { 0, 0 };
    params48.convParams.stride = { 1, 1 };

    ConvolutionParams params49;
    params49.inputType = Datatype::F16;
    params49.inputLayout = DataLayout::bfyx;
    params49.outputLayout = DataLayout::bfyx;
    params49.activationFunc = ActivationFunction::RELU;
    params49.nlParams = { 0, 0 };
    params49.inDims = { 28, 28, 256, 1 };
    params49.inDesc = { 0,{ 28, 784, 200704, 200704 }, false };
    params49.outDims = { 28, 28, 64, 1 };
    params49.outDesc = { 0,{ 28, 784, 50176, 50176 }, false };
    params49.convParams.filterSize = { 1, 1 };
    params49.convParams.padding = { 0, 0 };
    params49.convParams.stride = { 1, 1 };

    ConvolutionParams params50;
    params50.inputType = Datatype::F16;
    params50.inputLayout = DataLayout::bfyx;
    params50.outputLayout = DataLayout::bfyx;
    params50.activationFunc = ActivationFunction::RELU;
    params50.nlParams = { 0, 0 };
    params50.inDims = { 28, 28, 256, 1 };
    params50.inDesc = { 0,{ 28, 784, 200704, 200704 }, false };
    params50.outDims = { 28, 28, 32, 1 };
    params50.outDesc = { 0,{ 28, 784, 25088, 25088 }, false };
    params50.convParams.filterSize = { 1, 1 };
    params50.convParams.padding = { 0, 0 };
    params50.convParams.stride = { 1, 1 };

    ConvolutionParams params51;
    params51.inputType = Datatype::F16;
    params51.inputLayout = DataLayout::bfyx;
    params51.outputLayout = DataLayout::bfyx;
    params51.activationFunc = ActivationFunction::RELU;
    params51.nlParams = { 0, 0 };
    params51.inDims = { 28, 28, 32, 1 };
    params51.inDesc = { 0,{ 28, 784, 25088, 25088 }, false };
    params51.outDims = { 28, 28, 96, 1 };
    params51.outDesc = { 0,{ 28, 784, 75264, 75264 }, false };
    params51.convParams.filterSize = { 5, 5 };
    params51.convParams.padding = { 2, 2 };
    params51.convParams.stride = { 1, 1 };

    ConvolutionParams params52;
    params52.inputType = Datatype::F16;
    params52.inputLayout = DataLayout::bfyx;
    params52.outputLayout = DataLayout::bfyx;
    params52.activationFunc = ActivationFunction::RELU;
    params52.nlParams = { 0, 0 };
    params52.inDims = { 28, 28, 32, 1 };
    params52.inDesc = { 66,{ 32, 1024, 32768, 32768 }, false };
    params52.outDims = { 28, 28, 96, 1 };
    params52.outDesc = { 0,{ 28, 784, 75264, 75264 }, false };
    params52.convParams.filterSize = { 5, 5 };
    params52.convParams.padding = { 2, 2 };
    params52.convParams.stride = { 1, 1 };

    ConvolutionParams params53;
    params53.inputType = Datatype::F16;
    params53.inputLayout = DataLayout::bfyx;
    params53.outputLayout = DataLayout::bfyx;
    params53.activationFunc = ActivationFunction::RELU;
    params53.nlParams = { 0, 0 };
    params53.inDims = { 28, 28, 256, 1 };
    params53.inDesc = { 0,{ 28, 784, 200704, 200704 }, false };
    params53.outDims = { 28, 28, 128, 1 };
    params53.outDesc = { 0,{ 28, 784, 100352, 100352 }, false };
    params53.convParams.filterSize = { 1, 1 };
    params53.convParams.padding = { 0, 0 };
    params53.convParams.stride = { 1, 1 };

    ConvolutionParams params54;
    params54.inputType = Datatype::F16;
    params54.inputLayout = DataLayout::bfyx;
    params54.outputLayout = DataLayout::bfyx;
    params54.activationFunc = ActivationFunction::RELU;
    params54.nlParams = { 0, 0 };
    params54.inDims = { 28, 28, 128, 1 };
    params54.inDesc = { 0,{ 28, 784, 100352, 100352 }, false };
    params54.outDims = { 28, 28, 192, 1 };
    params54.outDesc = { 0,{ 28, 784, 150528, 150528 }, false };
    params54.convParams.filterSize = { 3, 3 };
    params54.convParams.padding = { 1, 1 };
    params54.convParams.stride = { 1, 1 };

    ConvolutionParams params55;
    params55.inputType = Datatype::F16;
    params55.inputLayout = DataLayout::bfyx;
    params55.outputLayout = DataLayout::bfyx;
    params55.activationFunc = ActivationFunction::RELU;
    params55.nlParams = { 0, 0 };
    params55.inDims = { 28, 28, 128, 1 };
    params55.inDesc = { 31,{ 30, 900, 115200, 115200 }, false };
    params55.outDims = { 28, 28, 192, 1 };
    params55.outDesc = { 0,{ 28, 784, 150528, 150528 }, false };
    params55.convParams.filterSize = { 3, 3 };
    params55.convParams.padding = { 1, 1 };
    params55.convParams.stride = { 1, 1 };

    ConvolutionParams params56;
    params56.inputType = Datatype::F16;
    params56.inputLayout = DataLayout::bfyx;
    params56.outputLayout = DataLayout::bfyx;
    params56.activationFunc = ActivationFunction::RELU;
    params56.nlParams = { 0, 0 };
    params56.inDims = { 14, 14, 480, 1 };
    params56.inDesc = { 0,{ 14, 196, 94080, 94080 }, false };
    params56.outDims = { 14, 14, 192, 1 };
    params56.outDesc = { 0,{ 14, 196, 37632, 37632 }, false };
    params56.convParams.filterSize = { 1, 1 };
    params56.convParams.padding = { 0, 0 };
    params56.convParams.stride = { 1, 1 };

    ConvolutionParams params57;
    params57.inputType = Datatype::F16;
    params57.inputLayout = DataLayout::bfyx;
    params57.outputLayout = DataLayout::bfyx;
    params57.activationFunc = ActivationFunction::RELU;
    params57.nlParams = { 0, 0 };
    params57.inDims = { 14, 14, 480, 1 };
    params57.inDesc = { 0,{ 14, 196, 94080, 94080 }, false };
    params57.outDims = { 14, 14, 96, 1 };
    params57.outDesc = { 0,{ 14, 196, 18816, 18816 }, false };
    params57.convParams.filterSize = { 1, 1 };
    params57.convParams.padding = { 0, 0 };
    params57.convParams.stride = { 1, 1 };

    ConvolutionParams params58;
    params58.inputType = Datatype::F16;
    params58.inputLayout = DataLayout::bfyx;
    params58.outputLayout = DataLayout::bfyx;
    params58.activationFunc = ActivationFunction::RELU;
    params58.nlParams = { 0, 0 };
    params58.inDims = { 14, 14, 480, 1 };
    params58.inDesc = { 0,{ 14, 196, 94080, 94080 }, false };
    params58.outDims = { 14, 14, 16, 1 };
    params58.outDesc = { 0,{ 14, 196, 3136, 3136 }, false };
    params58.convParams.filterSize = { 1, 1 };
    params58.convParams.padding = { 0, 0 };
    params58.convParams.stride = { 1, 1 };

    ConvolutionParams params59;
    params59.inputType = Datatype::F16;
    params59.inputLayout = DataLayout::bfyx;
    params59.outputLayout = DataLayout::bfyx;
    params59.activationFunc = ActivationFunction::RELU;
    params59.nlParams = { 0, 0 };
    params59.inDims = { 14, 14, 96, 1 };
    params59.inDesc = { 0,{ 14, 196, 18816, 18816 }, false };
    params59.outDims = { 14, 14, 208, 1 };
    params59.outDesc = { 0,{ 14, 196, 40768, 40768 }, false };
    params59.convParams.filterSize = { 3, 3 };
    params59.convParams.padding = { 1, 1 };
    params59.convParams.stride = { 1, 1 };

    ConvolutionParams params60;
    params60.inputType = Datatype::F16;
    params60.inputLayout = DataLayout::bfyx;
    params60.outputLayout = DataLayout::bfyx;
    params60.activationFunc = ActivationFunction::RELU;
    params60.nlParams = { 0, 0 };
    params60.inDims = { 14, 14, 96, 1 };
    params60.inDesc = { 17,{ 16, 256, 24576, 24576 }, false };
    params60.outDims = { 14, 14, 208, 1 };
    params60.outDesc = { 0,{ 14, 196, 40768, 40768 }, false };
    params60.convParams.filterSize = { 3, 3 };
    params60.convParams.padding = { 1, 1 };
    params60.convParams.stride = { 1, 1 };

    ConvolutionParams params61;
    params61.inputType = Datatype::F16;
    params61.inputLayout = DataLayout::bfyx;
    params61.outputLayout = DataLayout::bfyx;
    params61.activationFunc = ActivationFunction::RELU;
    params61.nlParams = { 0, 0 };
    params61.inDims = { 14, 14, 16, 1 };
    params61.inDesc = { 0,{ 14, 196, 3136, 3136 }, false };
    params61.outDims = { 14, 14, 48, 1 };
    params61.outDesc = { 0,{ 14, 196, 9408, 9408 }, false };
    params61.convParams.filterSize = { 5, 5 };
    params61.convParams.padding = { 2, 2 };
    params61.convParams.stride = { 1, 1 };

    ConvolutionParams params62;
    params62.inputType = Datatype::F16;
    params62.inputLayout = DataLayout::bfyx;
    params62.outputLayout = DataLayout::bfyx;
    params62.activationFunc = ActivationFunction::RELU;
    params62.nlParams = { 0, 0 };
    params62.inDims = { 14, 14, 16, 1 };
    params62.inDesc = { 38,{ 18, 324, 5184, 5184 }, false };
    params62.outDims = { 14, 14, 48, 1 };
    params62.outDesc = { 0,{ 14, 196, 9408, 9408 }, false };
    params62.convParams.filterSize = { 5, 5 };
    params62.convParams.padding = { 2, 2 };
    params62.convParams.stride = { 1, 1 };

    ConvolutionParams params63;
    params63.inputType = Datatype::F16;
    params63.inputLayout = DataLayout::bfyx;
    params63.outputLayout = DataLayout::bfyx;
    params63.activationFunc = ActivationFunction::RELU;
    params63.nlParams = { 0, 0 };
    params63.inDims = { 14, 14, 480, 1 };
    params63.inDesc = { 0,{ 14, 196, 94080, 94080 }, false };
    params63.outDims = { 14, 14, 64, 1 };
    params63.outDesc = { 0,{ 14, 196, 12544, 12544 }, false };
    params63.convParams.filterSize = { 1, 1 };
    params63.convParams.padding = { 0, 0 };
    params63.convParams.stride = { 1, 1 };

    ConvolutionParams params64;
    params64.inputType = Datatype::F16;
    params64.inputLayout = DataLayout::bfyx;
    params64.outputLayout = DataLayout::bfyx;
    params64.activationFunc = ActivationFunction::RELU;
    params64.nlParams = { 0, 0 };
    params64.inDims = { 14, 14, 512, 1 };
    params64.inDesc = { 0,{ 14, 196, 100352, 100352 }, false };
    params64.outDims = { 14, 14, 64, 1 };
    params64.outDesc = { 0,{ 14, 196, 12544, 12544 }, false };
    params64.convParams.filterSize = { 1, 1 };
    params64.convParams.padding = { 0, 0 };
    params64.convParams.stride = { 1, 1 };

    ConvolutionParams params65;
    params65.inputType = Datatype::F16;
    params65.inputLayout = DataLayout::bfyx;
    params65.outputLayout = DataLayout::bfyx;
    params65.activationFunc = ActivationFunction::RELU;
    params65.nlParams = { 0, 0 };
    params65.inDims = { 14, 14, 512, 1 };
    params65.inDesc = { 0,{ 14, 196, 100352, 100352 }, false };
    params65.outDims = { 14, 14, 24, 1 };
    params65.outDesc = { 0,{ 14, 196, 4704, 4704 }, false };
    params65.convParams.filterSize = { 1, 1 };
    params65.convParams.padding = { 0, 0 };
    params65.convParams.stride = { 1, 1 };

    ConvolutionParams params66;
    params66.inputType = Datatype::F16;
    params66.inputLayout = DataLayout::bfyx;
    params66.outputLayout = DataLayout::bfyx;
    params66.activationFunc = ActivationFunction::RELU;
    params66.nlParams = { 0, 0 };
    params66.inDims = { 14, 14, 24, 1 };
    params66.inDesc = { 0,{ 14, 196, 4704, 4704 }, false };
    params66.outDims = { 14, 14, 64, 1 };
    params66.outDesc = { 0,{ 14, 196, 12544, 12544 }, false };
    params66.convParams.filterSize = { 5, 5 };
    params66.convParams.padding = { 2, 2 };
    params66.convParams.stride = { 1, 1 };

    ConvolutionParams params67;
    params67.inputType = Datatype::F16;
    params67.inputLayout = DataLayout::bfyx;
    params67.outputLayout = DataLayout::bfyx;
    params67.activationFunc = ActivationFunction::RELU;
    params67.nlParams = { 0, 0 };
    params67.inDims = { 14, 14, 24, 1 };
    params67.inDesc = { 38,{ 18, 324, 7776, 7776 }, false };
    params67.outDims = { 14, 14, 64, 1 };
    params67.outDesc = { 0,{ 14, 196, 12544, 12544 }, false };
    params67.convParams.filterSize = { 5, 5 };
    params67.convParams.padding = { 2, 2 };
    params67.convParams.stride = { 1, 1 };

    ConvolutionParams params68;
    params68.inputType = Datatype::F16;
    params68.inputLayout = DataLayout::bfyx;
    params68.outputLayout = DataLayout::bfyx;
    params68.activationFunc = ActivationFunction::RELU;
    params68.nlParams = { 0, 0 };
    params68.inDims = { 14, 14, 512, 1 };
    params68.inDesc = { 0,{ 14, 196, 100352, 100352 }, false };
    params68.outDims = { 14, 14, 112, 1 };
    params68.outDesc = { 0,{ 14, 196, 21952, 21952 }, false };
    params68.convParams.filterSize = { 1, 1 };
    params68.convParams.padding = { 0, 0 };
    params68.convParams.stride = { 1, 1 };

    ConvolutionParams params69;
    params69.inputType = Datatype::F16;
    params69.inputLayout = DataLayout::bfyx;
    params69.outputLayout = DataLayout::bfyx;
    params69.activationFunc = ActivationFunction::RELU;
    params69.nlParams = { 0, 0 };
    params69.inDims = { 14, 14, 112, 1 };
    params69.inDesc = { 0,{ 14, 196, 21952, 21952 }, false };
    params69.outDims = { 14, 14, 224, 1 };
    params69.outDesc = { 0,{ 14, 196, 43904, 43904 }, false };
    params69.convParams.filterSize = { 3, 3 };
    params69.convParams.padding = { 1, 1 };
    params69.convParams.stride = { 1, 1 };

    ConvolutionParams params70;
    params70.inputType = Datatype::F16;
    params70.inputLayout = DataLayout::bfyx;
    params70.outputLayout = DataLayout::bfyx;
    params70.activationFunc = ActivationFunction::RELU;
    params70.nlParams = { 0, 0 };
    params70.inDims = { 14, 14, 112, 1 };
    params70.inDesc = { 17,{ 16, 256, 28672, 28672 }, false };
    params70.outDims = { 14, 14, 224, 1 };
    params70.outDesc = { 0,{ 14, 196, 43904, 43904 }, false };
    params70.convParams.filterSize = { 3, 3 };
    params70.convParams.padding = { 1, 1 };
    params70.convParams.stride = { 1, 1 };

    ConvolutionParams params71;
    params71.inputType = Datatype::F16;
    params71.inputLayout = DataLayout::bfyx;
    params71.outputLayout = DataLayout::bfyx;
    params71.activationFunc = ActivationFunction::RELU;
    params71.nlParams = { 0, 0 };
    params71.inDims = { 14, 14, 512, 1 };
    params71.inDesc = { 0,{ 14, 196, 100352, 100352 }, false };
    params71.outDims = { 14, 14, 160, 1 };
    params71.outDesc = { 0,{ 14, 196, 31360, 31360 }, false };
    params71.convParams.filterSize = { 1, 1 };
    params71.convParams.padding = { 0, 0 };
    params71.convParams.stride = { 1, 1 };

    ConvolutionParams params72;
    params72.inputType = Datatype::F16;
    params72.inputLayout = DataLayout::bfyx;
    params72.outputLayout = DataLayout::bfyx;
    params72.activationFunc = ActivationFunction::RELU;
    params72.nlParams = { 0, 0 };
    params72.inDims = { 14, 14, 512, 1 };
    params72.inDesc = { 0,{ 14, 196, 100352, 100352 }, false };
    params72.outDims = { 14, 14, 128, 1 };
    params72.outDesc = { 0,{ 14, 196, 25088, 25088 }, false };
    params72.convParams.filterSize = { 1, 1 };
    params72.convParams.padding = { 0, 0 };
    params72.convParams.stride = { 1, 1 };

    ConvolutionParams params73;
    params73.inputType = Datatype::F16;
    params73.inputLayout = DataLayout::bfyx;
    params73.outputLayout = DataLayout::bfyx;
    params73.activationFunc = ActivationFunction::RELU;
    params73.nlParams = { 0, 0 };
    params73.inDims = { 14, 14, 128, 1 };
    params73.inDesc = { 0,{ 14, 196, 25088, 25088 }, false };
    params73.outDims = { 14, 14, 256, 1 };
    params73.outDesc = { 0,{ 14, 196, 50176, 50176 }, false };
    params73.convParams.filterSize = { 3, 3 };
    params73.convParams.padding = { 1, 1 };
    params73.convParams.stride = { 1, 1 };

    ConvolutionParams params74;
    params74.inputType = Datatype::F16;
    params74.inputLayout = DataLayout::bfyx;
    params74.outputLayout = DataLayout::bfyx;
    params74.activationFunc = ActivationFunction::RELU;
    params74.nlParams = { 0, 0 };
    params74.inDims = { 14, 14, 128, 1 };
    params74.inDesc = { 17,{ 16, 256, 32768, 32768 }, false };
    params74.outDims = { 14, 14, 256, 1 };
    params74.outDesc = { 0,{ 14, 196, 50176, 50176 }, false };
    params74.convParams.filterSize = { 3, 3 };
    params74.convParams.padding = { 1, 1 };
    params74.convParams.stride = { 1, 1 };

    ConvolutionParams params75;
    params75.inputType = Datatype::F16;
    params75.inputLayout = DataLayout::bfyx;
    params75.outputLayout = DataLayout::bfyx;
    params75.activationFunc = ActivationFunction::RELU;
    params75.nlParams = { 0, 0 };
    params75.inDims = { 14, 14, 512, 1 };
    params75.inDesc = { 0,{ 14, 196, 100352, 100352 }, false };
    params75.outDims = { 14, 14, 32, 1 };
    params75.outDesc = { 0,{ 14, 196, 6272, 6272 }, false };
    params75.convParams.filterSize = { 1, 1 };
    params75.convParams.padding = { 0, 0 };
    params75.convParams.stride = { 1, 1 };

    ConvolutionParams params76;
    params76.inputType = Datatype::F16;
    params76.inputLayout = DataLayout::bfyx;
    params76.outputLayout = DataLayout::bfyx;
    params76.activationFunc = ActivationFunction::RELU;
    params76.nlParams = { 0, 0 };
    params76.inDims = { 14, 14, 32, 1 };
    params76.inDesc = { 0,{ 14, 196, 6272, 6272 }, false };
    params76.outDims = { 14, 14, 64, 1 };
    params76.outDesc = { 0,{ 14, 196, 12544, 12544 }, false };
    params76.convParams.filterSize = { 5, 5 };
    params76.convParams.padding = { 2, 2 };
    params76.convParams.stride = { 1, 1 };

    ConvolutionParams params77;
    params77.inputType = Datatype::F16;
    params77.inputLayout = DataLayout::bfyx;
    params77.outputLayout = DataLayout::bfyx;
    params77.activationFunc = ActivationFunction::RELU;
    params77.nlParams = { 0, 0 };
    params77.inDims = { 14, 14, 32, 1 };
    params77.inDesc = { 38,{ 18, 324, 10368, 10368 }, false };
    params77.outDims = { 14, 14, 64, 1 };
    params77.outDesc = { 0,{ 14, 196, 12544, 12544 }, false };
    params77.convParams.filterSize = { 5, 5 };
    params77.convParams.padding = { 2, 2 };
    params77.convParams.stride = { 1, 1 };

    ConvolutionParams params78;
    params78.inputType = Datatype::F16;
    params78.inputLayout = DataLayout::bfyx;
    params78.outputLayout = DataLayout::bfyx;
    params78.activationFunc = ActivationFunction::RELU;
    params78.nlParams = { 0, 0 };
    params78.inDims = { 14, 14, 512, 1 };
    params78.inDesc = { 0,{ 14, 196, 100352, 100352 }, false };
    params78.outDims = { 14, 14, 144, 1 };
    params78.outDesc = { 0,{ 14, 196, 28224, 28224 }, false };
    params78.convParams.filterSize = { 1, 1 };
    params78.convParams.padding = { 0, 0 };
    params78.convParams.stride = { 1, 1 };

    ConvolutionParams params79;
    params79.inputType = Datatype::F16;
    params79.inputLayout = DataLayout::bfyx;
    params79.outputLayout = DataLayout::bfyx;
    params79.activationFunc = ActivationFunction::RELU;
    params79.nlParams = { 0, 0 };
    params79.inDims = { 14, 14, 144, 1 };
    params79.inDesc = { 0,{ 14, 196, 28224, 28224 }, false };
    params79.outDims = { 14, 14, 288, 1 };
    params79.outDesc = { 0,{ 14, 196, 56448, 56448 }, false };
    params79.convParams.filterSize = { 3, 3 };
    params79.convParams.padding = { 1, 1 };
    params79.convParams.stride = { 1, 1 };

    ConvolutionParams params80;
    params80.inputType = Datatype::F16;
    params80.inputLayout = DataLayout::bfyx;
    params80.outputLayout = DataLayout::bfyx;
    params80.activationFunc = ActivationFunction::RELU;
    params80.nlParams = { 0, 0 };
    params80.inDims = { 14, 14, 144, 1 };
    params80.inDesc = { 17,{ 16, 256, 36864, 36864 }, false };
    params80.outDims = { 14, 14, 288, 1 };
    params80.outDesc = { 0,{ 14, 196, 56448, 56448 }, false };
    params80.convParams.filterSize = { 3, 3 };
    params80.convParams.padding = { 1, 1 };
    params80.convParams.stride = { 1, 1 };

    ConvolutionParams params81;
    params81.inputType = Datatype::F16;
    params81.inputLayout = DataLayout::bfyx;
    params81.outputLayout = DataLayout::bfyx;
    params81.activationFunc = ActivationFunction::RELU;
    params81.nlParams = { 0, 0 };
    params81.inDims = { 14, 14, 528, 1 };
    params81.inDesc = { 0,{ 14, 196, 103488, 103488 }, false };
    params81.outDims = { 14, 14, 128, 1 };
    params81.outDesc = { 0,{ 14, 196, 25088, 25088 }, false };
    params81.convParams.filterSize = { 1, 1 };
    params81.convParams.padding = { 0, 0 };
    params81.convParams.stride = { 1, 1 };

    ConvolutionParams params82;
    params82.inputType = Datatype::F16;
    params82.inputLayout = DataLayout::bfyx;
    params82.outputLayout = DataLayout::bfyx;
    params82.activationFunc = ActivationFunction::RELU;
    params82.nlParams = { 0, 0 };
    params82.inDims = { 14, 14, 528, 1 };
    params82.inDesc = { 0,{ 14, 196, 103488, 103488 }, false };
    params82.outDims = { 14, 14, 32, 1 };
    params82.outDesc = { 0,{ 14, 196, 6272, 6272 }, false };
    params82.convParams.filterSize = { 1, 1 };
    params82.convParams.padding = { 0, 0 };
    params82.convParams.stride = { 1, 1 };

    ConvolutionParams params83;
    params83.inputType = Datatype::F16;
    params83.inputLayout = DataLayout::bfyx;
    params83.outputLayout = DataLayout::bfyx;
    params83.activationFunc = ActivationFunction::RELU;
    params83.nlParams = { 0, 0 };
    params83.inDims = { 14, 14, 32, 1 };
    params83.inDesc = { 0,{ 14, 196, 6272, 6272 }, false };
    params83.outDims = { 14, 14, 128, 1 };
    params83.outDesc = { 0,{ 14, 196, 25088, 25088 }, false };
    params83.convParams.filterSize = { 5, 5 };
    params83.convParams.padding = { 2, 2 };
    params83.convParams.stride = { 1, 1 };

    ConvolutionParams params84;
    params84.inputType = Datatype::F16;
    params84.inputLayout = DataLayout::bfyx;
    params84.outputLayout = DataLayout::bfyx;
    params84.activationFunc = ActivationFunction::RELU;
    params84.nlParams = { 0, 0 };
    params84.inDims = { 14, 14, 32, 1 };
    params84.inDesc = { 38,{ 18, 324, 10368, 10368 }, false };
    params84.outDims = { 14, 14, 128, 1 };
    params84.outDesc = { 0,{ 14, 196, 25088, 25088 }, false };
    params84.convParams.filterSize = { 5, 5 };
    params84.convParams.padding = { 2, 2 };
    params84.convParams.stride = { 1, 1 };

    ConvolutionParams params85;
    params85.inputType = Datatype::F16;
    params85.inputLayout = DataLayout::bfyx;
    params85.outputLayout = DataLayout::bfyx;
    params85.activationFunc = ActivationFunction::RELU;
    params85.nlParams = { 0, 0 };
    params85.inDims = { 14, 14, 528, 1 };
    params85.inDesc = { 0,{ 14, 196, 103488, 103488 }, false };
    params85.outDims = { 14, 14, 160, 1 };
    params85.outDesc = { 0,{ 14, 196, 31360, 31360 }, false };
    params85.convParams.filterSize = { 1, 1 };
    params85.convParams.padding = { 0, 0 };
    params85.convParams.stride = { 1, 1 };

    ConvolutionParams params86;
    params86.inputType = Datatype::F16;
    params86.inputLayout = DataLayout::bfyx;
    params86.outputLayout = DataLayout::bfyx;
    params86.activationFunc = ActivationFunction::RELU;
    params86.nlParams = { 0, 0 };
    params86.inDims = { 14, 14, 160, 1 };
    params86.inDesc = { 0,{ 14, 196, 31360, 31360 }, false };
    params86.outDims = { 14, 14, 320, 1 };
    params86.outDesc = { 0,{ 14, 196, 62720, 62720 }, false };
    params86.convParams.filterSize = { 3, 3 };
    params86.convParams.padding = { 1, 1 };
    params86.convParams.stride = { 1, 1 };

    ConvolutionParams params87;
    params87.inputType = Datatype::F16;
    params87.inputLayout = DataLayout::bfyx;
    params87.outputLayout = DataLayout::bfyx;
    params87.activationFunc = ActivationFunction::RELU;
    params87.nlParams = { 0, 0 };
    params87.inDims = { 14, 14, 160, 1 };
    params87.inDesc = { 17,{ 16, 256, 40960, 40960 }, false };
    params87.outDims = { 14, 14, 320, 1 };
    params87.outDesc = { 0,{ 14, 196, 62720, 62720 }, false };
    params87.convParams.filterSize = { 3, 3 };
    params87.convParams.padding = { 1, 1 };
    params87.convParams.stride = { 1, 1 };

    ConvolutionParams params88;
    params88.inputType = Datatype::F16;
    params88.inputLayout = DataLayout::bfyx;
    params88.outputLayout = DataLayout::bfyx;
    params88.activationFunc = ActivationFunction::RELU;
    params88.nlParams = { 0, 0 };
    params88.inDims = { 14, 14, 528, 1 };
    params88.inDesc = { 0,{ 14, 196, 103488, 103488 }, false };
    params88.outDims = { 14, 14, 256, 1 };
    params88.outDesc = { 0,{ 14, 196, 50176, 50176 }, false };
    params88.convParams.filterSize = { 1, 1 };
    params88.convParams.padding = { 0, 0 };
    params88.convParams.stride = { 1, 1 };

    ConvolutionParams params89;
    params89.inputType = Datatype::F16;
    params89.inputLayout = DataLayout::bfyx;
    params89.outputLayout = DataLayout::bfyx;
    params89.activationFunc = ActivationFunction::RELU;
    params89.nlParams = { 0, 0 };
    params89.inDims = { 7, 7, 832, 1 };
    params89.inDesc = { 0,{ 8, 56, 46592, 46592 }, false };
    params89.outDims = { 7, 7, 256, 1 };
    params89.outDesc = { 0,{ 8, 56, 14336, 14336 }, false };
    params89.convParams.filterSize = { 1, 1 };
    params89.convParams.padding = { 0, 0 };
    params89.convParams.stride = { 1, 1 };

    ConvolutionParams params90;
    params90.inputType = Datatype::F16;
    params90.inputLayout = DataLayout::bfyx;
    params90.outputLayout = DataLayout::bfyx;
    params90.activationFunc = ActivationFunction::RELU;
    params90.nlParams = { 0, 0 };
    params90.inDims = { 7, 7, 832, 1 };
    params90.inDesc = { 0,{ 8, 56, 46592, 46592 }, false };
    params90.outDims = { 7, 7, 160, 1 };
    params90.outDesc = { 0,{ 8, 56, 8960, 8960 }, false };
    params90.convParams.filterSize = { 1, 1 };
    params90.convParams.padding = { 0, 0 };
    params90.convParams.stride = { 1, 1 };

    ConvolutionParams params91;
    params91.inputType = Datatype::F16;
    params91.inputLayout = DataLayout::bfyx;
    params91.outputLayout = DataLayout::bfyx;
    params91.activationFunc = ActivationFunction::RELU;
    params91.nlParams = { 0, 0 };
    params91.inDims = { 7, 7, 832, 1 };
    params91.inDesc = { 0,{ 8, 56, 46592, 46592 }, false };
    params91.outDims = { 7, 7, 32, 1 };
    params91.outDesc = { 0,{ 8, 56, 1792, 1792 }, false };
    params91.convParams.filterSize = { 1, 1 };
    params91.convParams.padding = { 0, 0 };
    params91.convParams.stride = { 1, 1 };

    ConvolutionParams params92;
    params92.inputType = Datatype::F16;
    params92.inputLayout = DataLayout::bfyx;
    params92.outputLayout = DataLayout::bfyx;
    params92.activationFunc = ActivationFunction::RELU;
    params92.nlParams = { 0, 0 };
    params92.inDims = { 7, 7, 160, 1 };
    params92.inDesc = { 0,{ 8, 56, 8960, 8960 }, false };
    params92.outDims = { 7, 7, 320, 1 };
    params92.outDesc = { 0,{ 8, 56, 17920, 17920 }, false };
    params92.convParams.filterSize = { 3, 3 };
    params92.convParams.padding = { 1, 1 };
    params92.convParams.stride = { 1, 1 };

    ConvolutionParams params93;
    params93.inputType = Datatype::F16;
    params93.inputLayout = DataLayout::bfyx;
    params93.outputLayout = DataLayout::bfyx;
    params93.activationFunc = ActivationFunction::RELU;
    params93.nlParams = { 0, 0 };
    params93.inDims = { 7, 7, 160, 1 };
    params93.inDesc = { 11,{ 10, 90, 14400, 14400 }, false };
    params93.outDims = { 7, 7, 320, 1 };
    params93.outDesc = { 0,{ 8, 56, 17920, 17920 }, false };
    params93.convParams.filterSize = { 3, 3 };
    params93.convParams.padding = { 1, 1 };
    params93.convParams.stride = { 1, 1 };

    ConvolutionParams params94;
    params94.inputType = Datatype::F16;
    params94.inputLayout = DataLayout::bfyx;
    params94.outputLayout = DataLayout::bfyx;
    params94.activationFunc = ActivationFunction::RELU;
    params94.nlParams = { 0, 0 };
    params94.inDims = { 7, 7, 32, 1 };
    params94.inDesc = { 0,{ 8, 56, 1792, 1792 }, false };
    params94.outDims = { 7, 7, 128, 1 };
    params94.outDesc = { 0,{ 8, 56, 7168, 7168 }, false };
    params94.convParams.filterSize = { 5, 5 };
    params94.convParams.padding = { 2, 2 };
    params94.convParams.stride = { 1, 1 };

    ConvolutionParams params95;
    params95.inputType = Datatype::F16;
    params95.inputLayout = DataLayout::bfyx;
    params95.outputLayout = DataLayout::bfyx;
    params95.activationFunc = ActivationFunction::RELU;
    params95.nlParams = { 0, 0 };
    params95.inDims = { 7, 7, 32, 1 };
    params95.inDesc = { 26,{ 12, 132, 4224, 4224 }, false };
    params95.outDims = { 7, 7, 128, 1 };
    params95.outDesc = { 0,{ 8, 56, 7168, 7168 }, false };
    params95.convParams.filterSize = { 5, 5 };
    params95.convParams.padding = { 2, 2 };
    params95.convParams.stride = { 1, 1 };

    ConvolutionParams params96;
    params96.inputType = Datatype::F16;
    params96.inputLayout = DataLayout::bfyx;
    params96.outputLayout = DataLayout::bfyx;
    params96.activationFunc = ActivationFunction::RELU;
    params96.nlParams = { 0, 0 };
    params96.inDims = { 7, 7, 832, 1 };
    params96.inDesc = { 0,{ 8, 56, 46592, 46592 }, false };
    params96.outDims = { 7, 7, 128, 1 };
    params96.outDesc = { 0,{ 8, 56, 7168, 7168 }, false };
    params96.convParams.filterSize = { 1, 1 };
    params96.convParams.padding = { 0, 0 };
    params96.convParams.stride = { 1, 1 };

    ConvolutionParams params97;
    params97.inputType = Datatype::F16;
    params97.inputLayout = DataLayout::bfyx;
    params97.outputLayout = DataLayout::bfyx;
    params97.activationFunc = ActivationFunction::RELU;
    params97.nlParams = { 0, 0 };
    params97.inDims = { 7, 7, 832, 1 };
    params97.inDesc = { 0,{ 8, 56, 46592, 46592 }, false };
    params97.outDims = { 7, 7, 48, 1 };
    params97.outDesc = { 0,{ 8, 56, 2688, 2688 }, false };
    params97.convParams.filterSize = { 1, 1 };
    params97.convParams.padding = { 0, 0 };
    params97.convParams.stride = { 1, 1 };

    ConvolutionParams params98;
    params98.inputType = Datatype::F16;
    params98.inputLayout = DataLayout::bfyx;
    params98.outputLayout = DataLayout::bfyx;
    params98.activationFunc = ActivationFunction::RELU;
    params98.nlParams = { 0, 0 };
    params98.inDims = { 7, 7, 48, 1 };
    params98.inDesc = { 0,{ 8, 56, 2688, 2688 }, false };
    params98.outDims = { 7, 7, 128, 1 };
    params98.outDesc = { 0,{ 8, 56, 7168, 7168 }, false };
    params98.convParams.filterSize = { 5, 5 };
    params98.convParams.padding = { 2, 2 };
    params98.convParams.stride = { 1, 1 };

    ConvolutionParams params99;
    params99.inputType = Datatype::F16;
    params99.inputLayout = DataLayout::bfyx;
    params99.outputLayout = DataLayout::bfyx;
    params99.activationFunc = ActivationFunction::RELU;
    params99.nlParams = { 0, 0 };
    params99.inDims = { 7, 7, 48, 1 };
    params99.inDesc = { 26,{ 12, 132, 6336, 6336 }, false };
    params99.outDims = { 7, 7, 128, 1 };
    params99.outDesc = { 0,{ 8, 56, 7168, 7168 }, false };
    params99.convParams.filterSize = { 5, 5 };
    params99.convParams.padding = { 2, 2 };
    params99.convParams.stride = { 1, 1 };

    ConvolutionParams params100;
    params100.inputType = Datatype::F16;
    params100.inputLayout = DataLayout::bfyx;
    params100.outputLayout = DataLayout::bfyx;
    params100.activationFunc = ActivationFunction::RELU;
    params100.nlParams = { 0, 0 };
    params100.inDims = { 7, 7, 832, 1 };
    params100.inDesc = { 0,{ 8, 56, 46592, 46592 }, false };
    params100.outDims = { 7, 7, 192, 1 };
    params100.outDesc = { 0,{ 8, 56, 10752, 10752 }, false };
    params100.convParams.filterSize = { 1, 1 };
    params100.convParams.padding = { 0, 0 };
    params100.convParams.stride = { 1, 1 };

    ConvolutionParams params101;
    params101.inputType = Datatype::F16;
    params101.inputLayout = DataLayout::bfyx;
    params101.outputLayout = DataLayout::bfyx;
    params101.activationFunc = ActivationFunction::RELU;
    params101.nlParams = { 0, 0 };
    params101.inDims = { 7, 7, 192, 1 };
    params101.inDesc = { 0,{ 8, 56, 10752, 10752 }, false };
    params101.outDims = { 7, 7, 384, 1 };
    params101.outDesc = { 0,{ 8, 56, 21504, 21504 }, false };
    params101.convParams.filterSize = { 3, 3 };
    params101.convParams.padding = { 1, 1 };
    params101.convParams.stride = { 1, 1 };

    ConvolutionParams params102;
    params102.inputType = Datatype::F16;
    params102.inputLayout = DataLayout::bfyx;
    params102.outputLayout = DataLayout::bfyx;
    params102.activationFunc = ActivationFunction::RELU;
    params102.nlParams = { 0, 0 };
    params102.inDims = { 7, 7, 192, 1 };
    params102.inDesc = { 11,{ 10, 90, 17280, 17280 }, false };
    params102.outDims = { 7, 7, 384, 1 };
    params102.outDesc = { 0,{ 8, 56, 21504, 21504 }, false };
    params102.convParams.filterSize = { 3, 3 };
    params102.convParams.padding = { 1, 1 };
    params102.convParams.stride = { 1, 1 };

    ConvolutionParams params103;
    params103.inputType = Datatype::F16;
    params103.inputLayout = DataLayout::bfyx;
    params103.outputLayout = DataLayout::bfyx;
    params103.activationFunc = ActivationFunction::RELU;
    params103.nlParams = { 0, 0 };
    params103.inDims = { 7, 7, 832, 1 };
    params103.inDesc = { 0,{ 8, 56, 46592, 46592 }, false };
    params103.outDims = { 7, 7, 384, 1 };
    params103.outDesc = { 0,{ 8, 56, 21504, 21504 }, false };
    params103.convParams.filterSize = { 1, 1 };
    params103.convParams.padding = { 0, 0 };
    params103.convParams.stride = { 1, 1 };

    ConvolutionParams params104;
    params104.inputType = Datatype::F16;
    params104.inputLayout = DataLayout::bfyx;
    params104.outputLayout = DataLayout::bfyx;
    params104.activationFunc = ActivationFunction::RELU;
    params104.nlParams = { 0, 0 };
    params104.inDims = { 27, 27, 48, 1 };
    params104.inDesc = { 64,{ 31, 961, 46128, 46128 }, false };
    params104.outDims = { 27, 27, 128, 1 };
    params104.outDesc = { 0,{ 28, 756, 96768, 96768 }, false };
    params104.convParams.filterSize = { 5, 5 };
    params104.convParams.padding = { 2, 2 };
    params104.convParams.stride = { 1, 1 };

    ConvolutionParams params105;
    params105.inputType = Datatype::F16;
    params105.inputLayout = DataLayout::bfyx;
    params105.outputLayout = DataLayout::bfyx;
    params105.activationFunc = ActivationFunction::RELU;
    params105.nlParams = { 0, 0 };
    params105.inDims = { 13, 13, 256, 1 };
    params105.inDesc = { 16,{ 15, 225, 57600, 57600 }, false };
    params105.outDims = { 13, 13, 384, 1 };
    params105.outDesc = { 0,{ 14, 182, 69888, 69888 }, false };
    params105.convParams.filterSize = { 3, 3 };
    params105.convParams.padding = { 1, 1 };
    params105.convParams.stride = { 1, 1 };

    ConvolutionParams params106;
    params106.inputType = Datatype::F16;
    params106.inputLayout = DataLayout::bfyx;
    params106.outputLayout = DataLayout::bfyx;
    params106.activationFunc = ActivationFunction::RELU;
    params106.nlParams = { 0, 0 };
    params106.inDims = { 13, 13, 192, 1 };
    params106.inDesc = { 16,{ 15, 225, 43200, 43200 }, false };
    params106.outDims = { 13, 13, 192, 1 };
    params106.outDesc = { 0,{ 14, 182, 34944, 34944 }, false };
    params106.convParams.filterSize = { 3, 3 };
    params106.convParams.padding = { 1, 1 };
    params106.convParams.stride = { 1, 1 };

    ConvolutionParams params107;
    params107.inputType = Datatype::F16;
    params107.inputLayout = DataLayout::bfyx;
    params107.outputLayout = DataLayout::bfyx;
    params107.activationFunc = ActivationFunction::RELU;
    params107.nlParams = { 0, 0 };
    params107.inDims = { 13, 13, 192, 1 };
    params107.inDesc = { 16,{ 15, 225, 43200, 43200 }, false };
    params107.outDims = { 13, 13, 128, 1 };
    params107.outDesc = { 0,{ 14, 182, 23296, 23296 }, false };
    params107.convParams.filterSize = { 3, 3 };
    params107.convParams.padding = { 1, 1 };
    params107.convParams.stride = { 1, 1 };

    ConvolutionParams params108;
    params108.inputType = Datatype::F16;
    params108.inputLayout = DataLayout::bfyx;
    params108.outputLayout = DataLayout::bfyx;
    params108.activationFunc = ActivationFunction::RELU;
    params108.nlParams = { 0, 0 };
    params108.inDims = { 7, 7, 160, 1 };
    params108.inDesc = { 10,{ 9, 81, 12960, 12960 }, false };
    params108.outDims = { 7, 7, 320, 1 };
    params108.outDesc = { 0,{ 8, 56, 17920, 17920 }, false };
    params108.convParams.filterSize = { 3, 3 };
    params108.convParams.padding = { 1, 1 };
    params108.convParams.stride = { 1, 1 };

    ConvolutionParams params109;
    params109.inputType = Datatype::F16;
    params109.inputLayout = DataLayout::bfyx;
    params109.outputLayout = DataLayout::bfyx;
    params109.activationFunc = ActivationFunction::RELU;
    params109.nlParams = { 0, 0 };
    params109.inDims = { 7, 7, 32, 1 };
    params109.inDesc = { 24,{ 11, 121, 3872, 3872 }, false };
    params109.outDims = { 7, 7, 128, 1 };
    params109.outDesc = { 0,{ 8, 56, 7168, 7168 }, false };
    params109.convParams.filterSize = { 5, 5 };
    params109.convParams.padding = { 2, 2 };
    params109.convParams.stride = { 1, 1 };

    ConvolutionParams params110;
    params110.inputType = Datatype::F16;
    params110.inputLayout = DataLayout::bfyx;
    params110.outputLayout = DataLayout::bfyx;
    params110.activationFunc = ActivationFunction::RELU;
    params110.nlParams = { 0, 0 };
    params110.inDims = { 7, 7, 48, 1 };
    params110.inDesc = { 24,{ 11, 121, 5808, 5808 }, false };
    params110.outDims = { 7, 7, 128, 1 };
    params110.outDesc = { 0,{ 8, 56, 7168, 7168 }, false };
    params110.convParams.filterSize = { 5, 5 };
    params110.convParams.padding = { 2, 2 };
    params110.convParams.stride = { 1, 1 };

    ConvolutionParams params111;
    params111.inputType = Datatype::F16;
    params111.inputLayout = DataLayout::bfyx;
    params111.outputLayout = DataLayout::bfyx;
    params111.activationFunc = ActivationFunction::RELU;
    params111.nlParams = { 0, 0 };
    params111.inDims = { 7, 7, 192, 1 };
    params111.inDesc = { 10,{ 9, 81, 15552, 15552 }, false };
    params111.outDims = { 7, 7, 384, 1 };
    params111.outDesc = { 0,{ 8, 56, 21504, 21504 }, false };
    params111.convParams.filterSize = { 3, 3 };
    params111.convParams.padding = { 1, 1 };
    params111.convParams.stride = { 1, 1 };

    ConvolutionParams params112;
    params112.inputType = Datatype::F16;
    params112.inputLayout = DataLayout::bfyx;
    params112.outputLayout = DataLayout::bfyx;
    params112.activationFunc = ActivationFunction::RELU;
    params112.nlParams = { 0, 0 };
    params112.inDims = { 1000, 600, 3, 1 };
    params112.inDesc = { 0,{ 1000, 600000, 1800000, 1800000 }, false };
    params112.outDims = { 500, 300, 96, 1 };
    params112.outDesc = { 0,{ 500, 150000, 14400000, 14400000 }, false };
    params112.convParams.filterSize = { 7, 7 };
    params112.convParams.padding = { 3, 3 };
    params112.convParams.stride = { 2, 2 };

    ConvolutionParams params113;
    params113.inputType = Datatype::F16;
    params113.inputLayout = DataLayout::bfyx;
    params113.outputLayout = DataLayout::bfyx;
    params113.activationFunc = ActivationFunction::RELU;
    params113.nlParams = { 0, 0 };
    params113.inDims = { 1000, 600, 3, 1 };
    params113.inDesc = { 3021,{ 1006, 609636, 1828908, 1828908 }, false };
    params113.outDims = { 500, 300, 96, 1 };
    params113.outDesc = { 0,{ 500, 150000, 14400000, 14400000 }, false };
    params113.convParams.filterSize = { 7, 7 };
    params113.convParams.padding = { 3, 3 };
    params113.convParams.stride = { 2, 2 };

    ConvolutionParams params114;
    params114.inputType = Datatype::F16;
    params114.inputLayout = DataLayout::bfyx;
    params114.outputLayout = DataLayout::bfyx;
    params114.activationFunc = ActivationFunction::RELU;
    params114.nlParams = { 0, 0 };
    params114.inDims = { 251, 151, 96, 1 };
    params114.inDesc = { 0,{ 252, 38052, 3652992, 3652992 }, false };
    params114.outDims = { 126, 76, 256, 1 };
    params114.outDesc = { 0,{ 126, 9576, 2451456, 2451456 }, false };
    params114.convParams.filterSize = { 5, 5 };
    params114.convParams.padding = { 2, 2 };
    params114.convParams.stride = { 2, 2 };

    ConvolutionParams params115;
    params115.inputType = Datatype::F16;
    params115.inputLayout = DataLayout::bfyx;
    params115.outputLayout = DataLayout::bfyx;
    params115.activationFunc = ActivationFunction::RELU;
    params115.nlParams = { 0, 0 };
    params115.inDims = { 251, 151, 96, 1 };
    params115.inDesc = { 512,{ 255, 39525, 3794400, 3794400 }, false };
    params115.outDims = { 126, 76, 256, 1 };
    params115.outDesc = { 0,{ 126, 9576, 2451456, 2451456 }, false };
    params115.convParams.filterSize = { 5, 5 };
    params115.convParams.padding = { 2, 2 };
    params115.convParams.stride = { 2, 2 };

    ConvolutionParams params116;
    params116.inputType = Datatype::F16;
    params116.inputLayout = DataLayout::bfyx;
    params116.outputLayout = DataLayout::bfyx;
    params116.activationFunc = ActivationFunction::RELU;
    params116.nlParams = { 0, 0 };
    params116.inDims = { 64, 39, 256, 1 };
    params116.inDesc = { 0,{ 64, 2496, 638976, 638976 }, false };
    params116.outDims = { 64, 39, 384, 1 };
    params116.outDesc = { 0,{ 64, 2496, 958464, 958464 }, false };
    params116.convParams.filterSize = { 3, 3 };
    params116.convParams.padding = { 1, 1 };
    params116.convParams.stride = { 1, 1 };

    ConvolutionParams params117;
    params117.inputType = Datatype::F16;
    params117.inputLayout = DataLayout::bfyx;
    params117.outputLayout = DataLayout::bfyx;
    params117.activationFunc = ActivationFunction::RELU;
    params117.nlParams = { 0, 0 };
    params117.inDims = { 64, 39, 256, 1 };
    params117.inDesc = { 67,{ 66, 2706, 692736, 692736 }, false };
    params117.outDims = { 64, 39, 384, 1 };
    params117.outDesc = { 0,{ 64, 2496, 958464, 958464 }, false };
    params117.convParams.filterSize = { 3, 3 };
    params117.convParams.padding = { 1, 1 };
    params117.convParams.stride = { 1, 1 };

    ConvolutionParams params118;
    params118.inputType = Datatype::F16;
    params118.inputLayout = DataLayout::bfyx;
    params118.outputLayout = DataLayout::bfyx;
    params118.activationFunc = ActivationFunction::RELU;
    params118.nlParams = { 0, 0 };
    params118.inDims = { 64, 39, 384, 1 };
    params118.inDesc = { 0,{ 64, 2496, 958464, 958464 }, false };
    params118.outDims = { 64, 39, 384, 1 };
    params118.outDesc = { 0,{ 64, 2496, 958464, 958464 }, false };
    params118.convParams.filterSize = { 3, 3 };
    params118.convParams.padding = { 1, 1 };
    params118.convParams.stride = { 1, 1 };

    ConvolutionParams params119;
    params119.inputType = Datatype::F16;
    params119.inputLayout = DataLayout::bfyx;
    params119.outputLayout = DataLayout::bfyx;
    params119.activationFunc = ActivationFunction::RELU;
    params119.nlParams = { 0, 0 };
    params119.inDims = { 64, 39, 384, 1 };
    params119.inDesc = { 67,{ 66, 2706, 1039104, 1039104 }, false };
    params119.outDims = { 64, 39, 384, 1 };
    params119.outDesc = { 0,{ 64, 2496, 958464, 958464 }, false };
    params119.convParams.filterSize = { 3, 3 };
    params119.convParams.padding = { 1, 1 };
    params119.convParams.stride = { 1, 1 };

    ConvolutionParams params120;
    params120.inputType = Datatype::F16;
    params120.inputLayout = DataLayout::bfyx;
    params120.outputLayout = DataLayout::bfyx;
    params120.activationFunc = ActivationFunction::RELU;
    params120.nlParams = { 0, 0 };
    params120.inDims = { 64, 39, 384, 1 };
    params120.inDesc = { 0,{ 64, 2496, 958464, 958464 }, false };
    params120.outDims = { 64, 39, 256, 1 };
    params120.outDesc = { 0,{ 64, 2496, 638976, 638976 }, false };
    params120.convParams.filterSize = { 3, 3 };
    params120.convParams.padding = { 1, 1 };
    params120.convParams.stride = { 1, 1 };

    ConvolutionParams params121;
    params121.inputType = Datatype::F16;
    params121.inputLayout = DataLayout::bfyx;
    params121.outputLayout = DataLayout::bfyx;
    params121.activationFunc = ActivationFunction::RELU;
    params121.nlParams = { 0, 0 };
    params121.inDims = { 64, 39, 384, 1 };
    params121.inDesc = { 67,{ 66, 2706, 1039104, 1039104 }, false };
    params121.outDims = { 64, 39, 256, 1 };
    params121.outDesc = { 0,{ 64, 2496, 638976, 638976 }, false };
    params121.convParams.filterSize = { 3, 3 };
    params121.convParams.padding = { 1, 1 };
    params121.convParams.stride = { 1, 1 };

    ConvolutionParams params122;
    params122.inputType = Datatype::F16;
    params122.inputLayout = DataLayout::bfyx;
    params122.outputLayout = DataLayout::bfyx;
    params122.activationFunc = ActivationFunction::RELU;
    params122.nlParams = { 0, 0 };
    params122.inDims = { 64, 39, 1024, 1 };
    params122.inDesc = { 0,{ 64, 2496, 2555904, 2555904 }, false };
    params122.outDims = { 64, 39, 256, 1 };
    params122.outDesc = { 0,{ 64, 2496, 638976, 638976 }, false };
    params122.convParams.filterSize = { 3, 3 };
    params122.convParams.padding = { 1, 1 };
    params122.convParams.stride = { 1, 1 };

    ConvolutionParams params123;
    params123.inputType = Datatype::F16;
    params123.inputLayout = DataLayout::bfyx;
    params123.outputLayout = DataLayout::bfyx;
    params123.activationFunc = ActivationFunction::RELU;
    params123.nlParams = { 0, 0 };
    params123.inDims = { 64, 39, 1024, 1 };
    params123.inDesc = { 67,{ 66, 2706, 2770944, 2770944 }, false };
    params123.outDims = { 64, 39, 256, 1 };
    params123.outDesc = { 0,{ 64, 2496, 638976, 638976 }, false };
    params123.convParams.filterSize = { 3, 3 };
    params123.convParams.padding = { 1, 1 };
    params123.convParams.stride = { 1, 1 };

    ConvolutionParams params124;
    params124.inputType = Datatype::F16;
    params124.inputLayout = DataLayout::bfyx;
    params124.outputLayout = DataLayout::bfyx;
    params124.activationFunc = ActivationFunction::RELU;
    params124.nlParams = { 0, 0 };
    params124.inDims = { 64, 39, 256, 1 };
    params124.inDesc = { 0,{ 64, 2496, 638976, 638976 }, false };
    params124.outDims = { 64, 39, 256, 1 };
    params124.outDesc = { 0,{ 64, 2496, 638976, 638976 }, false };
    params124.convParams.filterSize = { 3, 3 };
    params124.convParams.padding = { 1, 1 };
    params124.convParams.stride = { 1, 1 };

    ConvolutionParams params125;
    params125.inputType = Datatype::F16;
    params125.inputLayout = DataLayout::bfyx;
    params125.outputLayout = DataLayout::bfyx;
    params125.activationFunc = ActivationFunction::RELU;
    params125.nlParams = { 0, 0 };
    params125.inDims = { 64, 39, 256, 1 };
    params125.inDesc = { 67,{ 66, 2706, 692736, 692736 }, false };
    params125.outDims = { 64, 39, 256, 1 };
    params125.outDesc = { 0,{ 64, 2496, 638976, 638976 }, false };
    params125.convParams.filterSize = { 3, 3 };
    params125.convParams.padding = { 1, 1 };
    params125.convParams.stride = { 1, 1 };

    ConvolutionParams params126;
    params126.inputType = Datatype::F16;
    params126.inputLayout = DataLayout::bfyx;
    params126.outputLayout = DataLayout::bfyx;
    params126.activationFunc = ActivationFunction::NONE;
    params126.nlParams = { 1, 0 };
    params126.inDims = { 64, 39, 256, 1 };
    params126.inDesc = { 0,{ 64, 2496, 638976, 638976 }, false };
    params126.outDims = { 64, 39, 36, 1 };
    params126.outDesc = { 0,{ 64, 2496, 89856, 89856 }, false };
    params126.convParams.filterSize = { 1, 1 };
    params126.convParams.padding = { 0, 0 };
    params126.convParams.stride = { 1, 1 };

    ConvolutionParams params127;
    params127.inputType = Datatype::F16;
    params127.inputLayout = DataLayout::bfyx;
    params127.outputLayout = DataLayout::bfyx;
    params127.activationFunc = ActivationFunction::NONE;
    params127.nlParams = { 1, 0 };
    params127.inDims = { 64, 39, 256, 1 };
    params127.inDesc = { 0,{ 64, 2496, 638976, 638976 }, false };
    params127.outDims = { 64, 39, 9, 1 };
    params127.outDesc = { 0,{ 64, 2496, 22464, 22464 }, false };
    params127.convParams.filterSize = { 1, 1 };
    params127.convParams.padding = { 0, 0 };
    params127.convParams.stride = { 1, 1 };

    params_vec = {
        params0,
        params1,
        params2,
        params3,
        params4,
        params5,
        params6,
        params7,
        params8,
        params9,
        params10,
        params11,
        params12,
        params13,
        params14,
        params15,
        params16,
        params17,
        params18,
        params19,
        params20,
        params21,
        params22,
        params23,
        params24,
        params25,
        params26,
        params27,
        params28,
        params29,
        params30,
        params31,
        params32,
        params33,
        params34,
        params35,
        params36,
        params37,
        params38,
        params39,
        params40,
        params41,
        params42,
        params43,
        params44,
        params45,
        params46,
        params47,
        params48,
        params49,
        params50,
        params51,
        params52,
        params53,
        params54,
        params55,
        params56,
        params57,
        params58,
        params59,
        params60,
        params61,
        params62,
        params63,
        params64,
        params65,
        params66,
        params67,
        params68,
        params69,
        params70,
        params71,
        params72,
        params73,
        params74,
        params75,
        params76,
        params77,
        params78,
        params79,
        params80,
        params81,
        params82,
        params83,
        params84,
        params85,
        params86,
        params87,
        params88,
        params89,
        params90,
        params91,
        params92,
        params93,
        params94,
        params95,
        params96,
        params97,
        params98,
        params99,
        params100,
        params101,
        params102,
        params103,
        params104,
        params105,
        params106,
        params107,
        params108,
        params109,
        params110,
        params111,
        params112,
        params113,
        params114,
        params115,
        params116,
        params117,
        params118,
        params119,
        params120,
        params121,
        params122,
        params123,
        params124,
        params125,
        params126,
        params127,
    };
}