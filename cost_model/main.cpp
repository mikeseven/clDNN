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
#include "ks_ocl_toolkit.h"
#include "kernel_base.h"
#include "kernel_selector.h"
#include "kernel_selector_params.h"
#include "actual_kernels/convolution/convolution_kernel_selector.h"

using namespace KernelSelector;
using gpu_toolkit = KernelSelector::gpu::gpu_toolkit;

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
    program_cache m_binaryManager;
    std::shared_ptr<gpu_toolkit> gpu_context;

public:

    CLRunner()
    {
        KernelSelector::gpu::configuration cfg;
        cfg.enable_profiling = true;
        gpu_context = std::make_shared<gpu_toolkit>(cfg);
    }

    float run(const KernelData& kernelData)
    {
        cl_int status = CL_SUCCESS;
        const clKernelData& clData = kernelData.kernels[0];
        binary_data binary = clData.GetBinary({ gpu_context->context(), gpu_context->device() }, m_binaryManager);
        cl::Program::Binaries binaries;
        binaries.push_back(binary);
        auto devices = std::vector<cl::Device>(1, gpu_context->device());
        auto& clContext = gpu_context->context();
        cl::Program program(gpu_context->context(), devices, binaries, NULL, &status);

        CL_CHECK(status, "Error: cannot create program");

        cl::Kernel clKernel(program, clData.kernelString.entry_point.c_str(), &status);
        CL_CHECK(status, "Error: cannot create kernel");

        auto& newParams = *static_cast<ConvolutionParams*>(kernelData.params.get());
        std::size_t weightsSize =
            kernelData.weightsReorderParams.engine != WeightsReorderParams::Engine::NONE ?
            kernelData.weightsReorderParams.newBufferSize :
            newParams.convParams.filterSize.x * newParams.convParams.filterSize.y * newParams.inputs[0].Feature().v * newParams.output.Feature().v;
        cl::Buffer input(clContext, CL_MEM_READ_WRITE, newParams.inputs[0].PhysicalSize(), nullptr, &status);
        cl::Buffer output(clContext, CL_MEM_READ_WRITE, newParams.output.PhysicalSize(), nullptr, &status);
        cl::Buffer weights(clContext, CL_MEM_READ_WRITE, weightsSize, nullptr, &status);
        cl::Buffer bias(clContext, CL_MEM_READ_WRITE, newParams.output.Feature().v, nullptr, &status);

        ArgumentDescpirtor::SetArgumentParams params;
        params.inputs.push_back(&input);
        params.output = &output;
        params.weights = &weights;
        params.bias = &bias;
        if (!clData.argsDesc.SetArguments(clKernel, params))
        {
            printf("Error: setting args\n");
            return 0.f;
        }

        const uint32_t warm_up = 3;
        const uint32_t iteration = 100;
        std::vector<cl::Event> events;
        events.resize(iteration);

        for (uint32_t i = 0; i < warm_up; i++)
        {
            status = gpu_context->queue().enqueueNDRangeKernel(
                clKernel,
                cl::NullRange,
                clData.workGroups.global,
                clData.workGroups.local,
                nullptr);
            CL_CHECK(status, "Error: enqueue failed");
        }

        for (auto& event : events)
        {
            status = gpu_context->queue().enqueueNDRangeKernel(
                clKernel,
                cl::NullRange,
                clData.workGroups.global,
                clData.workGroups.local,
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

    std::vector<uint32_t> GetKernels()
    {
        std::vector<uint32_t> optional;
        std::vector<uint32_t> force;

        for (std::size_t i = 0; i < implementations.size(); i++)
        {
            const auto& implementation = implementations[i];
            const auto& it = forceKernels.find(implementation->GetName());
            if (it != forceKernels.end())
            {
                if (it->second == true)
                {
                    //std::cout << "Force: " << it->first << std::endl;
                    force.push_back((uint32_t)i);
                }
                else
                {
                    //std::cout << "Deny: " << it->first << std::endl;
                }
            }
            else
            {
                optional.push_back((uint32_t)i);
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
        InitConvParams();

        for (uint32_t i : GetKernels())
        {
            const auto& impl = implementations[i];
            const auto& kernel = *impl.get();

#if RUN_MODE == 1
            printf("%s\n", kernel.GetName().c_str());
            printf("static std::map<std::size_t, float> known_params_time = {\n");
#endif

            ConvolutionOptionalParams optParams;
            optParams.allowPadding = true;
            optParams.allowWeightsReorder = true;
            optParams.bSupportSubGroupExt = true;

            for (const auto& params : params_vec)
            {
                std::string param_str = params.to_string();
                //printf("%s\n", param_str.c_str());

                float avg_time = NOT_SUPPORTED;

                const auto impl_key = kernel.GetSupportedKey();
                const auto params_key = params.GetParamsKey().Merge(optParams.GetSupportedKey());
                if (impl_key.Support(params_key))
                {
                    ConvolutionOptionalParams optParams;
                    optParams.allowPadding = true;
                    optParams.allowWeightsReorder = true;
                    optParams.bSupportSubGroupExt = true;

                    KernelsData kernelsData = kernel.GetKernelsData(params, optParams);
                    if (kernelsData.size() && kernelsData[0].kernels.size())
                    {
                        avg_time = cl_runner.run(kernelsData[0]);
#if RUN_MODE == 2
                        float diff = abs((float)(avg_time - kernelsData[0].estimatedTime));
                        if (diff > 0.1)
                        {
                            printf("ERROR: bad value (%f, %f, %f) - %s\n", avg_time, kernelsData[0].estimatedTime, diff, param_str.c_str());
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
}

void InitConvParams()
{

    ConvolutionParams params0;
    params0.inputs.resize(1);
    params0.layerID = "params0";
    params0.inputs[0] = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 227, 227, 3, 1 } };
    params0.output = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 55, 55, 96, 1 } };
    params0.weights = { WeightsType::F16, WeightsLayout::yxio, PADDED_VAL::ZERO, 0,{ 11, 11, 3, 96 } };
    params0.bias.resize(1);
    params0.bias[0] = { Datatype::F16, DataLayout::bf, PADDED_VAL::ZERO, 0,{ 96, 1 } };
    params0.activationFunc = ActivationFunction::RELU;
    params0.nlParams = { 1, 0 };
    params0.convParams.filterSize = { 11, 11 };
    params0.convParams.padding = { 0, 0 };
    params0.convParams.stride = { 4, 4 };
    params0.convParams.dilation = { 1, 1 };
    params0.convParams.split = 1;

    ConvolutionParams params1;
    params1.inputs.resize(1);
    params1.layerID = "params1";
    params1.inputs[0] = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 27, 27, 48, 1 } };
    params1.output = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 27, 27, 128, 1 } };
    params1.weights = { WeightsType::F16, WeightsLayout::yxio, PADDED_VAL::ZERO, 0,{ 5, 5, 48, 128 } };
    params1.bias.resize(1);
    params1.bias[0] = { Datatype::F16, DataLayout::bf, PADDED_VAL::ZERO, 0,{ 128, 1 } };
    params1.activationFunc = ActivationFunction::RELU;
    params1.nlParams = { 1, 0 };
    params1.convParams.filterSize = { 5, 5 };
    params1.convParams.padding = { 2, 2 };
    params1.convParams.stride = { 1, 1 };
    params1.convParams.dilation = { 1, 1 };
    params1.convParams.split = 1;

    ConvolutionParams params2;
    params2.inputs.resize(1);
    params2.layerID = "params2";
    params2.inputs[0] = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 13, 13, 256, 1 } };
    params2.output = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 13, 13, 384, 1 } };
    params2.weights = { WeightsType::F16, WeightsLayout::yxio, PADDED_VAL::ZERO, 0,{ 3, 3, 256, 384 } };
    params2.bias.resize(1);
    params2.bias[0] = { Datatype::F16, DataLayout::bf, PADDED_VAL::ZERO, 0,{ 384, 1 } };
    params2.activationFunc = ActivationFunction::RELU;
    params2.nlParams = { 1, 0 };
    params2.convParams.filterSize = { 3, 3 };
    params2.convParams.padding = { 1, 1 };
    params2.convParams.stride = { 1, 1 };
    params2.convParams.dilation = { 1, 1 };
    params2.convParams.split = 1;

    ConvolutionParams params3;
    params3.inputs.resize(1);
    params3.layerID = "params3";
    params3.inputs[0] = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 13, 13, 192, 1 } };
    params3.output = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 13, 13, 192, 1 } };
    params3.weights = { WeightsType::F16, WeightsLayout::yxio, PADDED_VAL::ZERO, 0,{ 3, 3, 192, 192 } };
    params3.bias.resize(1);
    params3.bias[0] = { Datatype::F16, DataLayout::bf, PADDED_VAL::ZERO, 0,{ 192, 1 } };
    params3.activationFunc = ActivationFunction::RELU;
    params3.nlParams = { 1, 0 };
    params3.convParams.filterSize = { 3, 3 };
    params3.convParams.padding = { 1, 1 };
    params3.convParams.stride = { 1, 1 };
    params3.convParams.dilation = { 1, 1 };
    params3.convParams.split = 1;

    ConvolutionParams params4;
    params4.inputs.resize(1);
    params4.layerID = "params4";
    params4.inputs[0] = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 13, 13, 192, 1 } };
    params4.output = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 13, 13, 128, 1 } };
    params4.weights = { WeightsType::F16, WeightsLayout::yxio, PADDED_VAL::ZERO, 0,{ 3, 3, 192, 128 } };
    params4.bias.resize(1);
    params4.bias[0] = { Datatype::F16, DataLayout::bf, PADDED_VAL::ZERO, 0,{ 128, 1 } };
    params4.activationFunc = ActivationFunction::RELU;
    params4.nlParams = { 1, 0 };
    params4.convParams.filterSize = { 3, 3 };
    params4.convParams.padding = { 1, 1 };
    params4.convParams.stride = { 1, 1 };
    params4.convParams.dilation = { 1, 1 };
    params4.convParams.split = 1;

    ConvolutionParams params5;
    params5.inputs.resize(1);
    params5.layerID = "params5";
    params5.inputs[0] = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 224, 224, 3, 1 } };
    params5.output = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 112, 112, 64, 1 } };
    params5.weights = { WeightsType::F16, WeightsLayout::oiyx, PADDED_VAL::ZERO, 0,{ 7, 7, 3, 64 } };
    params5.bias.resize(1);
    params5.bias[0] = { Datatype::F16, DataLayout::bf, PADDED_VAL::ZERO, 0,{ 64, 1 } };
    params5.activationFunc = ActivationFunction::RELU;
    params5.nlParams = { 1, 0 };
    params5.convParams.filterSize = { 7, 7 };
    params5.convParams.padding = { 3, 3 };
    params5.convParams.stride = { 2, 2 };
    params5.convParams.dilation = { 1, 1 };
    params5.convParams.split = 1;

    ConvolutionParams params6;
    params6.inputs.resize(1);
    params6.layerID = "params6";
    params6.inputs[0] = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 56, 56, 64, 1 } };
    params6.output = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 56, 56, 64, 1 } };
    params6.weights = { WeightsType::F16, WeightsLayout::oiyx, PADDED_VAL::ZERO, 0,{ 1, 1, 64, 64 } };
    params6.bias.resize(1);
    params6.bias[0] = { Datatype::F16, DataLayout::bf, PADDED_VAL::ZERO, 0,{ 64, 1 } };
    params6.activationFunc = ActivationFunction::RELU;
    params6.nlParams = { 1, 0 };
    params6.convParams.filterSize = { 1, 1 };
    params6.convParams.padding = { 0, 0 };
    params6.convParams.stride = { 1, 1 };
    params6.convParams.dilation = { 1, 1 };
    params6.convParams.split = 1;

    ConvolutionParams params7;
    params7.inputs.resize(1);
    params7.layerID = "params7";
    params7.inputs[0] = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 56, 56, 64, 1 } };
    params7.output = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 56, 56, 192, 1 } };
    params7.weights = { WeightsType::F16, WeightsLayout::oiyx, PADDED_VAL::ZERO, 0,{ 3, 3, 64, 192 } };
    params7.bias.resize(1);
    params7.bias[0] = { Datatype::F16, DataLayout::bf, PADDED_VAL::ZERO, 0,{ 192, 1 } };
    params7.activationFunc = ActivationFunction::RELU;
    params7.nlParams = { 1, 0 };
    params7.convParams.filterSize = { 3, 3 };
    params7.convParams.padding = { 1, 1 };
    params7.convParams.stride = { 1, 1 };
    params7.convParams.dilation = { 1, 1 };
    params7.convParams.split = 1;

    ConvolutionParams params8;
    params8.inputs.resize(1);
    params8.layerID = "params8";
    params8.inputs[0] = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 28, 28, 192, 1 } };
    params8.output = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 28, 28, 32, 1 } };
    params8.weights = { WeightsType::F16, WeightsLayout::oiyx, PADDED_VAL::ZERO, 0,{ 1, 1, 192, 32 } };
    params8.bias.resize(1);
    params8.bias[0] = { Datatype::F16, DataLayout::bf, PADDED_VAL::ZERO, 0,{ 32, 1 } };
    params8.activationFunc = ActivationFunction::RELU;
    params8.nlParams = { 1, 0 };
    params8.convParams.filterSize = { 1, 1 };
    params8.convParams.padding = { 0, 0 };
    params8.convParams.stride = { 1, 1 };
    params8.convParams.dilation = { 1, 1 };
    params8.convParams.split = 1;

    ConvolutionParams params9;
    params9.inputs.resize(1);
    params9.layerID = "params9";
    params9.inputs[0] = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 28, 28, 192, 1 } };
    params9.output = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 28, 28, 16, 1 } };
    params9.weights = { WeightsType::F16, WeightsLayout::oiyx, PADDED_VAL::ZERO, 0,{ 1, 1, 192, 16 } };
    params9.bias.resize(1);
    params9.bias[0] = { Datatype::F16, DataLayout::bf, PADDED_VAL::ZERO, 0,{ 16, 1 } };
    params9.activationFunc = ActivationFunction::RELU;
    params9.nlParams = { 1, 0 };
    params9.convParams.filterSize = { 1, 1 };
    params9.convParams.padding = { 0, 0 };
    params9.convParams.stride = { 1, 1 };
    params9.convParams.dilation = { 1, 1 };
    params9.convParams.split = 1;

    ConvolutionParams params10;
    params10.inputs.resize(1);
    params10.layerID = "params10";
    params10.inputs[0] = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 28, 28, 16, 1 } };
    params10.output = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 28, 28, 32, 1 } };
    params10.weights = { WeightsType::F16, WeightsLayout::oiyx, PADDED_VAL::ZERO, 0,{ 5, 5, 16, 32 } };
    params10.bias.resize(1);
    params10.bias[0] = { Datatype::F16, DataLayout::bf, PADDED_VAL::ZERO, 0,{ 32, 1 } };
    params10.activationFunc = ActivationFunction::RELU;
    params10.nlParams = { 1, 0 };
    params10.convParams.filterSize = { 5, 5 };
    params10.convParams.padding = { 2, 2 };
    params10.convParams.stride = { 1, 1 };
    params10.convParams.dilation = { 1, 1 };
    params10.convParams.split = 1;

    ConvolutionParams params11;
    params11.inputs.resize(1);
    params11.layerID = "params11";
    params11.inputs[0] = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 28, 28, 192, 1 } };
    params11.output = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 28, 28, 96, 1 } };
    params11.weights = { WeightsType::F16, WeightsLayout::oiyx, PADDED_VAL::ZERO, 0,{ 1, 1, 192, 96 } };
    params11.bias.resize(1);
    params11.bias[0] = { Datatype::F16, DataLayout::bf, PADDED_VAL::ZERO, 0,{ 96, 1 } };
    params11.activationFunc = ActivationFunction::RELU;
    params11.nlParams = { 1, 0 };
    params11.convParams.filterSize = { 1, 1 };
    params11.convParams.padding = { 0, 0 };
    params11.convParams.stride = { 1, 1 };
    params11.convParams.dilation = { 1, 1 };
    params11.convParams.split = 1;

    ConvolutionParams params12;
    params12.inputs.resize(1);
    params12.layerID = "params12";
    params12.inputs[0] = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 28, 28, 96, 1 } };
    params12.output = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 28, 28, 128, 1 } };
    params12.weights = { WeightsType::F16, WeightsLayout::oiyx, PADDED_VAL::ZERO, 0,{ 3, 3, 96, 128 } };
    params12.bias.resize(1);
    params12.bias[0] = { Datatype::F16, DataLayout::bf, PADDED_VAL::ZERO, 0,{ 128, 1 } };
    params12.activationFunc = ActivationFunction::RELU;
    params12.nlParams = { 1, 0 };
    params12.convParams.filterSize = { 3, 3 };
    params12.convParams.padding = { 1, 1 };
    params12.convParams.stride = { 1, 1 };
    params12.convParams.dilation = { 1, 1 };
    params12.convParams.split = 1;

    ConvolutionParams params13;
    params13.inputs.resize(1);
    params13.layerID = "params13";
    params13.inputs[0] = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 28, 28, 192, 1 } };
    params13.output = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 28, 28, 64, 1 } };
    params13.weights = { WeightsType::F16, WeightsLayout::oiyx, PADDED_VAL::ZERO, 0,{ 1, 1, 192, 64 } };
    params13.bias.resize(1);
    params13.bias[0] = { Datatype::F16, DataLayout::bf, PADDED_VAL::ZERO, 0,{ 64, 1 } };
    params13.activationFunc = ActivationFunction::RELU;
    params13.nlParams = { 1, 0 };
    params13.convParams.filterSize = { 1, 1 };
    params13.convParams.padding = { 0, 0 };
    params13.convParams.stride = { 1, 1 };
    params13.convParams.dilation = { 1, 1 };
    params13.convParams.split = 1;

    ConvolutionParams params14;
    params14.inputs.resize(1);
    params14.layerID = "params14";
    params14.inputs[0] = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 28, 28, 256, 1 } };
    params14.output = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 28, 28, 64, 1 } };
    params14.weights = { WeightsType::F16, WeightsLayout::oiyx, PADDED_VAL::ZERO, 0,{ 1, 1, 256, 64 } };
    params14.bias.resize(1);
    params14.bias[0] = { Datatype::F16, DataLayout::bf, PADDED_VAL::ZERO, 0,{ 64, 1 } };
    params14.activationFunc = ActivationFunction::RELU;
    params14.nlParams = { 1, 0 };
    params14.convParams.filterSize = { 1, 1 };
    params14.convParams.padding = { 0, 0 };
    params14.convParams.stride = { 1, 1 };
    params14.convParams.dilation = { 1, 1 };
    params14.convParams.split = 1;

    ConvolutionParams params15;
    params15.inputs.resize(1);
    params15.layerID = "params15";
    params15.inputs[0] = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 28, 28, 256, 1 } };
    params15.output = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 28, 28, 32, 1 } };
    params15.weights = { WeightsType::F16, WeightsLayout::oiyx, PADDED_VAL::ZERO, 0,{ 1, 1, 256, 32 } };
    params15.bias.resize(1);
    params15.bias[0] = { Datatype::F16, DataLayout::bf, PADDED_VAL::ZERO, 0,{ 32, 1 } };
    params15.activationFunc = ActivationFunction::RELU;
    params15.nlParams = { 1, 0 };
    params15.convParams.filterSize = { 1, 1 };
    params15.convParams.padding = { 0, 0 };
    params15.convParams.stride = { 1, 1 };
    params15.convParams.dilation = { 1, 1 };
    params15.convParams.split = 1;

    ConvolutionParams params16;
    params16.inputs.resize(1);
    params16.layerID = "params16";
    params16.inputs[0] = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 28, 28, 32, 1 } };
    params16.output = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 28, 28, 96, 1 } };
    params16.weights = { WeightsType::F16, WeightsLayout::oiyx, PADDED_VAL::ZERO, 0,{ 5, 5, 32, 96 } };
    params16.bias.resize(1);
    params16.bias[0] = { Datatype::F16, DataLayout::bf, PADDED_VAL::ZERO, 0,{ 96, 1 } };
    params16.activationFunc = ActivationFunction::RELU;
    params16.nlParams = { 1, 0 };
    params16.convParams.filterSize = { 5, 5 };
    params16.convParams.padding = { 2, 2 };
    params16.convParams.stride = { 1, 1 };
    params16.convParams.dilation = { 1, 1 };
    params16.convParams.split = 1;

    ConvolutionParams params17;
    params17.inputs.resize(1);
    params17.layerID = "params17";
    params17.inputs[0] = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 28, 28, 256, 1 } };
    params17.output = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 28, 28, 128, 1 } };
    params17.weights = { WeightsType::F16, WeightsLayout::oiyx, PADDED_VAL::ZERO, 0,{ 1, 1, 256, 128 } };
    params17.bias.resize(1);
    params17.bias[0] = { Datatype::F16, DataLayout::bf, PADDED_VAL::ZERO, 0,{ 128, 1 } };
    params17.activationFunc = ActivationFunction::RELU;
    params17.nlParams = { 1, 0 };
    params17.convParams.filterSize = { 1, 1 };
    params17.convParams.padding = { 0, 0 };
    params17.convParams.stride = { 1, 1 };
    params17.convParams.dilation = { 1, 1 };
    params17.convParams.split = 1;

    ConvolutionParams params18;
    params18.inputs.resize(1);
    params18.layerID = "params18";
    params18.inputs[0] = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 28, 28, 128, 1 } };
    params18.output = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 28, 28, 192, 1 } };
    params18.weights = { WeightsType::F16, WeightsLayout::oiyx, PADDED_VAL::ZERO, 0,{ 3, 3, 128, 192 } };
    params18.bias.resize(1);
    params18.bias[0] = { Datatype::F16, DataLayout::bf, PADDED_VAL::ZERO, 0,{ 192, 1 } };
    params18.activationFunc = ActivationFunction::RELU;
    params18.nlParams = { 1, 0 };
    params18.convParams.filterSize = { 3, 3 };
    params18.convParams.padding = { 1, 1 };
    params18.convParams.stride = { 1, 1 };
    params18.convParams.dilation = { 1, 1 };
    params18.convParams.split = 1;

    ConvolutionParams params19;
    params19.inputs.resize(1);
    params19.layerID = "params19";
    params19.inputs[0] = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 28, 28, 256, 1 } };
    params19.output = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 28, 28, 128, 1 } };
    params19.weights = { WeightsType::F16, WeightsLayout::oiyx, PADDED_VAL::ZERO, 0,{ 1, 1, 256, 128 } };
    params19.bias.resize(1);
    params19.bias[0] = { Datatype::F16, DataLayout::bf, PADDED_VAL::ZERO, 0,{ 128, 1 } };
    params19.activationFunc = ActivationFunction::RELU;
    params19.nlParams = { 1, 0 };
    params19.convParams.filterSize = { 1, 1 };
    params19.convParams.padding = { 0, 0 };
    params19.convParams.stride = { 1, 1 };
    params19.convParams.dilation = { 1, 1 };
    params19.convParams.split = 1;

    ConvolutionParams params20;
    params20.inputs.resize(1);
    params20.layerID = "params20";
    params20.inputs[0] = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 14, 14, 480, 1 } };
    params20.output = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 14, 14, 64, 1 } };
    params20.weights = { WeightsType::F16, WeightsLayout::oiyx, PADDED_VAL::ZERO, 0,{ 1, 1, 480, 64 } };
    params20.bias.resize(1);
    params20.bias[0] = { Datatype::F16, DataLayout::bf, PADDED_VAL::ZERO, 0,{ 64, 1 } };
    params20.activationFunc = ActivationFunction::RELU;
    params20.nlParams = { 1, 0 };
    params20.convParams.filterSize = { 1, 1 };
    params20.convParams.padding = { 0, 0 };
    params20.convParams.stride = { 1, 1 };
    params20.convParams.dilation = { 1, 1 };
    params20.convParams.split = 1;

    ConvolutionParams params21;
    params21.inputs.resize(1);
    params21.layerID = "params21";
    params21.inputs[0] = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 14, 14, 480, 1 } };
    params21.output = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 14, 14, 16, 1 } };
    params21.weights = { WeightsType::F16, WeightsLayout::oiyx, PADDED_VAL::ZERO, 0,{ 1, 1, 480, 16 } };
    params21.bias.resize(1);
    params21.bias[0] = { Datatype::F16, DataLayout::bf, PADDED_VAL::ZERO, 0,{ 16, 1 } };
    params21.activationFunc = ActivationFunction::RELU;
    params21.nlParams = { 1, 0 };
    params21.convParams.filterSize = { 1, 1 };
    params21.convParams.padding = { 0, 0 };
    params21.convParams.stride = { 1, 1 };
    params21.convParams.dilation = { 1, 1 };
    params21.convParams.split = 1;

    ConvolutionParams params22;
    params22.inputs.resize(1);
    params22.layerID = "params22";
    params22.inputs[0] = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 14, 14, 16, 1 } };
    params22.output = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 14, 14, 48, 1 } };
    params22.weights = { WeightsType::F16, WeightsLayout::oiyx, PADDED_VAL::ZERO, 0,{ 5, 5, 16, 48 } };
    params22.bias.resize(1);
    params22.bias[0] = { Datatype::F16, DataLayout::bf, PADDED_VAL::ZERO, 0,{ 48, 1 } };
    params22.activationFunc = ActivationFunction::RELU;
    params22.nlParams = { 1, 0 };
    params22.convParams.filterSize = { 5, 5 };
    params22.convParams.padding = { 2, 2 };
    params22.convParams.stride = { 1, 1 };
    params22.convParams.dilation = { 1, 1 };
    params22.convParams.split = 1;

    ConvolutionParams params23;
    params23.inputs.resize(1);
    params23.layerID = "params23";
    params23.inputs[0] = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 14, 14, 480, 1 } };
    params23.output = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 14, 14, 96, 1 } };
    params23.weights = { WeightsType::F16, WeightsLayout::oiyx, PADDED_VAL::ZERO, 0,{ 1, 1, 480, 96 } };
    params23.bias.resize(1);
    params23.bias[0] = { Datatype::F16, DataLayout::bf, PADDED_VAL::ZERO, 0,{ 96, 1 } };
    params23.activationFunc = ActivationFunction::RELU;
    params23.nlParams = { 1, 0 };
    params23.convParams.filterSize = { 1, 1 };
    params23.convParams.padding = { 0, 0 };
    params23.convParams.stride = { 1, 1 };
    params23.convParams.dilation = { 1, 1 };
    params23.convParams.split = 1;

    ConvolutionParams params24;
    params24.inputs.resize(1);
    params24.layerID = "params24";
    params24.inputs[0] = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 14, 14, 96, 1 } };
    params24.output = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 14, 14, 208, 1 } };
    params24.weights = { WeightsType::F16, WeightsLayout::oiyx, PADDED_VAL::ZERO, 0,{ 3, 3, 96, 208 } };
    params24.bias.resize(1);
    params24.bias[0] = { Datatype::F16, DataLayout::bf, PADDED_VAL::ZERO, 0,{ 208, 1 } };
    params24.activationFunc = ActivationFunction::RELU;
    params24.nlParams = { 1, 0 };
    params24.convParams.filterSize = { 3, 3 };
    params24.convParams.padding = { 1, 1 };
    params24.convParams.stride = { 1, 1 };
    params24.convParams.dilation = { 1, 1 };
    params24.convParams.split = 1;

    ConvolutionParams params25;
    params25.inputs.resize(1);
    params25.layerID = "params25";
    params25.inputs[0] = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 14, 14, 480, 1 } };
    params25.output = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 14, 14, 192, 1 } };
    params25.weights = { WeightsType::F16, WeightsLayout::oiyx, PADDED_VAL::ZERO, 0,{ 1, 1, 480, 192 } };
    params25.bias.resize(1);
    params25.bias[0] = { Datatype::F16, DataLayout::bf, PADDED_VAL::ZERO, 0,{ 192, 1 } };
    params25.activationFunc = ActivationFunction::RELU;
    params25.nlParams = { 1, 0 };
    params25.convParams.filterSize = { 1, 1 };
    params25.convParams.padding = { 0, 0 };
    params25.convParams.stride = { 1, 1 };
    params25.convParams.dilation = { 1, 1 };
    params25.convParams.split = 1;

    ConvolutionParams params26;
    params26.inputs.resize(1);
    params26.layerID = "params26";
    params26.inputs[0] = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 14, 14, 512, 1 } };
    params26.output = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 14, 14, 64, 1 } };
    params26.weights = { WeightsType::F16, WeightsLayout::oiyx, PADDED_VAL::ZERO, 0,{ 1, 1, 512, 64 } };
    params26.bias.resize(1);
    params26.bias[0] = { Datatype::F16, DataLayout::bf, PADDED_VAL::ZERO, 0,{ 64, 1 } };
    params26.activationFunc = ActivationFunction::RELU;
    params26.nlParams = { 1, 0 };
    params26.convParams.filterSize = { 1, 1 };
    params26.convParams.padding = { 0, 0 };
    params26.convParams.stride = { 1, 1 };
    params26.convParams.dilation = { 1, 1 };
    params26.convParams.split = 1;

    ConvolutionParams params27;
    params27.inputs.resize(1);
    params27.layerID = "params27";
    params27.inputs[0] = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 14, 14, 512, 1 } };
    params27.output = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 14, 14, 24, 1 } };
    params27.weights = { WeightsType::F16, WeightsLayout::oiyx, PADDED_VAL::ZERO, 0,{ 1, 1, 512, 24 } };
    params27.bias.resize(1);
    params27.bias[0] = { Datatype::F16, DataLayout::bf, PADDED_VAL::ZERO, 0,{ 24, 1 } };
    params27.activationFunc = ActivationFunction::RELU;
    params27.nlParams = { 1, 0 };
    params27.convParams.filterSize = { 1, 1 };
    params27.convParams.padding = { 0, 0 };
    params27.convParams.stride = { 1, 1 };
    params27.convParams.dilation = { 1, 1 };
    params27.convParams.split = 1;

    ConvolutionParams params28;
    params28.inputs.resize(1);
    params28.layerID = "params28";
    params28.inputs[0] = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 14, 14, 24, 1 } };
    params28.output = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 14, 14, 64, 1 } };
    params28.weights = { WeightsType::F16, WeightsLayout::oiyx, PADDED_VAL::ZERO, 0,{ 5, 5, 24, 64 } };
    params28.bias.resize(1);
    params28.bias[0] = { Datatype::F16, DataLayout::bf, PADDED_VAL::ZERO, 0,{ 64, 1 } };
    params28.activationFunc = ActivationFunction::RELU;
    params28.nlParams = { 1, 0 };
    params28.convParams.filterSize = { 5, 5 };
    params28.convParams.padding = { 2, 2 };
    params28.convParams.stride = { 1, 1 };
    params28.convParams.dilation = { 1, 1 };
    params28.convParams.split = 1;

    ConvolutionParams params29;
    params29.inputs.resize(1);
    params29.layerID = "params29";
    params29.inputs[0] = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 14, 14, 512, 1 } };
    params29.output = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 14, 14, 112, 1 } };
    params29.weights = { WeightsType::F16, WeightsLayout::oiyx, PADDED_VAL::ZERO, 0,{ 1, 1, 512, 112 } };
    params29.bias.resize(1);
    params29.bias[0] = { Datatype::F16, DataLayout::bf, PADDED_VAL::ZERO, 0,{ 112, 1 } };
    params29.activationFunc = ActivationFunction::RELU;
    params29.nlParams = { 1, 0 };
    params29.convParams.filterSize = { 1, 1 };
    params29.convParams.padding = { 0, 0 };
    params29.convParams.stride = { 1, 1 };
    params29.convParams.dilation = { 1, 1 };
    params29.convParams.split = 1;

    ConvolutionParams params30;
    params30.inputs.resize(1);
    params30.layerID = "params30";
    params30.inputs[0] = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 14, 14, 112, 1 } };
    params30.output = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 14, 14, 224, 1 } };
    params30.weights = { WeightsType::F16, WeightsLayout::oiyx, PADDED_VAL::ZERO, 0,{ 3, 3, 112, 224 } };
    params30.bias.resize(1);
    params30.bias[0] = { Datatype::F16, DataLayout::bf, PADDED_VAL::ZERO, 0,{ 224, 1 } };
    params30.activationFunc = ActivationFunction::RELU;
    params30.nlParams = { 1, 0 };
    params30.convParams.filterSize = { 3, 3 };
    params30.convParams.padding = { 1, 1 };
    params30.convParams.stride = { 1, 1 };
    params30.convParams.dilation = { 1, 1 };
    params30.convParams.split = 1;

    ConvolutionParams params31;
    params31.inputs.resize(1);
    params31.layerID = "params31";
    params31.inputs[0] = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 14, 14, 512, 1 } };
    params31.output = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 14, 14, 160, 1 } };
    params31.weights = { WeightsType::F16, WeightsLayout::oiyx, PADDED_VAL::ZERO, 0,{ 1, 1, 512, 160 } };
    params31.bias.resize(1);
    params31.bias[0] = { Datatype::F16, DataLayout::bf, PADDED_VAL::ZERO, 0,{ 160, 1 } };
    params31.activationFunc = ActivationFunction::RELU;
    params31.nlParams = { 1, 0 };
    params31.convParams.filterSize = { 1, 1 };
    params31.convParams.padding = { 0, 0 };
    params31.convParams.stride = { 1, 1 };
    params31.convParams.dilation = { 1, 1 };
    params31.convParams.split = 1;

    ConvolutionParams params32;
    params32.inputs.resize(1);
    params32.layerID = "params32";
    params32.inputs[0] = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 14, 14, 512, 1 } };
    params32.output = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 14, 14, 64, 1 } };
    params32.weights = { WeightsType::F16, WeightsLayout::oiyx, PADDED_VAL::ZERO, 0,{ 1, 1, 512, 64 } };
    params32.bias.resize(1);
    params32.bias[0] = { Datatype::F16, DataLayout::bf, PADDED_VAL::ZERO, 0,{ 64, 1 } };
    params32.activationFunc = ActivationFunction::RELU;
    params32.nlParams = { 1, 0 };
    params32.convParams.filterSize = { 1, 1 };
    params32.convParams.padding = { 0, 0 };
    params32.convParams.stride = { 1, 1 };
    params32.convParams.dilation = { 1, 1 };
    params32.convParams.split = 1;

    ConvolutionParams params33;
    params33.inputs.resize(1);
    params33.layerID = "params33";
    params33.inputs[0] = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 14, 14, 512, 1 } };
    params33.output = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 14, 14, 24, 1 } };
    params33.weights = { WeightsType::F16, WeightsLayout::oiyx, PADDED_VAL::ZERO, 0,{ 1, 1, 512, 24 } };
    params33.bias.resize(1);
    params33.bias[0] = { Datatype::F16, DataLayout::bf, PADDED_VAL::ZERO, 0,{ 24, 1 } };
    params33.activationFunc = ActivationFunction::RELU;
    params33.nlParams = { 1, 0 };
    params33.convParams.filterSize = { 1, 1 };
    params33.convParams.padding = { 0, 0 };
    params33.convParams.stride = { 1, 1 };
    params33.convParams.dilation = { 1, 1 };
    params33.convParams.split = 1;

    ConvolutionParams params34;
    params34.inputs.resize(1);
    params34.layerID = "params34";
    params34.inputs[0] = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 14, 14, 24, 1 } };
    params34.output = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 14, 14, 64, 1 } };
    params34.weights = { WeightsType::F16, WeightsLayout::oiyx, PADDED_VAL::ZERO, 0,{ 5, 5, 24, 64 } };
    params34.bias.resize(1);
    params34.bias[0] = { Datatype::F16, DataLayout::bf, PADDED_VAL::ZERO, 0,{ 64, 1 } };
    params34.activationFunc = ActivationFunction::RELU;
    params34.nlParams = { 1, 0 };
    params34.convParams.filterSize = { 5, 5 };
    params34.convParams.padding = { 2, 2 };
    params34.convParams.stride = { 1, 1 };
    params34.convParams.dilation = { 1, 1 };
    params34.convParams.split = 1;

    ConvolutionParams params35;
    params35.inputs.resize(1);
    params35.layerID = "params35";
    params35.inputs[0] = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 14, 14, 512, 1 } };
    params35.output = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 14, 14, 128, 1 } };
    params35.weights = { WeightsType::F16, WeightsLayout::oiyx, PADDED_VAL::ZERO, 0,{ 1, 1, 512, 128 } };
    params35.bias.resize(1);
    params35.bias[0] = { Datatype::F16, DataLayout::bf, PADDED_VAL::ZERO, 0,{ 128, 1 } };
    params35.activationFunc = ActivationFunction::RELU;
    params35.nlParams = { 1, 0 };
    params35.convParams.filterSize = { 1, 1 };
    params35.convParams.padding = { 0, 0 };
    params35.convParams.stride = { 1, 1 };
    params35.convParams.dilation = { 1, 1 };
    params35.convParams.split = 1;

    ConvolutionParams params36;
    params36.inputs.resize(1);
    params36.layerID = "params36";
    params36.inputs[0] = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 14, 14, 128, 1 } };
    params36.output = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 14, 14, 256, 1 } };
    params36.weights = { WeightsType::F16, WeightsLayout::oiyx, PADDED_VAL::ZERO, 0,{ 3, 3, 128, 256 } };
    params36.bias.resize(1);
    params36.bias[0] = { Datatype::F16, DataLayout::bf, PADDED_VAL::ZERO, 0,{ 256, 1 } };
    params36.activationFunc = ActivationFunction::RELU;
    params36.nlParams = { 1, 0 };
    params36.convParams.filterSize = { 3, 3 };
    params36.convParams.padding = { 1, 1 };
    params36.convParams.stride = { 1, 1 };
    params36.convParams.dilation = { 1, 1 };
    params36.convParams.split = 1;

    ConvolutionParams params37;
    params37.inputs.resize(1);
    params37.layerID = "params37";
    params37.inputs[0] = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 14, 14, 512, 1 } };
    params37.output = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 14, 14, 128, 1 } };
    params37.weights = { WeightsType::F16, WeightsLayout::oiyx, PADDED_VAL::ZERO, 0,{ 1, 1, 512, 128 } };
    params37.bias.resize(1);
    params37.bias[0] = { Datatype::F16, DataLayout::bf, PADDED_VAL::ZERO, 0,{ 128, 1 } };
    params37.activationFunc = ActivationFunction::RELU;
    params37.nlParams = { 1, 0 };
    params37.convParams.filterSize = { 1, 1 };
    params37.convParams.padding = { 0, 0 };
    params37.convParams.stride = { 1, 1 };
    params37.convParams.dilation = { 1, 1 };
    params37.convParams.split = 1;

    ConvolutionParams params38;
    params38.inputs.resize(1);
    params38.layerID = "params38";
    params38.inputs[0] = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 14, 14, 512, 1 } };
    params38.output = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 14, 14, 64, 1 } };
    params38.weights = { WeightsType::F16, WeightsLayout::oiyx, PADDED_VAL::ZERO, 0,{ 1, 1, 512, 64 } };
    params38.bias.resize(1);
    params38.bias[0] = { Datatype::F16, DataLayout::bf, PADDED_VAL::ZERO, 0,{ 64, 1 } };
    params38.activationFunc = ActivationFunction::RELU;
    params38.nlParams = { 1, 0 };
    params38.convParams.filterSize = { 1, 1 };
    params38.convParams.padding = { 0, 0 };
    params38.convParams.stride = { 1, 1 };
    params38.convParams.dilation = { 1, 1 };
    params38.convParams.split = 1;

    ConvolutionParams params39;
    params39.inputs.resize(1);
    params39.layerID = "params39";
    params39.inputs[0] = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 14, 14, 512, 1 } };
    params39.output = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 14, 14, 32, 1 } };
    params39.weights = { WeightsType::F16, WeightsLayout::oiyx, PADDED_VAL::ZERO, 0,{ 1, 1, 512, 32 } };
    params39.bias.resize(1);
    params39.bias[0] = { Datatype::F16, DataLayout::bf, PADDED_VAL::ZERO, 0,{ 32, 1 } };
    params39.activationFunc = ActivationFunction::RELU;
    params39.nlParams = { 1, 0 };
    params39.convParams.filterSize = { 1, 1 };
    params39.convParams.padding = { 0, 0 };
    params39.convParams.stride = { 1, 1 };
    params39.convParams.dilation = { 1, 1 };
    params39.convParams.split = 1;

    ConvolutionParams params40;
    params40.inputs.resize(1);
    params40.layerID = "params40";
    params40.inputs[0] = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 14, 14, 32, 1 } };
    params40.output = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 14, 14, 64, 1 } };
    params40.weights = { WeightsType::F16, WeightsLayout::oiyx, PADDED_VAL::ZERO, 0,{ 5, 5, 32, 64 } };
    params40.bias.resize(1);
    params40.bias[0] = { Datatype::F16, DataLayout::bf, PADDED_VAL::ZERO, 0,{ 64, 1 } };
    params40.activationFunc = ActivationFunction::RELU;
    params40.nlParams = { 1, 0 };
    params40.convParams.filterSize = { 5, 5 };
    params40.convParams.padding = { 2, 2 };
    params40.convParams.stride = { 1, 1 };
    params40.convParams.dilation = { 1, 1 };
    params40.convParams.split = 1;

    ConvolutionParams params41;
    params41.inputs.resize(1);
    params41.layerID = "params41";
    params41.inputs[0] = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 14, 14, 512, 1 } };
    params41.output = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 14, 14, 144, 1 } };
    params41.weights = { WeightsType::F16, WeightsLayout::oiyx, PADDED_VAL::ZERO, 0,{ 1, 1, 512, 144 } };
    params41.bias.resize(1);
    params41.bias[0] = { Datatype::F16, DataLayout::bf, PADDED_VAL::ZERO, 0,{ 144, 1 } };
    params41.activationFunc = ActivationFunction::RELU;
    params41.nlParams = { 1, 0 };
    params41.convParams.filterSize = { 1, 1 };
    params41.convParams.padding = { 0, 0 };
    params41.convParams.stride = { 1, 1 };
    params41.convParams.dilation = { 1, 1 };
    params41.convParams.split = 1;

    ConvolutionParams params42;
    params42.inputs.resize(1);
    params42.layerID = "params42";
    params42.inputs[0] = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 14, 14, 144, 1 } };
    params42.output = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 14, 14, 288, 1 } };
    params42.weights = { WeightsType::F16, WeightsLayout::oiyx, PADDED_VAL::ZERO, 0,{ 3, 3, 144, 288 } };
    params42.bias.resize(1);
    params42.bias[0] = { Datatype::F16, DataLayout::bf, PADDED_VAL::ZERO, 0,{ 288, 1 } };
    params42.activationFunc = ActivationFunction::RELU;
    params42.nlParams = { 1, 0 };
    params42.convParams.filterSize = { 3, 3 };
    params42.convParams.padding = { 1, 1 };
    params42.convParams.stride = { 1, 1 };
    params42.convParams.dilation = { 1, 1 };
    params42.convParams.split = 1;

    ConvolutionParams params43;
    params43.inputs.resize(1);
    params43.layerID = "params43";
    params43.inputs[0] = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 14, 14, 512, 1 } };
    params43.output = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 14, 14, 112, 1 } };
    params43.weights = { WeightsType::F16, WeightsLayout::oiyx, PADDED_VAL::ZERO, 0,{ 1, 1, 512, 112 } };
    params43.bias.resize(1);
    params43.bias[0] = { Datatype::F16, DataLayout::bf, PADDED_VAL::ZERO, 0,{ 112, 1 } };
    params43.activationFunc = ActivationFunction::RELU;
    params43.nlParams = { 1, 0 };
    params43.convParams.filterSize = { 1, 1 };
    params43.convParams.padding = { 0, 0 };
    params43.convParams.stride = { 1, 1 };
    params43.convParams.dilation = { 1, 1 };
    params43.convParams.split = 1;

    ConvolutionParams params44;
    params44.inputs.resize(1);
    params44.layerID = "params44";
    params44.inputs[0] = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 14, 14, 528, 1 } };
    params44.output = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 14, 14, 128, 1 } };
    params44.weights = { WeightsType::F16, WeightsLayout::oiyx, PADDED_VAL::ZERO, 0,{ 1, 1, 528, 128 } };
    params44.bias.resize(1);
    params44.bias[0] = { Datatype::F16, DataLayout::bf, PADDED_VAL::ZERO, 0,{ 128, 1 } };
    params44.activationFunc = ActivationFunction::RELU;
    params44.nlParams = { 1, 0 };
    params44.convParams.filterSize = { 1, 1 };
    params44.convParams.padding = { 0, 0 };
    params44.convParams.stride = { 1, 1 };
    params44.convParams.dilation = { 1, 1 };
    params44.convParams.split = 1;

    ConvolutionParams params45;
    params45.inputs.resize(1);
    params45.layerID = "params45";
    params45.inputs[0] = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 14, 14, 528, 1 } };
    params45.output = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 14, 14, 32, 1 } };
    params45.weights = { WeightsType::F16, WeightsLayout::oiyx, PADDED_VAL::ZERO, 0,{ 1, 1, 528, 32 } };
    params45.bias.resize(1);
    params45.bias[0] = { Datatype::F16, DataLayout::bf, PADDED_VAL::ZERO, 0,{ 32, 1 } };
    params45.activationFunc = ActivationFunction::RELU;
    params45.nlParams = { 1, 0 };
    params45.convParams.filterSize = { 1, 1 };
    params45.convParams.padding = { 0, 0 };
    params45.convParams.stride = { 1, 1 };
    params45.convParams.dilation = { 1, 1 };
    params45.convParams.split = 1;

    ConvolutionParams params46;
    params46.inputs.resize(1);
    params46.layerID = "params46";
    params46.inputs[0] = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 14, 14, 32, 1 } };
    params46.output = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 14, 14, 128, 1 } };
    params46.weights = { WeightsType::F16, WeightsLayout::oiyx, PADDED_VAL::ZERO, 0,{ 5, 5, 32, 128 } };
    params46.bias.resize(1);
    params46.bias[0] = { Datatype::F16, DataLayout::bf, PADDED_VAL::ZERO, 0,{ 128, 1 } };
    params46.activationFunc = ActivationFunction::RELU;
    params46.nlParams = { 1, 0 };
    params46.convParams.filterSize = { 5, 5 };
    params46.convParams.padding = { 2, 2 };
    params46.convParams.stride = { 1, 1 };
    params46.convParams.dilation = { 1, 1 };
    params46.convParams.split = 1;

    ConvolutionParams params47;
    params47.inputs.resize(1);
    params47.layerID = "params47";
    params47.inputs[0] = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 14, 14, 528, 1 } };
    params47.output = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 14, 14, 160, 1 } };
    params47.weights = { WeightsType::F16, WeightsLayout::oiyx, PADDED_VAL::ZERO, 0,{ 1, 1, 528, 160 } };
    params47.bias.resize(1);
    params47.bias[0] = { Datatype::F16, DataLayout::bf, PADDED_VAL::ZERO, 0,{ 160, 1 } };
    params47.activationFunc = ActivationFunction::RELU;
    params47.nlParams = { 1, 0 };
    params47.convParams.filterSize = { 1, 1 };
    params47.convParams.padding = { 0, 0 };
    params47.convParams.stride = { 1, 1 };
    params47.convParams.dilation = { 1, 1 };
    params47.convParams.split = 1;

    ConvolutionParams params48;
    params48.inputs.resize(1);
    params48.layerID = "params48";
    params48.inputs[0] = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 14, 14, 160, 1 } };
    params48.output = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 14, 14, 320, 1 } };
    params48.weights = { WeightsType::F16, WeightsLayout::oiyx, PADDED_VAL::ZERO, 0,{ 3, 3, 160, 320 } };
    params48.bias.resize(1);
    params48.bias[0] = { Datatype::F16, DataLayout::bf, PADDED_VAL::ZERO, 0,{ 320, 1 } };
    params48.activationFunc = ActivationFunction::RELU;
    params48.nlParams = { 1, 0 };
    params48.convParams.filterSize = { 3, 3 };
    params48.convParams.padding = { 1, 1 };
    params48.convParams.stride = { 1, 1 };
    params48.convParams.dilation = { 1, 1 };
    params48.convParams.split = 1;

    ConvolutionParams params49;
    params49.inputs.resize(1);
    params49.layerID = "params49";
    params49.inputs[0] = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 14, 14, 528, 1 } };
    params49.output = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 14, 14, 256, 1 } };
    params49.weights = { WeightsType::F16, WeightsLayout::oiyx, PADDED_VAL::ZERO, 0,{ 1, 1, 528, 256 } };
    params49.bias.resize(1);
    params49.bias[0] = { Datatype::F16, DataLayout::bf, PADDED_VAL::ZERO, 0,{ 256, 1 } };
    params49.activationFunc = ActivationFunction::RELU;
    params49.nlParams = { 1, 0 };
    params49.convParams.filterSize = { 1, 1 };
    params49.convParams.padding = { 0, 0 };
    params49.convParams.stride = { 1, 1 };
    params49.convParams.dilation = { 1, 1 };
    params49.convParams.split = 1;

    ConvolutionParams params50;
    params50.inputs.resize(1);
    params50.layerID = "params50";
    params50.inputs[0] = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 7, 7, 832, 1 } };
    params50.output = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 7, 7, 128, 1 } };
    params50.weights = { WeightsType::F16, WeightsLayout::oiyx, PADDED_VAL::ZERO, 0,{ 1, 1, 832, 128 } };
    params50.bias.resize(1);
    params50.bias[0] = { Datatype::F16, DataLayout::bf, PADDED_VAL::ZERO, 0,{ 128, 1 } };
    params50.activationFunc = ActivationFunction::RELU;
    params50.nlParams = { 1, 0 };
    params50.convParams.filterSize = { 1, 1 };
    params50.convParams.padding = { 0, 0 };
    params50.convParams.stride = { 1, 1 };
    params50.convParams.dilation = { 1, 1 };
    params50.convParams.split = 1;

    ConvolutionParams params51;
    params51.inputs.resize(1);
    params51.layerID = "params51";
    params51.inputs[0] = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 7, 7, 832, 1 } };
    params51.output = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 7, 7, 32, 1 } };
    params51.weights = { WeightsType::F16, WeightsLayout::oiyx, PADDED_VAL::ZERO, 0,{ 1, 1, 832, 32 } };
    params51.bias.resize(1);
    params51.bias[0] = { Datatype::F16, DataLayout::bf, PADDED_VAL::ZERO, 0,{ 32, 1 } };
    params51.activationFunc = ActivationFunction::RELU;
    params51.nlParams = { 1, 0 };
    params51.convParams.filterSize = { 1, 1 };
    params51.convParams.padding = { 0, 0 };
    params51.convParams.stride = { 1, 1 };
    params51.convParams.dilation = { 1, 1 };
    params51.convParams.split = 1;

    ConvolutionParams params52;
    params52.inputs.resize(1);
    params52.layerID = "params52";
    params52.inputs[0] = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 7, 7, 32, 1 } };
    params52.output = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 7, 7, 128, 1 } };
    params52.weights = { WeightsType::F16, WeightsLayout::oiyx, PADDED_VAL::ZERO, 0,{ 5, 5, 32, 128 } };
    params52.bias.resize(1);
    params52.bias[0] = { Datatype::F16, DataLayout::bf, PADDED_VAL::ZERO, 0,{ 128, 1 } };
    params52.activationFunc = ActivationFunction::RELU;
    params52.nlParams = { 1, 0 };
    params52.convParams.filterSize = { 5, 5 };
    params52.convParams.padding = { 2, 2 };
    params52.convParams.stride = { 1, 1 };
    params52.convParams.dilation = { 1, 1 };
    params52.convParams.split = 1;

    ConvolutionParams params53;
    params53.inputs.resize(1);
    params53.layerID = "params53";
    params53.inputs[0] = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 7, 7, 832, 1 } };
    params53.output = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 7, 7, 160, 1 } };
    params53.weights = { WeightsType::F16, WeightsLayout::oiyx, PADDED_VAL::ZERO, 0,{ 1, 1, 832, 160 } };
    params53.bias.resize(1);
    params53.bias[0] = { Datatype::F16, DataLayout::bf, PADDED_VAL::ZERO, 0,{ 160, 1 } };
    params53.activationFunc = ActivationFunction::RELU;
    params53.nlParams = { 1, 0 };
    params53.convParams.filterSize = { 1, 1 };
    params53.convParams.padding = { 0, 0 };
    params53.convParams.stride = { 1, 1 };
    params53.convParams.dilation = { 1, 1 };
    params53.convParams.split = 1;

    ConvolutionParams params54;
    params54.inputs.resize(1);
    params54.layerID = "params54";
    params54.inputs[0] = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 7, 7, 160, 1 } };
    params54.output = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 7, 7, 320, 1 } };
    params54.weights = { WeightsType::F16, WeightsLayout::oiyx, PADDED_VAL::ZERO, 0,{ 3, 3, 160, 320 } };
    params54.bias.resize(1);
    params54.bias[0] = { Datatype::F16, DataLayout::bf, PADDED_VAL::ZERO, 0,{ 320, 1 } };
    params54.activationFunc = ActivationFunction::RELU;
    params54.nlParams = { 1, 0 };
    params54.convParams.filterSize = { 3, 3 };
    params54.convParams.padding = { 1, 1 };
    params54.convParams.stride = { 1, 1 };
    params54.convParams.dilation = { 1, 1 };
    params54.convParams.split = 1;

    ConvolutionParams params55;
    params55.inputs.resize(1);
    params55.layerID = "params55";
    params55.inputs[0] = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 7, 7, 832, 1 } };
    params55.output = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 7, 7, 256, 1 } };
    params55.weights = { WeightsType::F16, WeightsLayout::oiyx, PADDED_VAL::ZERO, 0,{ 1, 1, 832, 256 } };
    params55.bias.resize(1);
    params55.bias[0] = { Datatype::F16, DataLayout::bf, PADDED_VAL::ZERO, 0,{ 256, 1 } };
    params55.activationFunc = ActivationFunction::RELU;
    params55.nlParams = { 1, 0 };
    params55.convParams.filterSize = { 1, 1 };
    params55.convParams.padding = { 0, 0 };
    params55.convParams.stride = { 1, 1 };
    params55.convParams.dilation = { 1, 1 };
    params55.convParams.split = 1;

    ConvolutionParams params56;
    params56.inputs.resize(1);
    params56.layerID = "params56";
    params56.inputs[0] = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 7, 7, 832, 1 } };
    params56.output = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 7, 7, 128, 1 } };
    params56.weights = { WeightsType::F16, WeightsLayout::oiyx, PADDED_VAL::ZERO, 0,{ 1, 1, 832, 128 } };
    params56.bias.resize(1);
    params56.bias[0] = { Datatype::F16, DataLayout::bf, PADDED_VAL::ZERO, 0,{ 128, 1 } };
    params56.activationFunc = ActivationFunction::RELU;
    params56.nlParams = { 1, 0 };
    params56.convParams.filterSize = { 1, 1 };
    params56.convParams.padding = { 0, 0 };
    params56.convParams.stride = { 1, 1 };
    params56.convParams.dilation = { 1, 1 };
    params56.convParams.split = 1;

    ConvolutionParams params57;
    params57.inputs.resize(1);
    params57.layerID = "params57";
    params57.inputs[0] = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 7, 7, 832, 1 } };
    params57.output = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 7, 7, 48, 1 } };
    params57.weights = { WeightsType::F16, WeightsLayout::oiyx, PADDED_VAL::ZERO, 0,{ 1, 1, 832, 48 } };
    params57.bias.resize(1);
    params57.bias[0] = { Datatype::F16, DataLayout::bf, PADDED_VAL::ZERO, 0,{ 48, 1 } };
    params57.activationFunc = ActivationFunction::RELU;
    params57.nlParams = { 1, 0 };
    params57.convParams.filterSize = { 1, 1 };
    params57.convParams.padding = { 0, 0 };
    params57.convParams.stride = { 1, 1 };
    params57.convParams.dilation = { 1, 1 };
    params57.convParams.split = 1;

    ConvolutionParams params58;
    params58.inputs.resize(1);
    params58.layerID = "params58";
    params58.inputs[0] = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 7, 7, 48, 1 } };
    params58.output = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 7, 7, 128, 1 } };
    params58.weights = { WeightsType::F16, WeightsLayout::oiyx, PADDED_VAL::ZERO, 0,{ 5, 5, 48, 128 } };
    params58.bias.resize(1);
    params58.bias[0] = { Datatype::F16, DataLayout::bf, PADDED_VAL::ZERO, 0,{ 128, 1 } };
    params58.activationFunc = ActivationFunction::RELU;
    params58.nlParams = { 1, 0 };
    params58.convParams.filterSize = { 5, 5 };
    params58.convParams.padding = { 2, 2 };
    params58.convParams.stride = { 1, 1 };
    params58.convParams.dilation = { 1, 1 };
    params58.convParams.split = 1;

    ConvolutionParams params59;
    params59.inputs.resize(1);
    params59.layerID = "params59";
    params59.inputs[0] = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 7, 7, 832, 1 } };
    params59.output = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 7, 7, 192, 1 } };
    params59.weights = { WeightsType::F16, WeightsLayout::oiyx, PADDED_VAL::ZERO, 0,{ 1, 1, 832, 192 } };
    params59.bias.resize(1);
    params59.bias[0] = { Datatype::F16, DataLayout::bf, PADDED_VAL::ZERO, 0,{ 192, 1 } };
    params59.activationFunc = ActivationFunction::RELU;
    params59.nlParams = { 1, 0 };
    params59.convParams.filterSize = { 1, 1 };
    params59.convParams.padding = { 0, 0 };
    params59.convParams.stride = { 1, 1 };
    params59.convParams.dilation = { 1, 1 };
    params59.convParams.split = 1;

    ConvolutionParams params60;
    params60.inputs.resize(1);
    params60.layerID = "params60";
    params60.inputs[0] = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 7, 7, 192, 1 } };
    params60.output = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 7, 7, 384, 1 } };
    params60.weights = { WeightsType::F16, WeightsLayout::oiyx, PADDED_VAL::ZERO, 0,{ 3, 3, 192, 384 } };
    params60.bias.resize(1);
    params60.bias[0] = { Datatype::F16, DataLayout::bf, PADDED_VAL::ZERO, 0,{ 384, 1 } };
    params60.activationFunc = ActivationFunction::RELU;
    params60.nlParams = { 1, 0 };
    params60.convParams.filterSize = { 3, 3 };
    params60.convParams.padding = { 1, 1 };
    params60.convParams.stride = { 1, 1 };
    params60.convParams.dilation = { 1, 1 };
    params60.convParams.split = 1;

    ConvolutionParams params61;
    params61.inputs.resize(1);
    params61.layerID = "params61";
    params61.inputs[0] = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 7, 7, 832, 1 } };
    params61.output = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 7, 7, 384, 1 } };
    params61.weights = { WeightsType::F16, WeightsLayout::oiyx, PADDED_VAL::ZERO, 0,{ 1, 1, 832, 384 } };
    params61.bias.resize(1);
    params61.bias[0] = { Datatype::F16, DataLayout::bf, PADDED_VAL::ZERO, 0,{ 384, 1 } };
    params61.activationFunc = ActivationFunction::RELU;
    params61.nlParams = { 1, 0 };
    params61.convParams.filterSize = { 1, 1 };
    params61.convParams.padding = { 0, 0 };
    params61.convParams.stride = { 1, 1 };
    params61.convParams.dilation = { 1, 1 };
    params61.convParams.split = 1;

    ConvolutionParams params62;
    params62.inputs.resize(1);
    params62.layerID = "params62";
    params62.inputs[0] = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 224, 224, 3, 1 } };
    params62.output = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 224, 224, 64, 1 } };
    params62.weights = { WeightsType::F16, WeightsLayout::oiyx, PADDED_VAL::ZERO, 0,{ 3, 3, 3, 64 } };
    params62.bias.resize(1);
    params62.bias[0] = { Datatype::F16, DataLayout::bf, PADDED_VAL::ZERO, 0,{ 64, 1 } };
    params62.activationFunc = ActivationFunction::RELU;
    params62.nlParams = { 1, 0 };
    params62.convParams.filterSize = { 3, 3 };
    params62.convParams.padding = { 1, 1 };
    params62.convParams.stride = { 1, 1 };
    params62.convParams.dilation = { 1, 1 };
    params62.convParams.split = 1;

    ConvolutionParams params63;
    params63.inputs.resize(1);
    params63.layerID = "params63";
    params63.inputs[0] = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 224, 224, 64, 1 } };
    params63.output = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 224, 224, 64, 1 } };
    params63.weights = { WeightsType::F16, WeightsLayout::oiyx, PADDED_VAL::ZERO, 0,{ 3, 3, 64, 64 } };
    params63.bias.resize(1);
    params63.bias[0] = { Datatype::F16, DataLayout::bf, PADDED_VAL::ZERO, 0,{ 64, 1 } };
    params63.activationFunc = ActivationFunction::RELU;
    params63.nlParams = { 1, 0 };
    params63.convParams.filterSize = { 3, 3 };
    params63.convParams.padding = { 1, 1 };
    params63.convParams.stride = { 1, 1 };
    params63.convParams.dilation = { 1, 1 };
    params63.convParams.split = 1;

    ConvolutionParams params64;
    params64.inputs.resize(1);
    params64.layerID = "params64";
    params64.inputs[0] = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 112, 112, 64, 1 } };
    params64.output = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 112, 112, 128, 1 } };
    params64.weights = { WeightsType::F16, WeightsLayout::oiyx, PADDED_VAL::ZERO, 0,{ 3, 3, 64, 128 } };
    params64.bias.resize(1);
    params64.bias[0] = { Datatype::F16, DataLayout::bf, PADDED_VAL::ZERO, 0,{ 128, 1 } };
    params64.activationFunc = ActivationFunction::RELU;
    params64.nlParams = { 1, 0 };
    params64.convParams.filterSize = { 3, 3 };
    params64.convParams.padding = { 1, 1 };
    params64.convParams.stride = { 1, 1 };
    params64.convParams.dilation = { 1, 1 };
    params64.convParams.split = 1;

    ConvolutionParams params65;
    params65.inputs.resize(1);
    params65.layerID = "params65";
    params65.inputs[0] = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 112, 112, 128, 1 } };
    params65.output = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 112, 112, 128, 1 } };
    params65.weights = { WeightsType::F16, WeightsLayout::oiyx, PADDED_VAL::ZERO, 0,{ 3, 3, 128, 128 } };
    params65.bias.resize(1);
    params65.bias[0] = { Datatype::F16, DataLayout::bf, PADDED_VAL::ZERO, 0,{ 128, 1 } };
    params65.activationFunc = ActivationFunction::RELU;
    params65.nlParams = { 1, 0 };
    params65.convParams.filterSize = { 3, 3 };
    params65.convParams.padding = { 1, 1 };
    params65.convParams.stride = { 1, 1 };
    params65.convParams.dilation = { 1, 1 };
    params65.convParams.split = 1;

    ConvolutionParams params66;
    params66.inputs.resize(1);
    params66.layerID = "params66";
    params66.inputs[0] = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 56, 56, 128, 1 } };
    params66.output = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 56, 56, 256, 1 } };
    params66.weights = { WeightsType::F16, WeightsLayout::oiyx, PADDED_VAL::ZERO, 0,{ 3, 3, 128, 256 } };
    params66.bias.resize(1);
    params66.bias[0] = { Datatype::F16, DataLayout::bf, PADDED_VAL::ZERO, 0,{ 256, 1 } };
    params66.activationFunc = ActivationFunction::RELU;
    params66.nlParams = { 1, 0 };
    params66.convParams.filterSize = { 3, 3 };
    params66.convParams.padding = { 1, 1 };
    params66.convParams.stride = { 1, 1 };
    params66.convParams.dilation = { 1, 1 };
    params66.convParams.split = 1;

    ConvolutionParams params67;
    params67.inputs.resize(1);
    params67.layerID = "params67";
    params67.inputs[0] = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 56, 56, 256, 1 } };
    params67.output = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 56, 56, 256, 1 } };
    params67.weights = { WeightsType::F16, WeightsLayout::oiyx, PADDED_VAL::ZERO, 0,{ 3, 3, 256, 256 } };
    params67.bias.resize(1);
    params67.bias[0] = { Datatype::F16, DataLayout::bf, PADDED_VAL::ZERO, 0,{ 256, 1 } };
    params67.activationFunc = ActivationFunction::RELU;
    params67.nlParams = { 1, 0 };
    params67.convParams.filterSize = { 3, 3 };
    params67.convParams.padding = { 1, 1 };
    params67.convParams.stride = { 1, 1 };
    params67.convParams.dilation = { 1, 1 };
    params67.convParams.split = 1;

    ConvolutionParams params68;
    params68.inputs.resize(1);
    params68.layerID = "params68";
    params68.inputs[0] = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 56, 56, 256, 1 } };
    params68.output = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 56, 56, 256, 1 } };
    params68.weights = { WeightsType::F16, WeightsLayout::oiyx, PADDED_VAL::ZERO, 0,{ 3, 3, 256, 256 } };
    params68.bias.resize(1);
    params68.bias[0] = { Datatype::F16, DataLayout::bf, PADDED_VAL::ZERO, 0,{ 256, 1 } };
    params68.activationFunc = ActivationFunction::RELU;
    params68.nlParams = { 1, 0 };
    params68.convParams.filterSize = { 3, 3 };
    params68.convParams.padding = { 1, 1 };
    params68.convParams.stride = { 1, 1 };
    params68.convParams.dilation = { 1, 1 };
    params68.convParams.split = 1;

    ConvolutionParams params69;
    params69.inputs.resize(1);
    params69.layerID = "params69";
    params69.inputs[0] = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 28, 28, 256, 1 } };
    params69.output = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 28, 28, 512, 1 } };
    params69.weights = { WeightsType::F16, WeightsLayout::oiyx, PADDED_VAL::ZERO, 0,{ 3, 3, 256, 512 } };
    params69.bias.resize(1);
    params69.bias[0] = { Datatype::F16, DataLayout::bf, PADDED_VAL::ZERO, 0,{ 512, 1 } };
    params69.activationFunc = ActivationFunction::RELU;
    params69.nlParams = { 1, 0 };
    params69.convParams.filterSize = { 3, 3 };
    params69.convParams.padding = { 1, 1 };
    params69.convParams.stride = { 1, 1 };
    params69.convParams.dilation = { 1, 1 };
    params69.convParams.split = 1;

    ConvolutionParams params70;
    params70.inputs.resize(1);
    params70.layerID = "params70";
    params70.inputs[0] = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 28, 28, 512, 1 } };
    params70.output = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 28, 28, 512, 1 } };
    params70.weights = { WeightsType::F16, WeightsLayout::oiyx, PADDED_VAL::ZERO, 0,{ 3, 3, 512, 512 } };
    params70.bias.resize(1);
    params70.bias[0] = { Datatype::F16, DataLayout::bf, PADDED_VAL::ZERO, 0,{ 512, 1 } };
    params70.activationFunc = ActivationFunction::RELU;
    params70.nlParams = { 1, 0 };
    params70.convParams.filterSize = { 3, 3 };
    params70.convParams.padding = { 1, 1 };
    params70.convParams.stride = { 1, 1 };
    params70.convParams.dilation = { 1, 1 };
    params70.convParams.split = 1;

    ConvolutionParams params71;
    params71.inputs.resize(1);
    params71.layerID = "params71";
    params71.inputs[0] = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 28, 28, 512, 1 } };
    params71.output = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 28, 28, 512, 1 } };
    params71.weights = { WeightsType::F16, WeightsLayout::oiyx, PADDED_VAL::ZERO, 0,{ 3, 3, 512, 512 } };
    params71.bias.resize(1);
    params71.bias[0] = { Datatype::F16, DataLayout::bf, PADDED_VAL::ZERO, 0,{ 512, 1 } };
    params71.activationFunc = ActivationFunction::RELU;
    params71.nlParams = { 1, 0 };
    params71.convParams.filterSize = { 3, 3 };
    params71.convParams.padding = { 1, 1 };
    params71.convParams.stride = { 1, 1 };
    params71.convParams.dilation = { 1, 1 };
    params71.convParams.split = 1;

    ConvolutionParams params72;
    params72.inputs.resize(1);
    params72.layerID = "params72";
    params72.inputs[0] = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 14, 14, 512, 1 } };
    params72.output = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 14, 14, 512, 1 } };
    params72.weights = { WeightsType::F16, WeightsLayout::oiyx, PADDED_VAL::ZERO, 0,{ 3, 3, 512, 512 } };
    params72.bias.resize(1);
    params72.bias[0] = { Datatype::F16, DataLayout::bf, PADDED_VAL::ZERO, 0,{ 512, 1 } };
    params72.activationFunc = ActivationFunction::RELU;
    params72.nlParams = { 1, 0 };
    params72.convParams.filterSize = { 3, 3 };
    params72.convParams.padding = { 1, 1 };
    params72.convParams.stride = { 1, 1 };
    params72.convParams.dilation = { 1, 1 };
    params72.convParams.split = 1;

    ConvolutionParams params73;
    params73.inputs.resize(1);
    params73.layerID = "params73";
    params73.inputs[0] = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 14, 14, 512, 1 } };
    params73.output = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 14, 14, 512, 1 } };
    params73.weights = { WeightsType::F16, WeightsLayout::oiyx, PADDED_VAL::ZERO, 0,{ 3, 3, 512, 512 } };
    params73.bias.resize(1);
    params73.bias[0] = { Datatype::F16, DataLayout::bf, PADDED_VAL::ZERO, 0,{ 512, 1 } };
    params73.activationFunc = ActivationFunction::RELU;
    params73.nlParams = { 1, 0 };
    params73.convParams.filterSize = { 3, 3 };
    params73.convParams.padding = { 1, 1 };
    params73.convParams.stride = { 1, 1 };
    params73.convParams.dilation = { 1, 1 };
    params73.convParams.split = 1;

    ConvolutionParams params74;
    params74.inputs.resize(1);
    params74.layerID = "params74";
    params74.inputs[0] = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 14, 14, 512, 1 } };
    params74.output = { Datatype::F16, DataLayout::bfyx, PADDED_VAL::ZERO, 0,{ 14, 14, 512, 1 } };
    params74.weights = { WeightsType::F16, WeightsLayout::oiyx, PADDED_VAL::ZERO, 0,{ 3, 3, 512, 512 } };
    params74.bias.resize(1);
    params74.bias[0] = { Datatype::F16, DataLayout::bf, PADDED_VAL::ZERO, 0,{ 512, 1 } };
    params74.activationFunc = ActivationFunction::RELU;
    params74.nlParams = { 1, 0 };
    params74.convParams.filterSize = { 3, 3 };
    params74.convParams.padding = { 1, 1 };
    params74.convParams.stride = { 1, 1 };
    params74.convParams.dilation = { 1, 1 };
    params74.convParams.split = 1;

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
    };
}
