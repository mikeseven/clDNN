﻿/*
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

#include "convolution_kernel_tutorial1.h"
#include "kernel_selector_utils.h"

namespace KernelSelector {
    
    ParamsKey ConvolutionKernel_Tutorial1::GetSupportedKey() const
    {
        // Step 1:
        // - Update the features supported by the kernel below

        ParamsKey k;
        
        // Supported data type
        k.EnableInputDataType(Datatype::F16);
        k.EnableInputDataType(Datatype::F32);
        k.EnableOutputDataType(Datatype::F16);
        k.EnableOutputDataType(Datatype::F32);
        k.EnableInputWeightsType(WeightsType::F16);
        k.EnableInputWeightsType(WeightsType::F32);

        // Supported layout
        k.EnableAllInputLayout();
        k.EnableAllOutputLayout();

        // Supported tensor offset/pitch/padding
        k.EnableTensorOffset();
        k.EnableTensorPitches();
        k.EnableBatching();

        // Supported convolution extra data
        k.EnableDilation();
        k.EnableBiasPerFeature();
        k.EnableBiasPerOutput();
        k.EnableNonBiasTerm();

        // Supported convolution which get a split index and uses it as a view on the input/output (Alexnet only)
        k.EnableSplitSupport();

        return k;
    }

    KernelsData ConvolutionKernel_Tutorial1::GetKernelsData(const Params& /*params*/, const OptionalParams& /*options*/) const
    {
        return{};

        // Step 2:
        // - Uncomment and update the following lines

        // assert(params.GetType() == KernelType::CONVOLUTION && options.GetType() == KernelType::CONVOLUTION);
        // KernelData kd = KernelData::Default<ConvolutionParams>(params);
        // ConvolutionParams& newParams = *static_cast<ConvolutionParams*>(kd.params.get());
        // const ConvolutionOptionalParams& optParams = static_cast<const ConvolutionOptionalParams&>(options);
        // auto& kernel = kd.kernels[0];
        

        // Step 3:
        // - make sure that the input weights tensor fit to this kernel needs. 
        //   in case it's not and the flag "optParams.allowWeightsReorder" set to "true", please update
        //   the member "kd.weightsReorderParams" with the right OpenCL/CPU kernel which will be used to reorder the 
        //   weights in the loading time.
        //   you have three options:
        //   - provide a cpu code - inherit from "CPUKernel" and implement "Execute" function.
        //      (by default the input layout of CPU kernel is simple bfyx, and clDNN will reorder it for you before calling to Execute function)
        //   - provide a GPU code by filling clKernelData.
        //   - use existing layouts which clDNN support and use the auxiliary function "UpdateWeightsParams"


        // Step 4:
        // - make sure that the input tensor fits to this kernel's needs. 
        //   currently Convolution in clDNN doesn't allow the kernel to ask reordering


        // Step 5:
        // - fill "kernel.kernelString"
        //   - fill "kernel.kernelString->str"                  - the source of the kernel. 
        //     please use "db.get(kernelName)" in case you use "*.cl" file which located under "kernel_selector\core\cl_kernels\".
        //   - fill "kernel.kernelString->jit"                  - Dynamic jit of this params. 
        //   - fill "kernel.kernelString->options"              - options which pass to cl program build functions (like "-cl-no-subgroup-ifp")
        //   - fill "kernel.kernelString->entry_point"          - kernel entry point 
        //   - fill "kernel.kernelString->batch_compilation"    - A flag that allow clDNN kernel to compile this kernel as a part of a program
        //                                                        NOTE: this can only be used if you prevent symbol conflicts with other kernels (#undef is done automatically by clDNN)


        // Step 6:
        // - fill "kernel.WorkGroupSizes" - local/global work group sizes for OpenCL kernel


        // Step 7:
        // - fill "kernel.arguments" - which describe the argument of the kernel. 
        //   in this tutorial you can use:
        //     kernel.arguments.push_back({ ArgumentDescriptor::Types::INPUT, 0 }); // "0" mean index of the input in case of multiple inputs.
        //     kernel.arguments.push_back({ ArgumentDescriptor::Types::OUTPUT, 0 });
        //     kernel.arguments.push_back({ ArgumentDescriptor::Types::WEIGHTS, 0 });
        //     kernel.arguments.push_back({ ArgumentDescriptor::Types::BIAS, 0 });


        // Step 8:
        // - estimate the kernel's execution time. currently it's under development so please use FORCE_PRIORITY_<X> - lower is better.


        // return{ kd };
    }
}