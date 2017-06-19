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

#include "generic_eltwise_kernel_ref.h"
 
namespace KernelSelector {

    ParamsKey GenericEltwiseKernelRef::GetSupportedKey() const
    {
        ParamsKey k;
        k.SetInputDataType(Datatype::F16);
        k.SetInputDataType(Datatype::F32);
        k.SetOutputDataType(Datatype::F16);
        k.SetOutputDataType(Datatype::F32);
        k.EnableAllInputLayout();
        k.EnableAllOutputLayout();
        k.SetOffsetSupport();
        k.SetPitchesSupport();
        k.SetBatchingSupport();
        return k;
    }

    uint32_t get_number_of_inputs(EltwiseMode m)
    {
        switch (m)
        {
        case KernelSelector::EltwiseMode::ADD:
        case KernelSelector::EltwiseMode::SUB:
        case KernelSelector::EltwiseMode::MUL:
        case KernelSelector::EltwiseMode::DIV:
        case KernelSelector::EltwiseMode::MIN:
        case KernelSelector::EltwiseMode::MAX:
        case KernelSelector::EltwiseMode::POW:
        case KernelSelector::EltwiseMode::MODULU:
            return 2;
        case KernelSelector::EltwiseMode::SQRT:
        case KernelSelector::EltwiseMode::ASSIGN:
            return 1;
        default:
            return 0;
        }
    }

    jit_constants GenericEltwiseKernelRef::get_jit_constants(const EltwiseParams& params) const
    {
        auto jit = get_common_jit_constants(params);
        
        std::string inputs_decls;
        for (size_t i = 0; i < params.inputs.size(); i++)
        {
            inputs_decls += "const __global UNIT_TYPE* input" + std::to_string(i) + ",";
        }

        jit.add_constant(gpu::make_jit_constant("INPUTS_DECLS", inputs_decls));

        std::string do_eltwise;

        if (params.eltwiseParams.size() == 0)
        {
            throw std::runtime_error("eltwise without operations");
        }

        for (size_t i = 0; i < params.eltwiseParams.size(); i++)
        {
            const std::string i_str = std::to_string(i);
            const auto& ew = params.eltwiseParams[i];
            if (ew.inputs.size() != get_number_of_inputs(ew.mode))
            {
                throw std::runtime_error("error number of inputs to elwise params");
            }
            for (size_t j = 0; j < ew.inputs.size(); j++)
            {
                const auto& input = ew.inputs[i];
                const std::string name = "INPUT_" + i_str + "_" + std::to_string(j);
                switch (input.mode)
                {
                case EltwiseInputMode::SCALAR:
                    jit.add_constant(gpu::make_jit_constant(name, input.scalar));
                    break;
                case EltwiseInputMode::INPUT_BUFFER:
                    if (input.index >= params.inputs.size())
                    {
                        throw std::runtime_error("input index is greater than the provided inputs");
                    }
                    jit.add_constant(gpu::make_jit_constant(name, "input" + std::to_string(j) + "[GET_INDEX(INPUT, " + std::to_string(input.index) +")]"));
                    break;
                case EltwiseInputMode::TEMP_RESULTS_INDEX:
                    jit.add_constant(gpu::make_jit_constant(name, "tmp" + std::to_string(input.index)));
                    break;
                default:
                    break;
                }
            }

            std::string input0_str = "INPUT_" + i_str + "_0";
            std::string input1_str = "INPUT_" + i_str + "_1";

            std::string op = "UNIT_TYPE tmp" + i_str + " = ";
            switch (ew.mode)
            {
            case KernelSelector::EltwiseMode::ADD:      op += input0_str + " + " + input1_str; break;
            case KernelSelector::EltwiseMode::SUB:      op += input0_str + " - " + input1_str; break;
            case KernelSelector::EltwiseMode::MUL:      op += input0_str + " * " + input1_str; break;
            case KernelSelector::EltwiseMode::DIV:      op += input0_str + " / " + input1_str; break;
            case KernelSelector::EltwiseMode::MODULU:   op += input0_str + " % " + input1_str; break;
            case KernelSelector::EltwiseMode::MIN:      op += "fmin(" + input0_str + ", " + input1_str + ")"; break;
            case KernelSelector::EltwiseMode::MAX:      op += "fmax(" + input0_str + ", " + input1_str + ")"; break;
            case KernelSelector::EltwiseMode::POW:      op += "pow("  + input0_str + ", " + input1_str + ")"; break;
            case KernelSelector::EltwiseMode::SQRT:     op += "sqrt(" + input0_str + ")"; break;
            case KernelSelector::EltwiseMode::ASSIGN:   op += input0_str; break;
            default:
                break;;
            }

            std::string opname = "OPERATION" + i_str;
            jit.add_constant(gpu::make_jit_constant(opname, op));
            do_eltwise += opname + ";";
        }

        do_eltwise += "res = tmp" + std::to_string(params.eltwiseParams.size() - 1) + ";";

        jit.add_constant(gpu::make_jit_constant("DO_ELTWISE", do_eltwise));

        jit.add_constants({
            gpu::make_jit_constant("ACTIVATION_FUNCTION_" + toString(params.activationFunc), ""),
            gpu::make_jit_constant("TYPE_" + toString(params.output.dtype), ""),
            gpu::make_jit_constant("NL_M", params.nlParams.m),
            gpu::make_jit_constant("NL_N", params.nlParams.n),
        });

        return jit;
    }

    KernelsData GenericEltwiseKernelRef::GetKernelsData(const Params& params, const OptionalParams&) const
    {
        assert(params.GetType() == KernelType::ELTWISE);

        KernelData kd = KernelData::Default<EltwiseParams>(params, 1);
        EltwiseParams& newParams = *static_cast<EltwiseParams*>(kd.params.get());

        std::string jit;

        auto entry_point = get_entry_point(kernel_name, newParams.layerID);

        try
        {
            auto cldnn_jit = get_jit_constants(newParams);
            jit = create_jit_from_template(kernel_name, cldnn_jit.get_definitions(), entry_point, false);
        }
        catch (const std::runtime_error&)
        {
            return KernelsData();
        }

        const auto& out = newParams.output;
        auto& kernel = kd.kernels[0];
        std::vector<size_t> gws;
        for (const auto& o : out.dims)
        {
            gws.push_back(o.v);
        }

        for (size_t i = gws.size(); i < 4; i++)
        {
            gws.push_back(1U);
        }

        kernel.work_groups.global = cl::NDRange(gws[0], gws[1], gws[2] * gws[3]);
        kernel.kernel_string = get_kernel_string(kernel_name, jit, entry_point, ROUND_ROBIN);
        kernel.args_desc = get_args_desc(1, false, false);

        kd.estimated_time = DONT_USE_IF_HAVE_SOMETHING_ELSE;

        return{ kd };
    }
}