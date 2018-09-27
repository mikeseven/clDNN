/*
// Copyright (c) 2018 Intel Corporation
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

#include "jitter.h"

namespace kernel_selector {

    std::string toCLType(WeightsType wType)
    {
        switch (wType)
        {
        case WeightsType::INT8: return GetTypeName<int8_t>();
        case WeightsType::F16:  return "half";
        case WeightsType::F32:  return GetTypeName<float>();
        default: return "";
        }
    }

    std::string toCLType(Datatype dType)
    {
        switch (dType)
        {
        case Datatype::INT8:    return GetTypeName<int8_t>();
        case Datatype::UINT8:   return GetTypeName<uint8_t>();
        case Datatype::INT16:   return GetTypeName<int16_t>();
        case Datatype::UINT16:  return GetTypeName<uint16_t>();
        case Datatype::INT32:   return GetTypeName<int32_t>();
        case Datatype::UINT32:  return GetTypeName<uint32_t>();
        case Datatype::INT64:   return GetTypeName<int64_t>();
        case Datatype::F16:     return "half";
        case Datatype::F32:     return GetTypeName<float>();
        default: return "";
        }
    }

    std::string getMeanOpString(MeanOp op)
    {
        switch (op)
        {
        case MeanOp::NONE:   return "val";
        case MeanOp::DIV:    return "val/mean_val";
        case MeanOp::MUL:    return "val*mean_val";
        case MeanOp::SUB:    return "val-mean_val";
        default: return "";
        }
    }

    JitDefinitions JitConstants::GetDefinitions() const
    {
        JitDefinitions definitons;
        definitons.reserve(_constants.size() * 6); //assuming max 6 pairs per jit_constant

        for (auto& constant : _constants) {
            auto def = constant->GetDefinitions();
            definitons.insert(definitons.end(), def.begin(), def.end());
        }
        return definitons;
    }

    JitDefinitions DataTensorJitConstant::GetDefinitions() const
    {
        JitDefinitions baseDefinitions = TensorBaseTJitConstant::GetDefinitions(_tensor);

        JitDefinitions definitions{
            { _name + "_SIZE_X",        toCodeString(_tensor.X().v) },
        { _name + "_SIZE_Y",        toCodeString(_tensor.Y().v) },
        { _name + "_FEATURE_NUM",   toCodeString(_tensor.Feature().v) },
        { _name + "_ROI_NUM",       toCodeString(_tensor.ROI().v) },
        { _name + "_BATCH_NUM",     toCodeString(_tensor.Batch().v) },
        { _name + "_X_PITCH",       toCodeString(_tensor.X().pitch) },
        { _name + "_Y_PITCH",       toCodeString(_tensor.Y().pitch) },
        { _name + "_FEATURE_PITCH", toCodeString(_tensor.Feature().pitch) },
        { _name + "_ROI_PITCH",     toCodeString(_tensor.ROI().pitch) },
        { _name + "_BATCH_PITCH",   toCodeString(_tensor.Batch().pitch) },
        { _name + "_PAD_BEFORE_SIZE_X",        toCodeString(_tensor.X().pad.before) },
        { _name + "_PAD_BEFORE_SIZE_Y",        toCodeString(_tensor.Y().pad.before) },
        { _name + "_PAD_BEFORE_FEATURE_NUM",   toCodeString(_tensor.Feature().pad.before) },
        { _name + "_PAD_BEFORE_BATCH_NUM",     toCodeString(_tensor.Batch().pad.before) },
        { _name + "_PAD_AFTER_SIZE_X",         toCodeString(_tensor.X().pad.after) },
        { _name + "_PAD_AFTER_SIZE_Y",         toCodeString(_tensor.Y().pad.after) },
        { _name + "_PAD_AFTER_FEATURE_NUM",    toCodeString(_tensor.Feature().pad.after) },
        { _name + "_PAD_AFTER_BATCH_NUM",      toCodeString(_tensor.Batch().pad.after) },
        };

        definitions.insert(definitions.end(), baseDefinitions.begin(), baseDefinitions.end());

        return definitions;
    }

    JitDefinitions WeightTensorJitConstant::GetDefinitions() const
    {
        JitDefinitions baseDefinitions = TensorBaseTJitConstant::GetDefinitions(_tensor);

        JitDefinitions definitions{
            { _name + "_SIZE_X",        toCodeString(_tensor.X().v) },
        { _name + "_SIZE_Y",        toCodeString(_tensor.Y().v) },
        { _name + "_IFM_NUM",       toCodeString(_tensor.IFM().v) },
        { _name + "_OFM_NUM",       toCodeString(_tensor.OFM().v) },
        { _name + "_X_PITCH",       toCodeString(_tensor.X().pitch) },
        { _name + "_Y_PITCH",       toCodeString(_tensor.Y().pitch) },
        { _name + "_IFM_PITCH",     toCodeString(_tensor.IFM().pitch) },
        { _name + "_OFM_PITCH",     toCodeString(_tensor.OFM().pitch) },
        };

        definitions.insert(definitions.end(), baseDefinitions.begin(), baseDefinitions.end());

        return definitions;
    }

    JitConstants MakeUnitTypeJitConstants(Datatype dataType)
    {
        std::string unit_type;
        std::string unit_max_val;
        std::string unit_min_val;
        std::string unit_val_one;
        std::string unit_val_zero;
        std::string to_unit_type;
        std::string unit_max_func;
        std::string unit_min_func;
        switch (dataType)
        {
        case Datatype::INT8:
            unit_type = "char";
            unit_max_val = "CHAR_MAX";
            unit_min_val = "-UNIT_VAL_MAX";
            unit_val_one = "(char) 1";
            unit_val_zero = "(char) 0";
            to_unit_type = "convert_char(v)";
            unit_max_func = "max";
            unit_min_func = "min";
            break;
        case Datatype::INT32:
            unit_type = "int";
            unit_max_val = "INT_MAX";
            unit_min_val = "-UNIT_VAL_MAX";
            unit_val_one = "(int) 1";
            unit_val_zero = "(int) 0";
            to_unit_type = "convert_int(v)";
            unit_max_func = "max";
            unit_min_func = "min";
            break;
        case Datatype::INT64:
            unit_type = "long";
            unit_max_val = "LONG_MAX";
            unit_min_val = "-UNIT_VAL_MAX";
            unit_val_one = "(long) 1";
            unit_val_zero = "(long) 0";
            to_unit_type = "convert_long(v)";
            unit_max_func = "max";
            unit_min_func = "min";
            break;
        case Datatype::F16:
            unit_type = "half";
            unit_max_val = "HALF_MAX";
            unit_min_val = "-UNIT_VAL_MAX";
            unit_val_one = "1.0h";
            unit_val_zero = "0.0h";
            to_unit_type = "convert_half(v)";
            unit_max_func = "fmax";
            unit_min_func = "fmin";
            break;
        default:
            unit_type = "float";
            unit_max_val = "FLT_MAX";
            unit_min_val = "-UNIT_VAL_MAX";
            unit_val_one = "1.0f";
            unit_val_zero = "0.0f";
            to_unit_type = "(float)(v)";
            unit_max_func = "fmax";
            unit_min_func = "fmin";
            break;
        }

        return JitConstants
        {
            MakeJitConstant("UNIT_TYPE",            unit_type),
            MakeJitConstant("UNIT_VAL_MAX",         unit_max_val),
            MakeJitConstant("UNIT_VAL_MIN",         unit_min_val),
            MakeJitConstant("UNIT_VAL_ONE",         unit_val_one),
            MakeJitConstant("UNIT_VAL_ZERO",        unit_val_zero),
            MakeJitConstant("TO_UNIT_TYPE(v)",      to_unit_type),
            MakeJitConstant("UNIT_MAX_FUNC",        unit_max_func),
            MakeJitConstant("UNIT_MIN_FUNC",        unit_min_func),
        };
    }
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // MakeBaseParamsJitConstants
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    JitConstants MakeBaseParamsJitConstants(const base_params& params)
    {
        bool bFP16Used = params.output.GetDType() == Datatype::F16;
        bool bInt8Used = params.output.GetDType() == Datatype::INT8;
        bool bInt32Used = params.output.GetDType() == Datatype::INT32;
        bool bInt64Used = params.output.GetDType() == Datatype::INT64;
        for (const auto& i : params.inputs)
        {
            bFP16Used |= i.GetDType() == Datatype::F16;
            bInt8Used |= i.GetDType() == Datatype::INT8;
            bInt32Used |= i.GetDType() == Datatype::INT32;
            bInt64Used |= i.GetDType() == Datatype::INT64;
        }

        JitConstants jit{
            MakeJitConstant("OUTPUT",               params.output),
            MakeJitConstant("FP64_SUPPORTED",       params.engineInfo.bFP64Support),
            MakeJitConstant("FP16_SUPPORTED",       params.engineInfo.bFP16Support),
            MakeJitConstant("FP16_UNIT_USED",       bFP16Used),
            MakeJitConstant("INT8_UNIT_USED",       bInt8Used),
            MakeJitConstant("INT32_UNIT_USED",      bInt32Used),
            MakeJitConstant("INT64_UNIT_USED",      bInt64Used),
            MakeJitConstant("NL_M",                 params.activationParams.m),
            MakeJitConstant("NL_N",                 params.activationParams.n),
            MakeJitConstant("ACTIVATION_FUNCTION_" + toString(params.activationFunc), ""),
            MakeJitConstant("GRADIENT",             params.gradient),
        };

        if (bInt8Used)
        {
            jit.Merge(MakeUnitTypeJitConstants(Datatype::INT8));
        }
        else if (bFP16Used)
        {
            jit.Merge(MakeUnitTypeJitConstants(Datatype::F16));
        }
        else if (bInt32Used)
        {
            jit.Merge(MakeUnitTypeJitConstants(Datatype::INT32));
        }
        else if (bInt64Used)
        {
            jit.Merge(MakeUnitTypeJitConstants(Datatype::INT64));
        }
        else
        {
            jit.Merge(MakeUnitTypeJitConstants(Datatype::F32));
        }

        for (size_t i = 0; i < params.inputs.size(); i++)
        {
            jit.AddConstant(MakeJitConstant("INPUT" + toCodeString(i), params.inputs[i]));
        }

        return jit;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // MakeLoopUnrollParamsJitConstants
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    JitConstants MakeLoopUnrollParamsJitConstants(uint32_t loopCount)
    {
        JitConstants jit{
            MakeJitConstant("LOOP0(VAR, STMT)", ""),
            MakeJitConstant("LOOP1(VAR, STMT)", "(STMT); (VAR)++;"),
        };

        for (uint32_t i = 2; i <= loopCount + 1; i++)
        {
            jit.AddConstant({
                MakeJitConstant("LOOP" + toCodeString(i) + "(VAR, STMT)", "LOOP" + toCodeString(i - 1) + "(VAR, STMT); (STMT); (VAR)++;"),
                });
        }

        jit.AddConstant({
            MakeJitConstant("LOOP(N, VAR, STMT)", "CAT(LOOP, N)((VAR), (STMT))"),
            });

        return jit;
    }

}