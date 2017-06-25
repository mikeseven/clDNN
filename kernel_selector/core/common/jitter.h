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

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "tensor_type.h"
#include "kernel_selector_params.h"
#include <algorithm>
#include <iostream>
#include <sstream>
#include <cmath>

namespace KernelSelector {

using JitDefinitions = std::vector<std::pair<std::string, std::string>>;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Helpers
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T>
inline std::string get_type_name() { throw std::runtime_error("Implement me"); }
template <>
inline std::string get_type_name<double>() { return "double"; }
template <>
inline std::string get_type_name<float>() { return "float"; }
template <>
inline std::string get_type_name<int>() { return "int"; }
template <>
inline std::string get_type_name<unsigned>() { return "unsigned"; }
template <>
inline std::string get_type_name<char>() { return "char"; }
template <>
inline std::string get_type_name<short>() { return "short"; }
template <>
inline std::string get_type_name<uint16_t>() { return "unsigned short"; }

inline std::string ToCLType(WeightsType wType)
{
    switch (wType)
    {
    case WeightsType::F16: return "half";
    case WeightsType::F32: return "float";
    case WeightsType::INT8: return "char";
    default: return "";
    }
}

inline std::string ToCLType(Datatype wType)
{
    switch (wType)
    {
    case Datatype::F16: return "half";
    case Datatype::F32: return "float";
    default: return "";
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// ToCodeString functions
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// TODO improve to_code_string specializations
template<typename T>
std::string toCodeString(T val) { return std::to_string(val); }

template<>
inline std::string toCodeString<std::string>(std::string val) { return val; }

template<>
inline std::string toCodeString<const char*>(const char* val) { return val; }

template<>
inline std::string toCodeString<char*>(char* val) { return val; }

template<>
inline std::string toCodeString<float>(float val) {
    // 64 chars should be enought to store: "-0x0.123456p-123f /*-0.123456e-123*/"
    char buffer[64] = "";
    if (std::isinf(val))
        std::snprintf(buffer, sizeof(buffer), "%sINFINITY", std::signbit(val) ? "-" : "");
    else
        std::snprintf(buffer, sizeof(buffer), "%.6af /*%.4g*/", double(val), double(val));
    return buffer;
}

template<>
inline std::string toCodeString<double>(double val) {
    // 64 chars should be enought to store: "-0x0.1234567890123p-1234 /*-0.1234567890123e-1074*/"
    char buffer[64] = "";
    if (std::isinf(val))
        std::snprintf(buffer, sizeof(buffer), "%sINFINITY", std::signbit(val) ? "-" : "");
    else
        std::snprintf(buffer, sizeof(buffer), "%.13a /*%.4g*/", val, val);
    return buffer;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// JitConstant
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename VecT, typename Func>
inline std::string toVectorString(const VecT& vec, const std::string& vertorType, size_t maxDim, int padFillingVal, Func fetchVal)
{
    std::stringstream ss;
    ss << "(" << vertorType << " []){ ";
    for (size_t i = 0; i < vec.size(); i++)
        ss << toCodeString(fetchVal(vec[i])) << ",";
    for (size_t i = vec.size(); i < maxDim; i++)
        ss << padFillingVal << ",";
    ss << " } ";
    return ss.str();
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// JitConstant
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class JitConstant {
protected:
    const std::string _name;
    JitConstant(const std::string& name):_name(name){}

public:
    virtual JitDefinitions GetDefinitions() const = 0;
    virtual ~JitConstant() {}
};

class simple_jit_constant : public JitConstant {
    const std::string _value;

public:
    simple_jit_constant(const std::string& name, const std::string& value)
        :JitConstant(name), _value(value) {}

    JitDefinitions GetDefinitions() const override {
        return JitDefinitions{ {_name, _value} };
    }
};

template<typename T>
std::shared_ptr<JitConstant> MakeJitConstant(const std::string& name, T value) {
    return std::static_pointer_cast<JitConstant>(std::make_shared<simple_jit_constant>(name, toCodeString(value)));
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// DataTensorJitConstant
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class DataTensorJitConstant : public JitConstant 
{
    const DataTensor _tensor;

public:
    DataTensorJitConstant(const std::string& name, const DataTensor& t) : JitConstant(name), _tensor(t) {}

    JitDefinitions GetDefinitions() const override 
    {
        JitDefinitions definitions{
            { _name + "_TYPE",          ToCLType(_tensor.GetDType()) },
            { _name + "_OFFSET",        std::to_string(_tensor.GetOffset()) },
            { _name + "_LIMIT",         std::to_string(_tensor.LengthWithPadding()) },
            { _name + "_LENGTH",        std::to_string(_tensor.Length()) },
            { _name + "_DIMS",          std::to_string(_tensor.GetDims().size()) },
            { _name + "_SIZE_X",        std::to_string(_tensor.X().v) },
            { _name + "_SIZE_Y",        std::to_string(_tensor.Y().v) },
            { _name + "_FEATURE_NUM",   std::to_string(_tensor.Feature().v) },
            { _name + "_ROI_NUM",       std::to_string(_tensor.ROI().v) },
            { _name + "_BATCH_NUM",     std::to_string(_tensor.Batch().v) },
            { _name + "_X_PITCH",       std::to_string(_tensor.X().pitch) },
            { _name + "_Y_PITCH",       std::to_string(_tensor.Y().pitch) },
            { _name + "_FEATURE_PITCH", std::to_string(_tensor.Feature().pitch) },
            { _name + "_ROI_PITCH",     std::to_string(_tensor.ROI().pitch) },
            { _name + "_BATCH_PITCH",   std::to_string(_tensor.Batch().pitch) },
            { _name + "_SIMPLE",        std::to_string(_tensor.SimpleLayout()) },
            { "TO_" + _name + "_TYPE",  "convert_" + ToCLType(_tensor.GetDType()) },
            { _name + "_LAYOUT_" + toString(_tensor.GetLayout()), "1" },
        };

        definitions.push_back({ _name + "_SIZE", std::to_string(_tensor.GetDims().size()) });
        definitions.push_back({ _name + "_SIZES", toVectorString(_tensor.GetDims(), "size_t", CLDNN_TENSOR_DIM_MAX, 1, [](const Tensor::Dim& d) { return d.v; }) });
        definitions.push_back({ _name + "_PITCHES", toVectorString(_tensor.GetDims(), "size_t", CLDNN_TENSOR_DIM_MAX, 1, [](const Tensor::Dim& d) { return d.pitch; }) });

        return definitions;
    }
};

inline std::shared_ptr<JitConstant> MakeJitConstant(const std::string& name, const DataTensor& value) {
    return std::static_pointer_cast<JitConstant>(std::make_shared<DataTensorJitConstant>(name, value));
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// WeightTensorJitConstant
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class WeightTensorJitConstant : public JitConstant 
{
    const WeightsTensor _tensor;

public:
    WeightTensorJitConstant(const std::string& name, const WeightsTensor& t) : JitConstant(name), _tensor(t) {}

    JitDefinitions GetDefinitions() const override 
    {
        JitDefinitions definitions{
            { _name + "_TYPE",          ToCLType(_tensor.GetDType()) },
            { _name + "_OFFSET",        std::to_string(_tensor.GetOffset()) },
            { _name + "_LIMIT",         std::to_string(_tensor.LengthWithPadding()) },
            { _name + "_DIMS",          std::to_string(_tensor.GetDims().size()) },
            { _name + "_SIZE_X",        std::to_string(_tensor.X().v) },
            { _name + "_SIZE_Y",        std::to_string(_tensor.Y().v) },
            { _name + "_IFM_NUM",       std::to_string(_tensor.IFM().v) },
            { _name + "_OFM_NUM",       std::to_string(_tensor.OFM().v) },
            { _name + "_X_PITCH",       std::to_string(_tensor.X().pitch) },
            { _name + "_Y_PITCH",       std::to_string(_tensor.Y().pitch) },
            { _name + "_IFM_PITCH",     std::to_string(_tensor.IFM().pitch) },
            { _name + "_OFM_PITCH",     std::to_string(_tensor.OFM().pitch) },
            { _name + "_SIMPLE",        std::to_string(_tensor.SimpleLayout()) },
            { "TO_" + _name + "_TYPE",  "convert_" + ToCLType(_tensor.GetDType()) },
            { _name + "_LAYOUT_" + toString(_tensor.GetLayout()), "1" },
        };

        // TODO: refactor it
        
        definitions.push_back({ _name + "_SIZE", std::to_string(_tensor.GetDims().size()) });
        definitions.push_back({ _name + "_SIZES", toVectorString(_tensor.GetDims(), "size_t", CLDNN_TENSOR_DIM_MAX, 1, [](const Tensor::Dim& d) { return d.v; }) });
        definitions.push_back({ _name + "_PITCHES", toVectorString(_tensor.GetDims(), "size_t", CLDNN_TENSOR_DIM_MAX, 1, [](const Tensor::Dim& d) { return d.pitch; }) });

        return definitions;
    }
};

inline std::shared_ptr<JitConstant> MakeJitConstant(const std::string& name, const WeightsTensor& value) {
    return std::static_pointer_cast<JitConstant>(std::make_shared<WeightTensorJitConstant>(name, value));
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// VectorDataJitConstant
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T>
class VectorDataJitConstant : public JitConstant 
{
    const std::vector<T> _data;

public:
    VectorDataJitConstant(const std::string& name, const std::vector<T>& data) : JitConstant(name), _data(data) {}

    JitDefinitions GetDefinitions() const override 
    {
        JitDefinitions result{
            { _name + "_SIZE", std::to_string(_data.size()) },
            { _name, toVectorString(_data, get_type_name<T>(), _data.size(), 1, [](const T& v) {return v; } ) },
        };
        return result;
    }
};

template <typename T>
inline  std::shared_ptr<JitConstant> MakeJitConstant(const std::string& name, const std::vector<T>& value) {
    return std::static_pointer_cast<JitConstant>(std::make_shared<VectorDataJitConstant<T>>(name, value));
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Size
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T>
class SizeJitConstant : public JitConstant
{
    const Size<T> _size;

public:
    SizeJitConstant(const std::string& name, const Size<T>& size) : JitConstant(name), _size(size) {}

    JitDefinitions GetDefinitions() const override
    {
        JitDefinitions definitions{
            { _name + "_SIZE_X",        std::to_string(_size.x) },
            { _name + "_SIZE_Y",        std::to_string(_size.y) },
        };
        return definitions;
    }
};

template <typename T>
inline std::shared_ptr<JitConstant> MakeJitConstant(const std::string& name, const Size<T>& value) {
    return std::static_pointer_cast<JitConstant>(std::make_shared<SizeJitConstant<T>>(name, value));
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// jit_constants
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class JitConstants 
{
    std::vector<std::shared_ptr<JitConstant>> _constants;
public:
    JitConstants(std::initializer_list<std::shared_ptr<JitConstant>> constants) :_constants(constants) {}

    void AddConstant(std::shared_ptr<JitConstant> constant)
    {
        _constants.push_back(constant);
    }

    void AddConstants(const std::vector<std::shared_ptr<JitConstant>>& constants)
    {
        for (const auto& c : constants)
        {
            _constants.push_back(c);
        }
    }

    void Merge(const JitConstants& jit)
    {
        AddConstants(jit._constants);
    }

    JitDefinitions GetDefinitions() const 
    {
        JitDefinitions definitons;
        definitons.reserve(_constants.size() * 6); //assuming max 6 pairs per jit_constant

        for (auto& constant : _constants) {
            auto def = constant->GetDefinitions();
            definitons.insert(definitons.end(), def.begin(), def.end());
        }
        return definitons;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// MakeBaseParamsJitConstants
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
inline JitConstants MakeBaseParamsJitConstants(const BaseParams& params)
{
    const bool relu =
        params.activationFunc == ActivationFunction::RELU ||
        params.activationFunc == ActivationFunction::RELU_NEGATIVE_SLOPE;
    const float negative_slope =
        params.activationFunc == ActivationFunction::RELU_NEGATIVE_SLOPE ?
        params.nlParams.m : 0.f;

    bool fp16_unit_used = params.output.GetDType() == Datatype::F16;
    for (const auto& i : params.inputs)
    {
        fp16_unit_used |= i.GetDType() == Datatype::F16;
    }

    JitConstants jit{
        MakeJitConstant("OUTPUT",               params.output),
        MakeJitConstant("FP16_SUPPORTED",       static_cast<int>(fp16_unit_used)),                      // TODO: use engine
        MakeJitConstant("FP16_UNIT_USED",       static_cast<int>(fp16_unit_used)),
        MakeJitConstant("UNIT_TYPE",            fp16_unit_used ? "half" : "float"),
        MakeJitConstant("UNIT_VAL_ZERO",        fp16_unit_used ? "0.0h" : "0.0f"),
        MakeJitConstant("UNIT_VAL_MAX",         fp16_unit_used ? "HALF_MAX" : "FLT_MAX"),
        MakeJitConstant("UNIT_VAL_MIN",         "-(UNIT_VAL_MAX)"),
        MakeJitConstant("TO_UNIT_TYPE_V1(v)",   fp16_unit_used ? "convert_half(v)" : "(float)(v)"),
        MakeJitConstant("RELU",                 static_cast<int>(relu)),                                // TODO: remove it
        MakeJitConstant("NEGATIVE_SLOPE",       negative_slope),
        MakeJitConstant("NL_M",                 params.nlParams.m),
        MakeJitConstant("NL_N",                 params.nlParams.n),
        MakeJitConstant("ACTIVATION_FUNCTION_"  + toString(params.activationFunc), ""),
        MakeJitConstant("TYPE_"                 + toString(params.output.GetDType()), ""),
        
    };

    if (params.inputs.size() >= 1)
    {
        // default input is input 0
        jit.AddConstant(MakeJitConstant("INPUT", params.inputs[0]));                                    // TODO: remove it
    }

    for (size_t i = 0; i < params.inputs.size(); i++)
    {
        jit.AddConstant(MakeJitConstant("INPUT" + std::to_string(i), params.inputs[i]));
    }

    return jit;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// MakeWeightBiasParamsJitConstants
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
inline JitConstants MakeWeightBiasParamsJitConstants(const WeightBiasParams& params)
{
    JitConstants jit = MakeBaseParamsJitConstants(params);
    jit.AddConstants({
        MakeJitConstant("FILTER",                       params.weights),
        MakeJitConstant("BIAS_TERM",                    static_cast<int>(!params.bias.empty())),
        MakeJitConstant("FILTER_OUTPUT_FEATURE_NUM",    "FILTER_OFM_NUM"),  // TODO: remove it
        MakeJitConstant("FILTER_INPUT_FEATURE_NUM",     "FILTER_IFM_NUM"),  // TODO: remove it
    });

    if (params.bias.empty() == false)
    {
        jit.AddConstant(MakeJitConstant("BIAS", params.bias[0]));
    }

    return jit;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// MakeConvolutionJitConstants
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
inline JitConstants MakeConvolutionParamsJitConstants(const ConvolutionParams& params)
{
    JitConstants jit = MakeWeightBiasParamsJitConstants(params);
    const auto& padding = params.convParams.padding;
    const auto& input = params.inputs[0];
    
    int64_t input_offset_with_padding = (int64_t)input.GetOffset() - padding.x*input.X().pitch - input.Y().pitch*padding.y;
    input_offset_with_padding = std::max(input_offset_with_padding, (int64_t)0);

    jit.AddConstants({
        MakeJitConstant("STRIDE",                       params.convParams.stride),
        MakeJitConstant("PADDING",                      params.convParams.padding),
        MakeJitConstant("DILATION",                     params.convParams.dilation),
        MakeJitConstant("FILTER_ARRAY_NUM",             params.convParams.split),
        MakeJitConstant("INPUT_OFFSET_WITH_PADDING",    input_offset_with_padding),
    });

    return jit;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// MakeFullyConnectedJitConstants
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
inline JitConstants MakeFullyConnectedJitConstants(const FullyConnectedParams& params)
{
    JitConstants jit = MakeWeightBiasParamsJitConstants(params);
    const auto& input = params.inputs[0];
    const auto x_size = input.Length() / input.Batch().v;

    jit.AddConstants({
        MakeJitConstant("INPUT_ELEMENTS_COUNT",      x_size),
        MakeJitConstant("WEIGHTS_BATCH_NUM",         "FILTER_OFM_NUM"),     // TODO: remove it
    });

    return jit;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// MakeLRNJitConstants
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
inline JitConstants MakeLRNJitConstants(const LRNParams& params)
{
    JitConstants jit = MakeBaseParamsJitConstants(params);

    const auto& np = params.lrnParams;

    const auto padding = (np.localSize - 1) / 2;

    jit.AddConstants({
        MakeJitConstant("LOCAL_SIZE",   np.localSize),
        MakeJitConstant("PADDING",      padding),
        MakeJitConstant("ALPHA",        np.alpha),
        MakeJitConstant("BETA",         np.beta),
        MakeJitConstant("K",            np.k),
    });

    return jit;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// MakePoolingJitConstants
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
inline JitConstants MakePoolingJitConstants(const PoolingParams& params)
{
    JitConstants jit = MakeBaseParamsJitConstants(params);

    const auto& pp = params.poolParams;

    jit.AddConstants({
        MakeJitConstant("WINDOW",   pp.poolSize),
        MakeJitConstant("POOL",     pp.poolSize),   // TODO: remove it or WINDOW
        MakeJitConstant("STRIDE",   pp.poolStride),
        MakeJitConstant("PADDING",  pp.poolPad),
        MakeJitConstant(toString(pp.poolType) + "_POOLING", 1),
    });

    return jit;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// MakeSoftmaxJitConstants
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
inline JitConstants MakeSoftmaxJitConstants(const SoftmaxParams& params)
{
    JitConstants jit = MakeBaseParamsJitConstants(params);

    const auto& sm = params.smParams;

    jit.AddConstants({
        MakeJitConstant("ALONG_" + toString(sm.dim), ""),
    });

    return jit;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// MakeNormalizeJitConstants
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
inline JitConstants MakeNormalizeJitConstants(const NormalizeParams& params)
{
    JitConstants jit = MakeBaseParamsJitConstants(params);

    const auto& np = params.normParams;

    jit.AddConstants({
        MakeJitConstant("SCALE_TABLE",          np.scaleTable),
        MakeJitConstant("EPSILON",              np.epsilon),
        MakeJitConstant(toString(np.normMode),  ""),
    });

    return jit;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// MakePermuteJitConstants
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
inline JitConstants MakePermuteJitConstants(const PermuteParams& params)
{
    JitConstants jit = MakeBaseParamsJitConstants(params);

    jit.AddConstants({
        MakeJitConstant("PERMUTE_ORDER", params.permuteParams.order)
    });

    return jit;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// MakeDeconvolutionJitConstants
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
inline JitConstants MakeDeconvolutionJitConstants(const DeconvolutionParams& params)
{
    JitConstants jit = MakeWeightBiasParamsJitConstants(params);
    const auto& dp = params.deconvParams;
    const auto& padding = dp.padding;
    const auto& input = params.inputs[0];

    int64_t input_offset_with_padding = (int64_t)input.GetOffset() - padding.x*input.X().pitch - input.Y().pitch*padding.y;
    input_offset_with_padding = std::max(input_offset_with_padding, (int64_t)0);

    jit.AddConstants({
        MakeJitConstant("STRIDE",                       dp.stride),
        MakeJitConstant("PADDING",                      dp.padding),
        MakeJitConstant("DILATION",                     dp.dilation),
        MakeJitConstant("FILTER_ARRAY_NUM",             dp.split),
        MakeJitConstant("INPUT_OFFSET_WITH_PADDING",    input_offset_with_padding),
    });

    return jit;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// MakeEltwiseJitConstants
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
inline JitConstants MakeEltwiseJitConstants(const EltwiseParams& params)
{
    JitConstants jit = MakeBaseParamsJitConstants(params);

    jit.AddConstants({
        MakeJitConstant("ELTWISE_LAYOUT_BASED", params.eltwiseParams.layoutBased),
    });

    return jit;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// MakeReorderBaseJitConstants
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
inline JitConstants MakeReorderBaseJitConstants(const ReorderBaseParams& params)
{
    return MakeBaseParamsJitConstants(params);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// MakeReorderJitConstants
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
inline JitConstants MakeReorderJitConstants(const ReorderParams& params)
{
    JitConstants jit = MakeReorderBaseJitConstants(params);

    jit.AddConstant(MakeJitConstant("MEAN_SUBTRUCT_" + toString(params.reorderParams.mode), 1));

    if (params.reorderParams.mode == MeanSubtructMode::INSIDE_PARAMS)
    {
        jit.AddConstant(MakeJitConstant("VALUE_TO_SUBTRACT", params.reorderParams.meanValues));
    }
    else if (params.reorderParams.mode == MeanSubtructMode::IN_BUFFER)
    {
        jit.AddConstant(MakeJitConstant("MEAN_SUBTRUCT", params.reorderParams.mean));
    }

    Datatype calc_type = params.inputs[0].GetDType();
    jit.AddConstants({
        MakeJitConstant("CALC_TYPE",                      ToCLType(calc_type)),
        MakeJitConstant("TO_CALC_TYPE",      "convert_" + ToCLType(calc_type)),
    });

    return jit;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// MakeReorderWeightsJitConstants
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
inline JitConstants MakeReorderWeightsJitConstants(const ReorderWeightsParams& params)
{
    const auto& input = params.reorderParams.input;
    const auto& output = params.reorderParams.output;
    const bool fp16Supported = output.GetDType() == WeightsType::F16 || input.GetDType() == WeightsType::F16;

    JitConstants jit{
        MakeJitConstant("FP16_SUPPORTED",   static_cast<int>(fp16Supported)),                      // TODO: use engine
        MakeJitConstant("FP16_UNIT_USED",   static_cast<int>(fp16Supported)),
        MakeJitConstant("INPUT",            input),
        MakeJitConstant("OUTPUT",           output),
    };

    return jit;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// MakeROIPoolingV1JitConstants
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
inline JitConstants MakeROIPoolingV1JitConstants(const ROIPoolingV1Params& params)
{
    JitConstants jit = MakeBaseParamsJitConstants(params);

    const auto& rp = params.roiParams;

    jit.AddConstants({
        MakeJitConstant("POOLED_HEIGHT",     rp.pooledHeight),
        MakeJitConstant("POOLED_WIDTH",      rp.pooledWidth),
        MakeJitConstant("SPATIAL_SCALE",     rp.spatialScale),
        MakeJitConstant("GORUP_SIZE",        rp.groupSize),
        MakeJitConstant(toString(rp.mode) + "_POOLING", 1),
    });

    return jit;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// MakeConcatenationJitConstants
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
inline JitConstants MakeConcatenationJitConstants(const ConcatenationParams& params)
{
    JitConstants jit = MakeBaseParamsJitConstants(params);

    jit.AddConstants({
        MakeJitConstant("CONCAT_" + toString(params.concatParams.axis), 1),
    });

    return jit;
}

}
