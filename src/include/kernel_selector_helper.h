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

#pragma once
#include "api/C/cldnn.h"
#include "api/CPP/program.hpp"
#include "program_impl.h"
#include "gpu/ocl_toolkit.h"
#include "tensor_type.h"
#include "kernel_selector_params.h"

using namespace cldnn;

inline KernelSelector::Datatype ConvertDataType(data_types dt)
{
    switch (dt)
    {
    case cldnn::data_types::f16:    return KernelSelector::Datatype::F16;
    case cldnn::data_types::f32:    return KernelSelector::Datatype::F32;
    default:
        assert(0);
        return KernelSelector::Datatype::F16;
    }
}

inline KernelSelector::WeightsType ConvertWeightsType(data_types dt)
{
    switch (dt)
    {
    case cldnn::data_types::i8:     return KernelSelector::WeightsType::INT8;
    case cldnn::data_types::f16:    return KernelSelector::WeightsType::F16;
    case cldnn::data_types::f32:    return KernelSelector::WeightsType::F32;
    default:
        assert(0);
        return KernelSelector::WeightsType::F16;
    }
}

inline KernelSelector::DataLayout FromDataLayout(format f)
{
    switch (f)
    {
    case format::bfyx:              return KernelSelector::DataLayout::bfyx;
    case format::yxfb:              return KernelSelector::DataLayout::yxfb;
    case format::byxf:              return KernelSelector::DataLayout::byxf;
    case format::fyxb:              return KernelSelector::DataLayout::fyxb;
    case format::bs_x_bsv16:        return KernelSelector::DataLayout::bs_f_bsv16__af8;
    case format::bs_xs_xsv8_bsv8:   return KernelSelector::DataLayout::bs_f_bsv8__af8;
    case format::bs_xs_xsv8_bsv16:  return KernelSelector::DataLayout::bs_f_bsv16__af8;
//     case format::brfyx:          return KernelSelector::DataLayout::brfyx;
    default:
        return KernelSelector::DataLayout::bfyx;
    }
}

static inline cldnn::format ToDataLayout(KernelSelector::DataLayout l)
{
    switch (l)
    {
    case KernelSelector::DataLayout::bf:                return cldnn::format::bfyx;
    case KernelSelector::DataLayout::fb:                return cldnn::format::fyxb;
    case KernelSelector::DataLayout::bfyx:              return cldnn::format::bfyx;
    case KernelSelector::DataLayout::yxfb:              return cldnn::format::yxfb;
    case KernelSelector::DataLayout::byxf:              return cldnn::format::byxf;
    case KernelSelector::DataLayout::fyxb:              return cldnn::format::fyxb;
    case KernelSelector::DataLayout::bs_f_bsv8__af8:    return cldnn::format::bs_xs_xsv8_bsv8;
    case KernelSelector::DataLayout::bs_f_bsv16__af8:   return cldnn::format::bs_x_bsv16;
    case KernelSelector::DataLayout::brfyx:             return cldnn::format::bfyx;
    default:
        return cldnn::format::bfyx;
        break;
    }
}

inline KernelSelector::WeightsLayout FromWeightsLayout(format f)
{
    switch (f)
    {
    case format::bfyx:              return KernelSelector::WeightsLayout::oiyx;
    case format::fyxb:              return KernelSelector::WeightsLayout::iyxo;
    case format::byxf:              return KernelSelector::WeightsLayout::oyxi;
    case format::yxfb:              return KernelSelector::WeightsLayout::yxio;
    case format::os_iyx_osv16:      return KernelSelector::WeightsLayout::os_iyx_osv16;
    case format::bs_xs_xsv8_bsv8:   return KernelSelector::WeightsLayout::os_i_osv8__ai8;
    case format::bs_xs_xsv8_bsv16:  return KernelSelector::WeightsLayout::os_i_osv16__ai8;
    case format::bs_x_bsv16:        return KernelSelector::WeightsLayout::os_i_osv16;
    default:
        return KernelSelector::WeightsLayout::oi;
    }
}

static inline cldnn::format ToWeightsLayout(KernelSelector::WeightsLayout l)
{
    switch (l)
    {
    case KernelSelector::WeightsLayout::oi:
    case KernelSelector::WeightsLayout::oiyx:               return cldnn::format::bfyx;
    case KernelSelector::WeightsLayout::oyxi:               return cldnn::format::byxf;
    case KernelSelector::WeightsLayout::io:
    case KernelSelector::WeightsLayout::iyxo:               return cldnn::format::fyxb;
    case KernelSelector::WeightsLayout::yxio:               return cldnn::format::yxfb;
    case KernelSelector::WeightsLayout::os_iyx_osv16:       return cldnn::format::os_iyx_osv16;
    case KernelSelector::WeightsLayout::os_i_osv16:         return cldnn::format::bs_x_bsv16;
    case KernelSelector::WeightsLayout::os_i_osv8__ai8:     return cldnn::format::bs_xs_xsv8_bsv8;
    case KernelSelector::WeightsLayout::os_i_osv16__ai8:    return cldnn::format::bs_xs_xsv8_bsv16;
    default:
        return cldnn::format::bfyx;
    }
}

inline KernelSelector::DataTensor ConvertDataTensor(const layout& l, uint32_t split = 1, const tensor additional_offest = {})
{
    const auto& pad = l.data_padding;
    const auto& vals = l.size.sizes(l.format);
    const auto& add_offsets = additional_offest.sizes(l.format);
    const auto& lower_pad = pad.lower_size().sizes(l.format);
    const auto& upper_pad = pad.upper_size().sizes(l.format);
    const auto ks_layout = FromDataLayout(l.format);
    KernelSelector::Tensor::NDims vec(KernelSelector::Tensor::ChannelsCount(ks_layout));

    size_t pitch = 1;
    size_t offset = 0;
    for (size_t i = 0; i < vec.size(); i++)
    {
        const size_t tensor_index = vec.size() - 1 - i;
        const auto d = vals[tensor_index];
        const auto lp = lower_pad[tensor_index];
        const auto up = upper_pad[tensor_index];

        auto& elm = vec[i];
        elm.v = static_cast<size_t>(d);
        elm.pitch = pitch;

        offset += pitch*(lp + add_offsets[tensor_index]);
        pitch *= (d + lp + up);
    }

    const int feature_index = KernelSelector::Tensor::Channelndex(ks_layout, KernelSelector::Tensor::DataChannelName::FEATURE);
    vec[feature_index].v /= split;

    return KernelSelector::DataTensor(
        ConvertDataType(l.data_type),
        ks_layout,
        KernelSelector::Tensor::PADDED_VAL::ZERO,
        offset,
        vec);
}

inline KernelSelector::WeightsTensor ConvertWeightsTensor(const layout& l)
{
    assert(l.format.dimension() == 4);
    const auto& t = l.size.sizes(format::bfyx);
    const auto base_layout = KernelSelector::WeightsLayout::oiyx;
    const auto ks_type = ConvertWeightsType(l.data_type);
    const auto ks_layout = FromWeightsLayout(l.format);
    std::vector<size_t> vec(KernelSelector::Tensor::ChannelsCount(base_layout));

    for (size_t i = 0; i < vec.size(); i++)
    {
        const size_t tensor_index = t.size() - 1 - i;
        const auto d = t[tensor_index];
        vec[i] = static_cast<size_t>(d);
    }

    return KernelSelector::WeightsTensor(
        ks_type,
        base_layout,
        KernelSelector::PADDED_VAL::UNDEFINED,
        0,
        vec).Transform(ks_layout);
}

template <typename PType>
inline void ConvertActivationFuncParams(const PType primitive, KernelSelector::BaseParams& params)
{
    if (primitive->with_activation)
    {
        const float negative_slope = primitive->activation_negative_slope;
        if (negative_slope)
        {
            params.nlParams.m = negative_slope;
            params.activationFunc = KernelSelector::ActivationFunction::RELU_NEGATIVE_SLOPE;
        }
        else
        {
            params.activationFunc = KernelSelector::ActivationFunction::RELU;
        }
    }
    else
    {
        params.activationFunc = KernelSelector::ActivationFunction::NONE;
    }
}

template <typename ParamsT, typename ArgT>
inline ParamsT GetDefaultParams(const ArgT& arg, uint32_t split = 1)
{
    ParamsT params;
    
    const auto& input_layout    = arg.input().get_output_layout();
    const auto& output_layout   = arg.get_output_layout();

    params.inputs[0] = ConvertDataTensor(input_layout, split);
    params.output = ConvertDataTensor(output_layout, split);

    params.layerID = arg.id();

    return params;
}

template <typename ParamsT, typename ArgT>
inline ParamsT GetWeightsBiasDefaultParams(const ArgT& arg, uint32_t split = 1)
{
    ParamsT params = GetDefaultParams<ParamsT>(arg, split);

    const auto& weights_layout = arg.weights().get_output_layout();
    params.weights = ConvertWeightsTensor(weights_layout);

    if (arg.bias_term())
    {
        const auto& bias_layout = arg.bias().get_output_layout();
        // bias per output is not supported on cldnn
        params.bias.push_back(ConvertDataTensor(bias_layout).FlattenFeatureAndSpatials());
    }

    return params;
}

template <typename OptionalParamsT>
inline OptionalParamsT GetDefaultOptionalParams(const program_impl& program)
{
    OptionalParamsT params;
    
    const auto& context = program.get_engine()->get_context();
    const auto& engine_info = context->get_engine_info();

    params.bSupportSubGroupExt = program.get_engine()->get_context()->extension_supported("cl_intel_subgroups_short");
    params.maxWorkGroupSize = engine_info.max_work_group_size;
    params.maxLocalMemSize = engine_info.max_local_mem_size;
    return params;
}

template <typename OptionalParamsT>
inline OptionalParamsT GetDefaultWeightsBiasOptionalParams(const program_impl& program)
{
    OptionalParamsT params = GetDefaultOptionalParams<OptionalParamsT>(program);
    //params.allow_padding = true; - TODO:
    params.allowWeightsReorder = program.get_options().get<build_option_type::optimize_data>()->enabled();
    return params;
}