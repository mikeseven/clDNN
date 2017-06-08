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

#include "kernel_base.h"
#include "jitter.h"
#include <sstream>
#include <assert.h>
#include "api/CPP/tensor.hpp"

namespace KernelSelector {

    using jit_definitions = KernelSelector::gpu::jit_definitions;
    using jit_constants = KernelSelector::gpu::jit_constants;
    using tensor_vt = cldnn::tensor::value_type;

    struct CommonDispatchData
    {
        size_t gws0, gws1, gws2;
        size_t lws0, lws1, lws2;
        bool fp16_unit_used;           ///< Value indicating that FP16 half precision floating point type will be used (instead of single precision).
        float effiency;
    };

    class IGKKernelBase : public KernelBase
    {
    public:
        using KernelBase::KernelBase;
        virtual ~IGKKernelBase() {}

    protected:
        std::string create_jit_from_template(const std::string& template_name, jit_definitions definitions, std::string kernel_name) const;
        std::string get_entry_point(const std::string& template_name, const std::string& layer_id) const;
        ArgumentDescpirtor get_args_desc(uint32_t num_of_input, bool use_weights, bool use_bias) const;
        KernelString get_kernel_string(std::string kernel_name, std::string jit, std::string entry_point, std::string exe_mode = ROUND_ROBIN) const;
        void fill_cl_kernel_data(clKernelData& kernel, const CommonDispatchData& run_info, std::string kernel_map_name, std::string jit, std::string entry_point, bool weights = false, bool bias = false) const;
        jit_constants get_common_jit_constants(const BaseParams& params, const CommonDispatchData& kd) const;
    };

    inline cldnn::format params_2_cldnn(Tensor::DataLayout l)
    {
        switch (l)
        {
        case Tensor::DataLayout::bf: return cldnn::format::bfyx;
        case Tensor::DataLayout::fb: return cldnn::format::yxfb;
        case Tensor::DataLayout::bfyx: return cldnn::format::bfyx;
        case Tensor::DataLayout::yxfb: return cldnn::format::yxfb;
        case Tensor::DataLayout::byxf: return cldnn::format::byxf;
        case Tensor::DataLayout::fyxb: return cldnn::format::fyxb;
        case Tensor::DataLayout::brfyx: return cldnn::format::bfyx;
        default:
            assert(0);
            return cldnn::format::bfyx;
        }
    }

    inline cldnn::tensor ks_tensor_2_tensor(const DataTensor& ksTensor)
    {
        return{ 
            static_cast<tensor_vt>(ksTensor.batch().v),
            static_cast<tensor_vt>(ksTensor.feature().v),
            static_cast<tensor_vt>(ksTensor.x().v),
            static_cast<tensor_vt>(ksTensor.y().v) };
    }

    inline bool check_activation_support(ActivationFunction func)
    {
        switch (func)
        {
        case KernelSelector::ActivationFunction::NONE:
        case KernelSelector::ActivationFunction::RELU:
        case KernelSelector::ActivationFunction::RELU_NEGATIVE_SLOPE:
            return true;
        default:
            return false;
        }
    }
}