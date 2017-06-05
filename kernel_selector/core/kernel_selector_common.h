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
#include <string>
#include "cache/program_cache.h"
#include "cache/cache_types.h"
#include "cache/primitive_db.h"
#include "kernel_selector_params.h"
#include <float.h>
 
#define AGE_BASED "-cl-no-subgroup-ifp"
#define ROUND_ROBIN ""

namespace KernelSelector {

#ifndef UNUSED
#define UNUSED(a) (void)a
#endif


// TODO: current solution until we will have kernel selection time based
#define FORCE_PRIORITY_1 (0.0000001f)
#define FORCE_PRIORITY_2 (0.0000002f)
#define FORCE_PRIORITY_3 (0.0000003f)
#define FORCE_PRIORITY_4 (0.0000004f)
#define FORCE_PRIORITY_5 (0.0000005f)
#define FORCE_PRIORITY_6 (0.0000006f)
#define FORCE_PRIORITY_7 (0.0000007f)
#define FORCE_PRIORITY_8 (0.0000008f)
#define FORCE_PRIORITY_9 (0.0000009f)
#define DONT_USE_IF_HAVE_SOMETHING_ELSE (1000000.f)
#define NOT_SUPPORTED (FLT_MAX)

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // usings
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    using context_device = KernelSelector::gpu::context_device;
    using primitive_db = KernelSelector::gpu::cache::primitive_db;
    using program_cache = KernelSelector::gpu::cache::program_cache;
    using binary_data = KernelSelector::gpu::cache::binary_data;


    std::string GetStringEnv(const char* varName);

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // KernelString
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct KernelString
    {
        std::string str;
        std::string jit;
        std::string options;
        std::string entry_point;
        bool        batch_compilation = false;
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // WorkGroupSizes
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct WorkGroupSizes
    {
        cl::NDRange global = cl::NullRange;
        cl::NDRange local = cl::NullRange;
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // ArgumentDescpirtor
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct ArgumentDescpirtor
    {
        union ValueT
        {
            uint8_t  u8;
            uint16_t u16;
            uint32_t u32;
            uint64_t u64;
            int8_t   s8;
            int16_t  s16;
            int32_t  s32;
            int64_t  s64;
            float    f32;
            double   f64;
        };

        enum class Types
        {
            INPUT,
            OUTPUT,
            WEIGHTS,
            BIAS,
            LOOKUP_TABLE,
            SPLIT,
            UINT8,
            UINT16,
            UINT32,
            UINT64,
            INT8,
            INT16,
            INT32,
            INT64,
            FLOAT32,
            FLOAT64,
        };

        struct Args
        {
            Types t;
            ValueT v;
        };

        std::vector<Args> data;

        bool SetArguments(
            cl::Kernel& kernel,
            std::vector<const cl::Buffer*> inputs,
            const cl::Buffer* output,
            const cl::Buffer* weights,
            const cl::Buffer* bias,
            const cl::Buffer* lookup_table,
            uint32_t split) const;
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // clKernelData
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct clKernelData
    {
        binary_data GetBinary(context_device cl_context, program_cache& compiler) const;

        KernelString kernel_string;
        WorkGroupSizes work_groups;
        ArgumentDescpirtor args_desc;
        // TODO: maybe we want an estimated time per cl kernel
        // float estimated_time = DONT_USE_IF_HAVE_SOMETHING_ELSE;
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // CPUKernel
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct CPUKernel
    {
        virtual WeightsLayout GetInputLayout() const { return WeightsLayout::oiyx; }
        virtual void Execute(void* input, size_t input_size, void* output, size_t output_size) const = 0;
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // WeightsReorder
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct WeightsReorderParams
    {
        enum class Engine
        {
            NONE,
            CPU,
            GPU
        };

        Engine engine = Engine::NONE;
        clKernelData cl_kernel;
        std::shared_ptr<CPUKernel> cpu_kernel;
        size_t new_buffer_size = 0;
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // InputReorderParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct InputReorderParams
    {
        bool pad_buffer = false;
        DataLayout layout = DataLayout::bfyx;
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // KernelData
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct KernelData
    {
        std::shared_ptr<Params> params;
        std::vector<clKernelData> kernels;
        float estimated_time = DONT_USE_IF_HAVE_SOMETHING_ELSE;

        bool reorder_input = false;
        WeightsReorderParams weights_reorder_params;

        template <typename T>
        inline static KernelData Default(const Params& _params, size_t kernel_nums)
        {
            KernelData kd;
            const T& orgParams = static_cast<const T&>(_params);
            kd.params = std::make_shared<T>(orgParams);
            kd.kernels.resize(kernel_nums);
            kd.estimated_time = DONT_USE_IF_HAVE_SOMETHING_ELSE; // for KW
            kd.reorder_input = false; // for KW
            return kd;
        }
    };

    using KernelsData = std::vector<KernelData>;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // to string functions
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    inline std::string toString(ActivationFunction activation)
    {
        std::string method("LINEAR");
        switch (activation)
        {
        case ActivationFunction::LOGISTIC:              method = "LOGISTIC"; break;
        case ActivationFunction::HYPERBOLIC_TAN:        method = "HYPERBOLIC_TAN"; break;
        case ActivationFunction::RELU:                  method = "RELU"; break;
        case ActivationFunction::RELU_NEGATIVE_SLOPE:   method = "RELU_NEGATIVE_SLOPE"; break;
        case ActivationFunction::BRELU:                 method = "BRELU"; break;
        case ActivationFunction::SOFTRELU:              method = "SOFTRELU"; break;
        case ActivationFunction::ABS:                   method = "ABS"; break;
        case ActivationFunction::SQUARE:                method = "SQUARE"; break;
        case ActivationFunction::SQRT:                  method = "SQRT"; break;
        case ActivationFunction::LINEAR:                method = "LINEAR"; break;
        case ActivationFunction::NONE:                  method = "NONE"; break;
        default: break;
        }
        return method;
    }

    inline std::string toString(DataLayout l)
    {
        switch (l)
        {
        case KernelSelector::DataLayout::bf:     return "BF";
        case KernelSelector::DataLayout::fb:     return "FB";
        case KernelSelector::DataLayout::bfyx:   return "BFYX";
        case KernelSelector::DataLayout::yxfb:   return "YXFB";
        case KernelSelector::DataLayout::byxf:   return "BYXF";
        case KernelSelector::DataLayout::fyxb:   return "FYXB";
        case KernelSelector::DataLayout::brfyx:  return "BRFYX";
        default: return "";
        }
    }

    inline std::string toString(Datatype dType)
    {
        switch (dType)
        {
        case Datatype::F16: return "F16";
        case Datatype::F32: return "F32";
        default: return "";
        }
    }

    inline std::string toString(WeightsType wType)
    {
        switch (wType)
        {
        case WeightsType::F16: return "F16";
        case WeightsType::F32: return "F32";
        case WeightsType::INT8: return "INT8";
        default: return "";
        }
    }

    inline std::string toString(ConvertTypes dType)
    {
        switch (dType)
        {
        case ConvertTypes::U8 : return "U8";
        case ConvertTypes::U16: return "U16";
        case ConvertTypes::U32: return "U32";
        case ConvertTypes::S8 : return "S8";
        case ConvertTypes::S16: return "S16";
        case ConvertTypes::S32: return "S32";
        case ConvertTypes::F16: return "F16";
        case ConvertTypes::F32: return "F32";
        default: return "";
        }
    }

    inline std::string toString(KernelType kt)
    {
        switch (kt)
        {
        case KernelType::UNKNOWN: return "UNKNOWN";
        case KernelType::CONVOLUTION: return "CONVOLUTION";
        case KernelType::LRN: return "LRN";
        case KernelType::POOLING: return "POOLING";
        case KernelType::ROI_POOLING: return "ROI_POOLING";
        case KernelType::FULLY_CONNECTED: return "FULLY_CONNECTED";
        case KernelType::LOCALLY_CONNECTED: return "LOCALLY_CONNECTED";
        case KernelType::ACTIVATION: return "ACTIVATION";
        case KernelType::SOFT_MAX: return "SOFT_MAX";
        case KernelType::ELTWISE: return "ELTWISE";
        case KernelType::TABLE_LOOKUP: return "TABLE_LOOKUP";
        case KernelType::REORDER: return "REORDER";
        case KernelType::CONVERT: return "CONVERT";
        default:
            return "";
        }
    }

    inline std::string toString(EltwiseMode b_mode)
    {
        switch (b_mode)
        {
        case EltwiseMode::ADD: return "ADD";
        case EltwiseMode::SUB: return "SUB";
        case EltwiseMode::MUL: return "MUL";
        case EltwiseMode::DIV: return "DIV";
        default:
            return "";
        }
    }

    inline std::string toString(ReorderMode mode)
    {
        switch (mode)
        {
        case ReorderMode::xyzw: return "XYZW";
        case ReorderMode::xywz: return "XYWZ";
        case ReorderMode::xwyz: return "XWYZ";
        case ReorderMode::wxyz: return "WXYZ";
        case ReorderMode::xzyw: return "XZYW";
        case ReorderMode::zyxw: return "ZYXW";
        case ReorderMode::yxzw: return "YXZW";
        default:
            return "XYZW";
            break;
        }
    }

    inline std::string toString(WeightsLayout layout)
    {
        switch (layout)
        {
        case WeightsLayout::oi:                 return "OI";
        case WeightsLayout::io:                 return "IO";
        case WeightsLayout::oiyx:               return "OIYX";
        case WeightsLayout::oyxi:               return "OYXI";
        case WeightsLayout::iyxo:               return "IYXO";
        case WeightsLayout::yxio:               return "YXIO";
        case WeightsLayout::os_iyx_osv16:       return "OS_IYX_OSV16";
        case WeightsLayout::os_is_isv8_osv8:    return "OS_IS_ISV8_OSV8";
        case WeightsLayout::os_i_osv16:         return "OS_I_OSV16";
        case WeightsLayout::iyxo_om16x2_axy:    return "IYXO_OM16X2_AXY";
        case WeightsLayout::iyxo_om16x2_ax_g32: return "IYXO_OM16X2_AX_G32";
        case WeightsLayout::iyxo_om8x2_ax_g32:  return "IYXO_OM8X2_AX_G32";
        default:
            return "";
            break;
        }
    }
}