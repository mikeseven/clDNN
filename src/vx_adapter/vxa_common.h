﻿#pragma once

#include <cstddef>
#include "api/vx_cldnn_adapter.h"
#include "api/vx_cldnn_adapter_types.h"
#include "gpu/ocl_toolkit.h"
#include "gpu/cache/primitive_selector.h"
#include "gpu/cache/cache_types.h"
#include "common/khronos_ocl_clhpp/cl2_wrapper.h"

#ifndef UNUSED
#define UNUSED(a) (void)a
#endif

#define CLDNN_ALIGN(size, alignment) (((size) + (alignment) - 1) / (alignment) * (alignment))

inline bool isAligned(std::size_t val, std::size_t alignment)
{
    return (((val) % (alignment)) == 0);
}

namespace clDNN
{
    using stDims = Dims<std::size_t>;
    using primitive_selector = neural::gpu::manager::primitive_selector;

    bool UseReferenceKernel();
    bool IsSupported(const Params& params);

    struct KernelInfo
    {
        uint subBlockDimM = 1;
        uint subBlockDimK = 1;
        uint subBlockDimN = 1;
        uint localWorkSizeX = 0;
        uint localWorkSizeY = 0;
        uint localWorkSizeZ = 0;
        uint globalWorkSizeX = 0;
        uint globalWorkSizeY = 0;
        uint globalWorkSizeZ = 0;
        uint globalWorkSizeDX = 1;
        uint globalWorkSizeDY = 1;
        uint globalWorkSizeDZ = 1;

        KernelInfo() = default;

        KernelInfo(
            uint sBlockDimM, uint sBlockDimK, uint sBlockDimN,
            uint lWorkSzX, uint lWorkSzY, uint lWorkSzZ,
            uint gWorkDX, uint gWorkDY, uint gWorkDZ) :
            subBlockDimM(sBlockDimM),
            subBlockDimK(sBlockDimK),
            subBlockDimN(sBlockDimN),
            localWorkSizeX(lWorkSzX),
            localWorkSizeY(lWorkSzY),
            localWorkSizeZ(lWorkSzZ),
            globalWorkSizeDX(gWorkDX),
            globalWorkSizeDY(gWorkDY),
            globalWorkSizeDZ(gWorkDZ)
        {}

        void SetGlobalWGS(uint gWorkSzX, uint gWorkSzY, uint gWorkSzZ)
        {
            globalWorkSizeX = gWorkSzX;
            globalWorkSizeY = gWorkSzY;
            globalWorkSizeZ = gWorkSzZ;
        }

        void SetLocalWGS(uint lWorkSzX, uint lWorkSzY, uint lWorkSzZ)
        {
            localWorkSizeX = lWorkSzX;
            localWorkSizeY = lWorkSzY;
            localWorkSizeZ = lWorkSzZ;
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // to string funcs
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    inline std::string toString(ActivationFunction activation)
    {
        std::string method("LINEAR");
        switch (activation)
        {
        case ActivationFunction::LOGISTIC:       method = "LOGISTIC"; break;
        case ActivationFunction::HYPERBOLIC_TAN: method = "HYPERBOLIC_TAN"; break;
        case ActivationFunction::RELU:           method = "RELU"; break;
        case ActivationFunction::BRELU:          method = "BRELU"; break;
        case ActivationFunction::SOFTRELU:       method = "SOFTRELU"; break;
        case ActivationFunction::ABS:            method = "ABS"; break;
        case ActivationFunction::SQUARE:         method = "SQUARE"; break;
        case ActivationFunction::SQRT:           method = "SQRT"; break;
        case ActivationFunction::LINEAR:         method = "LINEAR"; break;
        case ActivationFunction::NONE:           method = "NONE"; break;
        default: break;
        }
        return method;
    }

    inline std::string toString(Datatype dType)
    {
        switch (dType)
        {
        case clDNN::Datatype::F16: return "F16";
        case clDNN::Datatype::F32: return "F32";
        default: return "";
        }
    }

    inline std::string toString(ConvertTypes dType)
    {
        switch (dType)
        {
        case clDNN::ConvertTypes::U8 : return "U8";
        case clDNN::ConvertTypes::U16: return "U16";
        case clDNN::ConvertTypes::U32: return "U32";
        case clDNN::ConvertTypes::S8 : return "S8";
        case clDNN::ConvertTypes::S16: return "S16";
        case clDNN::ConvertTypes::S32: return "S32";
        case clDNN::ConvertTypes::F16: return "F16";
        case clDNN::ConvertTypes::F32: return "F32";
        default: return "";
        }
    }

    inline std::string toString(KernelType kt)
    {
        switch (kt)
        {
        case clDNN::KernelType::UNKNOWN: return "UNKNOWN";
        case clDNN::KernelType::CONVOLUTION: return "CONVOLUTION";
        case clDNN::KernelType::NORMALIZATION: return "NORMALIZATION";
        case clDNN::KernelType::POOLING: return "POOLING";
        case clDNN::KernelType::ROI_POOLING: return "ROI_POOLING";
        case clDNN::KernelType::FULLY_CONNECTED: return "FULLY_CONNECTED";
        case clDNN::KernelType::LOCALLY_CONNECTED: return "LOCALLY_CONNECTED";
        case clDNN::KernelType::ACTIVATION: return "ACTIVATION";
        case clDNN::KernelType::SOFT_MAX: return "SOFT_MAX";
        default:
            return "";
        }
    }

    inline std::string toString(BinaryOpMode b_mode)
    {
        switch (b_mode)
        {
        case clDNN::BinaryOpMode::ADD: return "ADD";
        case clDNN::BinaryOpMode::SUB: return "SUB";
        case clDNN::BinaryOpMode::MUL: return "MUL";
        case clDNN::BinaryOpMode::DIV: return "DIV";
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
        case ReorderMode::zxyw: return "ZXYW";
        case ReorderMode::yxzw: return "YXZW";
        default:
            return "XYZW";
            break;
        }
    }

    inline std::shared_ptr<primitive_selector> GetPrimitiveSelector()
    {
        static std::recursive_mutex mutex;
        static std::shared_ptr<primitive_selector> primitive_handle;
        std::lock_guard<std::recursive_mutex> create_lock{ mutex };
        
        if (primitive_handle == nullptr) 
        {
            primitive_handle = std::make_shared<primitive_selector>();
        }

        return primitive_handle;
    }

    inline uint BytesPerElement(Datatype dt)
    {
        switch (dt)
        {
        case clDNN::Datatype::F16:
            return 2;
            break;
        case clDNN::Datatype::F32:
            return 4;
            break;
        default:
            return 0;
            break;
        }
    }
}