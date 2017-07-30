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

#include "fully_connected_kernel_selector.h"
#include "fully_connected_kernel_bfyx_ref.h"
#include "fully_connected_kernel_bf_io_gemm.h"
#include "fully_connected_kernel_bs_f_bsv16_b1.h"
#include "fully_connected_kernel_bs_f_bsv16_af8.h"
#include "fully_connected_kernel_bs_f_bsv8_af8.h"
#include "fully_connected_kernel_yxfb_ref.h"
#include "fully_connected_kernel_fb_oi_ref.h"
#include "fully_connected_kernel_fb_io_ref.h"
#include "fully_connected_kernel_bf_io_ref.h"
#include "fully_connected_kernel_fb_oi_b8_ref.h"
#include "fully_connected_kernel_fb_io_b8_f8.h"
#include "fully_connected_kernel_fb_io_block.h"
 
namespace KernelSelector {

    FullyConnectedKernelSelctor::FullyConnectedKernelSelctor()
    {
        Attach<FullyConnected_bfyx_Ref>();
        Attach<FullyConnected_bf_io_GEMM>();
        Attach<FullyConnected_bs_f_bsv16_b1>();
        Attach<FullyConnected_bs_f_bsv16_af8>();
        Attach<FullyConnected_bs_f_bsv8_af8>();
        Attach<FullyConnected_yxfb_ref>();
        Attach<FullyConnected_fb_oi_ref>();
        Attach<FullyConnected_fb_io_ref>();
        Attach<FullyConnected_bf_io_ref>();
        Attach<FullyConnected_fb_oi_b8_ref>();
        Attach<FullyConnected_fb_io_block>();
        Attach<FullyConnected_fb_io_b8_f8>();
    }

    KernelsData FullyConnectedKernelSelctor::GetBestKernels(const Params& params, const OptionalParams& options) const
    {
        return GetNaiveBestKernel(params, options, KernelType::FULLY_CONNECTED);
    }
}