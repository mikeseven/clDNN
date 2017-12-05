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

#include "convolution_kernel_base.h"
 
namespace KernelSelector {
    
	typedef std::pair<uint32_t, uint32_t> WinogradSizeFused;


	class CPUWinogradTransformKernelsFused : public CPUKernel
	{
	public:
		CPUWinogradTransformKernelsFused(ConvolutionParams params, WinogradSizeFused F);

		virtual void Execute(void* input, std::size_t input_size, void* output, std::size_t output_size) const;
		virtual uint32_t GetOutputSizeInBytes() const;
		virtual WeightsType   GetExpectedInputType() { return WeightsType::F32; } 

		template<class T, uint32_t N>
		class Tensor
		{
		private:
			std::vector<uint32_t> dims;
			T* data;

		public:
			Tensor<T, N>(T* data, std::vector<uint32_t> const& dims) : data(data), dims(dims)
			{
			}

		public:
			Tensor<T, N - 1> operator[](uint32_t n)
			{
				uint32_t offset = n;
				for (size_t i = 0; i < N - 1; i++)
					offset *= dims[dims.size() - i - 1];
				return Tensor<T, N - 1>(data + offset, dims);
			}

		};

		template<class T>
		class Tensor<T, 1>
		{
		private:
			T* data;
			uint32_t d0;

		public:
			Tensor<T, 1>(T* data, std::vector<uint32_t> const& dims) : data(data)
			{
				this->d0 = dims[dims.size() - 1];
			}

		public:
			float* operator[](uint32_t n)
			{
				assert(n < d0);
				return &data[n];
			}
		};

	protected:

		//void InitalizeTransformMaps();
		void TransformKernles(void* input, void* output) const;

		ConvolutionParams params;

		std::map<WinogradSizeFused, const float*> m_BMap;
		std::map<WinogradSizeFused, const float*> m_GMap;
		WinogradSizeFused m_winogradSize;
		uint32_t m_TileSize;
		uint32_t m_nKernels;
	};

	
    class ConvolutionKernel_WinogradFused : public ConvolutionKernelBase
    {
	protected:
		virtual std::vector<WeightsLayout> GetSupportedWeightLayouts(const ConvolutionParams&) const override
		{
			return{
				WeightsLayout::yxio,
				WeightsLayout::iyxo,
				WeightsLayout::oyxi,
				WeightsLayout::oiyx,
			};
		}

    public:
		ConvolutionKernel_WinogradFused() : ConvolutionKernelBase("convolution_winograd") {}
        virtual ~ConvolutionKernel_WinogradFused() {}

        virtual KernelsData GetKernelsData(const Params& params, const OptionalParams& options) const override;
        virtual ParamsKey GetSupportedKey() const override;

		//ConvolutionKernel_Winograd::DispatchData SetDefault(const ConvolutionParams& arg) const override;
		std::string GetEntryPoint(const std::string& templateName, const std::string& layerID, const OptionalParams& options) const;
		WeightsReorderParams GetWeightReorderParams(const ConvolutionParams& newParams, WeightsType wType) const;
		//static size_t UniqeID() { return counter++; } // TODO: use interlocked
		//static size_t counter;
    };
}