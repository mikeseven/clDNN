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

#include "convolution_kernel_winograd_fused.h"
#include "kernel_selector_utils.h"

using namespace std;

namespace KernelSelector {


	std::string ConvolutionKernel_WinogradFused::GetEntryPoint(const std::string& templateName, const std::string& layerID, const OptionalParams& options) const
	{
		std::string kernelID = layerID;

		if (kernelID.empty() || !options.meaningfulKernelsNames)
		{
			kernelID = templateName;
		}

		std::replace(kernelID.begin(), kernelID.end(), '.', '_');

		kernelID += "_" + std::to_string(UniqeID());

		return kernelID;
	}

	
    ParamsKey ConvolutionKernel_WinogradFused::GetSupportedKey() const
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
		k.EnableInputLayout(DataLayout::bfyx);
		k.EnableOutputLayout(DataLayout::bfyx);
        //k.EnableAllInputLayout();
        //k.EnableAllOutputLayout();

        // Supported tensor offset/pitch/padding
        k.EnableTensorOffset();
        k.EnableTensorPitches();
        k.EnableBatching();

        // Supported convolution extra data
        //k.EnableDilation();
        k.EnableBiasPerFeature();
        k.EnableBiasPerOutput();
        k.EnableNonBiasTerm();

        // Supported convolution which get a split index and uses it as a view on the input/output (Alexnet only)
        //k.EnableSplitSupport();

        return k;
    }

	template<class T>
	void* Alloc(std::initializer_list<uint32_t> const& dims)
	{
		auto d = dims.begin();

		int size = 1;
		for (size_t n = 0; n < dims.size(); n++)
			size *= d[n];

		return malloc(size*sizeof(float));
	}

	#define winograd_size_fused_d 4 // current options 4 for F(2,3), 6 for F(4,3), 8 for F(6,3), 10 for F(8,3)
	int winograd_size_fused = winograd_size_fused_d;
	void CPUWinogradTransformKernelsFused::TransformKernles
	(
		void* inKernel,
		void* outKernel
	) const
	{
		// tile rows
		//uint32_t TROWS = (uint32_t)params.inputs[0].Y().v; //ROWS

		// kernel rows & cols in Winograd domain
		uint32_t KROWS = (uint32_t)params.weights.Y().v; //  KROWS;
		uint32_t KCOLS = (uint32_t)params.weights.X().v; //  ?

		uint32_t TCOLS = winograd_size_fused;// (uint32_t)type;
		uint32_t KROWSW = 3; // Layer::KROWS;
		uint32_t KCOLSW = TCOLS;

		const uint32_t ODEPTH = (uint32_t)params.output.Feature().v;
		const uint32_t IDEPTH = (uint32_t)params.inputs[0].Feature().v;

		//outKernel = Alloc<float>({ ODEPTH, IDEPTH, KROWSW, KCOLSW });

		Tensor<float, 4> k_yxio(const_cast<float*>((float*)inKernel), { KROWS, KCOLS, IDEPTH, ODEPTH });
		Tensor<float, 4> k_oiyx(const_cast<float*>((float*)inKernel), { ODEPTH, IDEPTH, KROWS, KCOLS });
		
		//Tensor<float, 4> kw((float*)outKernel, { KROWSW, KCOLSW, IDEPTH, ODEPTH });
		uint32_t weightOSplit  = ((uint32_t)(winograd_size_fused * 2));
		Tensor<float, 5> kw((float*)outKernel, { IDEPTH, ODEPTH/ weightOSplit, KCOLSW, KROWSW, weightOSplit });

		// output depth
		for (uint32_t od = 0; od < ODEPTH; od++)
		{
			// input depth
			for (uint32_t id = 0; id < IDEPTH; id++)
			{
				// Note: this condition used to be needed for Alexnet, but changed at some point
				if (false)//params.weights.GetLayout() == WeightsLayout::yxio)
				{ 
					if (winograd_size_fused == 4)
					{
						*kw[id][od / 8][0][0][od % 8] = *k_yxio[0][0][id][od];
						*kw[id][od / 8][1][0][od % 8] = (*k_yxio[0][0][id][od] + *k_yxio[1][0][id][od] + *k_yxio[2][0][id][od]) / 2.0f;
						*kw[id][od / 8][2][0][od % 8] = *kw[id][od / 8][1][0][od % 8] - *k_yxio[1][0][id][od];
						*kw[id][od / 8][3][0][od % 8] = *k_yxio[2][0][id][od];

						*kw[id][od / 8][0][1][od % 8] = *k_yxio[0][1][id][od];
						*kw[id][od / 8][1][1][od % 8] = (*k_yxio[0][1][id][od] + *k_yxio[1][1][id][od] + *k_yxio[2][1][id][od]) / 2.0f;
						*kw[id][od / 8][2][1][od % 8] = *kw[id][od / 8][1][1][od % 8] - *k_yxio[1][1][id][od];
						*kw[id][od / 8][3][1][od % 8] = *k_yxio[2][1][id][od];

						*kw[id][od / 8][0][2][od % 8] = *k_yxio[0][2][id][od];
						*kw[id][od / 8][1][2][od % 8] = (*k_yxio[0][2][id][od] + *k_yxio[1][2][id][od] + *k_yxio[2][2][id][od]) / 2.0f;
						*kw[id][od / 8][2][2][od % 8] = *kw[id][od / 8][1][2][od % 8] - *k_yxio[1][2][id][od];
						*kw[id][od / 8][3][2][od % 8] = *k_yxio[2][2][id][od];
					}
					/*else if (winograd_size_fused == 6)
					{
						*kw[0][0][id][od] = (+6 * *k_yxio[0][0][id][od]) / 24.0f;
						*kw[1][0][id][od] = (-4 * *k_yxio[0][0][id][od] - 4 * *k_yxio[0][1][id][od] - 4 * *k_yxio[0][2][id][od]) / 24.0f;
						*kw[2][0][id][od] = (-4 * *k_yxio[0][0][id][od] + 4 * *k_yxio[0][1][id][od] - 4 * *k_yxio[0][2][id][od]) / 24.0f;
						*kw[3][0][id][od] = (+*k_yxio[0][0][id][od] + 2 * *k_yxio[0][1][id][od] + 4 * *k_yxio[0][2][id][od]) / 24.0f;
						*kw[4][0][id][od] = (+*k_yxio[0][0][id][od] - 2 * *k_yxio[0][1][id][od] + 4 * *k_yxio[0][2][id][od]) / 24.0f;
						*kw[5][0][id][od] = (+24 * *k_yxio[0][2][id][od]) / 24.0f;

						*kw[0][1][id][od] = (+6 * *k_yxio[1][0][id][od]) / 24.0f;
						*kw[1][1][id][od] = (-4 * *k_yxio[1][0][id][od] - 4 * *k_yxio[1][1][id][od] - 4 * *k_yxio[1][2][id][od]) / 24.0f;
						*kw[2][1][id][od] = (-4 * *k_yxio[1][0][id][od] + 4 * *k_yxio[1][1][id][od] - 4 * *k_yxio[1][2][id][od]) / 24.0f;
						*kw[3][1][id][od] = (+*k_yxio[1][0][id][od] + 2 * *k_yxio[1][1][id][od] + 4 * *k_yxio[1][2][id][od]) / 24.0f;
						*kw[4][1][id][od] = (+*k_yxio[1][0][id][od] - 2 * *k_yxio[1][1][id][od] + 4 * *k_yxio[1][2][id][od]) / 24.0f;
						*kw[5][1][id][od] = (+24 * *k_yxio[1][2][id][od]) / 24.0f;

						*kw[0][2][id][od] = (+6 * *k_yxio[2][0][id][od]) / 24.0f;
						*kw[1][2][id][od] = (-4 * *k_yxio[2][0][id][od] - 4 * *k_yxio[2][1][id][od] - 4 * *k_yxio[2][2][id][od]) / 24.0f;
						*kw[2][2][id][od] = (-4 * *k_yxio[2][0][id][od] + 4 * *k_yxio[2][1][id][od] - 4 * *k_yxio[2][2][id][od]) / 24.0f;
						*kw[3][2][id][od] = (+*k_yxio[2][0][id][od] + 2 * *k_yxio[2][1][id][od] + 4 * *k_yxio[2][2][id][od]) / 24.0f;
						*kw[4][2][id][od] = (+*k_yxio[2][0][id][od] - 2 * *k_yxio[2][1][id][od] + 4 * *k_yxio[2][2][id][od]) / 24.0f;
						*kw[5][2][id][od] = (+24 * *k_yxio[2][2][id][od]) / 24.0f;
					}*/
				}
				else //if (params.weights.GetLayout() == WeightsLayout::oiyx)
				{
					if (winograd_size_fused == 4)
					{
						*kw[id][od / 8][0][0][od % 8] = *k_oiyx[od][id][0][0];
						*kw[id][od / 8][1][0][od % 8] = (*k_oiyx[od][id][0][0] + *k_oiyx[od][id][1][0] + *k_oiyx[od][id][2][0]) / 2.0f;
						*kw[id][od / 8][2][0][od % 8] = *kw[id][od / 8][1][0][od % 8] - *k_oiyx[od][id][1][0];
						*kw[id][od / 8][3][0][od % 8] = *k_oiyx[od][id][2][0];

						*kw[id][od / 8][0][1][od % 8] = *k_oiyx[od][id][0][1];
						*kw[id][od / 8][1][1][od % 8] = (*k_oiyx[od][id][0][1] + *k_oiyx[od][id][1][1] + *k_oiyx[od][id][2][1]) / 2.0f;
						*kw[id][od / 8][2][1][od % 8] = *kw[id][od / 8][1][1][od % 8] - *k_oiyx[od][id][1][1];
						*kw[id][od / 8][3][1][od % 8] = *k_oiyx[od][id][2][1];

						*kw[id][od / 8][0][2][od % 8] = *k_oiyx[od][id][0][2];
						*kw[id][od / 8][1][2][od % 8] = (*k_oiyx[od][id][0][2] + *k_oiyx[od][id][1][2] + *k_oiyx[od][id][2][2]) / 2.0f;
						*kw[id][od / 8][2][2][od % 8] = *kw[id][od / 8][1][2][od % 8] - *k_oiyx[od][id][1][2];
						*kw[id][od / 8][3][2][od % 8] = *k_oiyx[od][id][2][2];
					}
					else if (winograd_size_fused == 6)
					{
						*kw[id][od / 8][0][0][od % 8] = (+6 * *k_oiyx[od][id][0][0]) / 24.0f;
						*kw[id][od / 8][1][0][od % 8] = (-4 * *k_oiyx[od][id][0][0] - 4 * *k_oiyx[od][id][1][0] - 4 * *k_oiyx[od][id][2][0]) / 24.0f;
						*kw[id][od / 8][2][0][od % 8] = (-4 * *k_oiyx[od][id][0][0] + 4 * *k_oiyx[od][id][1][0] - 4 * *k_oiyx[od][id][2][0]) / 24.0f;
						*kw[id][od / 8][3][0][od % 8] = (+*k_oiyx[od][id][0][0] + 2 * *k_oiyx[od][id][1][0] + 4 * *k_oiyx[od][id][2][0]) / 24.0f;
						*kw[id][od / 8][4][0][od % 8] = (+*k_oiyx[od][id][0][0] - 2 * *k_oiyx[od][id][1][0] + 4 * *k_oiyx[od][id][2][0]) / 24.0f;
						*kw[id][od / 8][5][0][od % 8] = (+24 * *k_oiyx[od][id][2][0]) / 24.0f;

						*kw[id][od / 8][0][1][od % 8] = (+6 * *k_oiyx[od][id][0][1]) / 24.0f;
						*kw[id][od / 8][1][1][od % 8] = (-4 * *k_oiyx[od][id][0][1] - 4 * *k_oiyx[od][id][1][1] - 4 * *k_oiyx[od][id][2][1]) / 24.0f;
						*kw[id][od / 8][2][1][od % 8] = (-4 * *k_oiyx[od][id][0][1] + 4 * *k_oiyx[od][id][1][1] - 4 * *k_oiyx[od][id][2][1]) / 24.0f;
						*kw[id][od / 8][3][1][od % 8] = (+*k_oiyx[od][id][0][1] + 2 * *k_oiyx[od][id][1][1] + 4 * *k_oiyx[od][id][2][1]) / 24.0f;
						*kw[id][od / 8][4][1][od % 8] = (+*k_oiyx[od][id][0][1] - 2 * *k_oiyx[od][id][1][1] + 4 * *k_oiyx[od][id][2][1]) / 24.0f;
						*kw[id][od / 8][5][1][od % 8] = (+24 * *k_oiyx[od][id][2][1]) / 24.0f;

						*kw[id][od / 8][0][2][od % 8] = (+6 * *k_oiyx[od][id][0][2]) / 24.0f;
						*kw[id][od / 8][1][2][od % 8] = (-4 * *k_oiyx[od][id][0][2] - 4 * *k_oiyx[od][id][1][2] - 4 * *k_oiyx[od][id][2][2]) / 24.0f;
						*kw[id][od / 8][2][2][od % 8] = (-4 * *k_oiyx[od][id][0][2] + 4 * *k_oiyx[od][id][1][2] - 4 * *k_oiyx[od][id][2][2]) / 24.0f;
						*kw[id][od / 8][3][2][od % 8] = (+*k_oiyx[od][id][0][2] + 2 * *k_oiyx[od][id][1][2] + 4 * *k_oiyx[od][id][2][2]) / 24.0f;
						*kw[id][od / 8][4][2][od % 8] = (+*k_oiyx[od][id][0][2] - 2 * *k_oiyx[od][id][1][2] + 4 * *k_oiyx[od][id][2][2]) / 24.0f;
						*kw[id][od / 8][5][2][od % 8] = (+24 * *k_oiyx[od][id][2][2]) / 24.0f;
					}
					else if (winograd_size_fused == 8)
					{
						*kw[id][od / 16][0][0][od % 16] = (float)(+90.0 / 90 * *k_oiyx[od][id][0][0]);
						*kw[id][od / 16][1][0][od % 16] = (float)(-20.0 / 90 * *k_oiyx[od][id][0][0] - 20.0 / 90 * *k_oiyx[od][id][1][0] - 20.0 / 90 * *k_oiyx[od][id][2][0]);
						*kw[id][od / 16][2][0][od % 16] = (float)(-20.0 / 90 * *k_oiyx[od][id][0][0] + 20.0 / 90 * *k_oiyx[od][id][1][0] - 20.0 / 90 * *k_oiyx[od][id][2][0]);
						*kw[id][od / 16][3][0][od % 16] = (float)(+1.0 / 90 * *k_oiyx[od][id][0][0] + 2.0 / 90 * *k_oiyx[od][id][1][0] + 4.0 / 90 * *k_oiyx[od][id][2][0]);
						*kw[id][od / 16][4][0][od % 16] = (float)(+1.0 / 90 * *k_oiyx[od][id][0][0] - 2.0 / 90 * *k_oiyx[od][id][1][0] + 4.0 / 90 * *k_oiyx[od][id][2][0]);
						*kw[id][od / 16][5][0][od % 16] = (float)(+64.0 / 90 * *k_oiyx[od][id][0][0] + 32.0 / 90 * *k_oiyx[od][id][1][0] + 16.0 / 90 * *k_oiyx[od][id][2][0]);
						*kw[id][od / 16][6][0][od % 16] = (float)(+64.0 / 90 * *k_oiyx[od][id][0][0] - 32.0 / 90 * *k_oiyx[od][id][1][0] + 16.0 / 90 * *k_oiyx[od][id][2][0]);
						*kw[id][od / 16][7][0][od % 16] = (float)(+90.0 / 90 * *k_oiyx[od][id][2][0]);

						*kw[id][od / 16][0][1][od % 16] = (float)(+90.0 / 90 * *k_oiyx[od][id][0][1]);
						*kw[id][od / 16][1][1][od % 16] = (float)(-20.0 / 90 * *k_oiyx[od][id][0][1] - 20.0 / 90 * *k_oiyx[od][id][1][1] - 20.0 / 90 * *k_oiyx[od][id][2][1]);
						*kw[id][od / 16][2][1][od % 16] = (float)(-20.0 / 90 * *k_oiyx[od][id][0][1] + 20.0 / 90 * *k_oiyx[od][id][1][1] - 20.0 / 90 * *k_oiyx[od][id][2][1]);
						*kw[id][od / 16][3][1][od % 16] = (float)(+1.0 / 90 * *k_oiyx[od][id][0][1] + 2.0 / 90 * *k_oiyx[od][id][1][1] + 4.0 / 90 * *k_oiyx[od][id][2][1]);
						*kw[id][od / 16][4][1][od % 16] = (float)(+1.0 / 90 * *k_oiyx[od][id][0][1] - 2.0 / 90 * *k_oiyx[od][id][1][1] + 4.0 / 90 * *k_oiyx[od][id][2][1]);
						*kw[id][od / 16][5][1][od % 16] = (float)(+64.0 / 90 * *k_oiyx[od][id][0][1] + 32.0 / 90 * *k_oiyx[od][id][1][1] + 16.0 / 90 * *k_oiyx[od][id][2][1]);
						*kw[id][od / 16][6][1][od % 16] = (float)(+64.0 / 90 * *k_oiyx[od][id][0][1] - 32.0 / 90 * *k_oiyx[od][id][1][1] + 16.0 / 90 * *k_oiyx[od][id][2][1]);
						*kw[id][od / 16][7][1][od % 16] = (float)(+90.0 / 90 * *k_oiyx[od][id][2][1]);

						*kw[id][od / 16][0][2][od % 16] = (float)(+90.0 / 90 * *k_oiyx[od][id][0][2]);
						*kw[id][od / 16][1][2][od % 16] = (float)(-20.0 / 90 * *k_oiyx[od][id][0][2] - 20.0 / 90 * *k_oiyx[od][id][1][2] - 20.0 / 90 * *k_oiyx[od][id][2][2]);
						*kw[id][od / 16][2][2][od % 16] = (float)(-20.0 / 90 * *k_oiyx[od][id][0][2] + 20.0 / 90 * *k_oiyx[od][id][1][2] - 20.0 / 90 * *k_oiyx[od][id][2][2]);
						*kw[id][od / 16][3][2][od % 16] = (float)(+1.0 / 90 * *k_oiyx[od][id][0][2] + 2.0 / 90 * *k_oiyx[od][id][1][2] + 4.0 / 90 * *k_oiyx[od][id][2][2]);
						*kw[id][od / 16][4][2][od % 16] = (float)(+1.0 / 90 * *k_oiyx[od][id][0][2] - 2.0 / 90 * *k_oiyx[od][id][1][2] + 4.0 / 90 * *k_oiyx[od][id][2][2]);
						*kw[id][od / 16][5][2][od % 16] = (float)(+64.0 / 90 * *k_oiyx[od][id][0][2] + 32.0 / 90 * *k_oiyx[od][id][1][2] + 16.0 / 90 * *k_oiyx[od][id][2][2]);
						*kw[id][od / 16][6][2][od % 16] = (float)(+64.0 / 90 * *k_oiyx[od][id][0][2] - 32.0 / 90 * *k_oiyx[od][id][1][2] + 16.0 / 90 * *k_oiyx[od][id][2][2]);
						*kw[id][od / 16][7][2][od % 16] = (float)(+90.0 / 90 * *k_oiyx[od][id][2][2]);
					}
					/*else if (winograd_size_fused == 10)
					{
						*kw[0][0][id][od] = (float)(+16.000000000 * *k_oiyx[od][id][0][0]);
						*kw[1][0][id][od] = (float)(-0.237037037 * *k_oiyx[od][id][0][0] - 0.237037037 * *k_oiyx[od][id][0][1] - 0.237037037 * *k_oiyx[od][id][0][2]);
						*kw[2][0][id][od] = (float)(-0.237037037 * *k_oiyx[od][id][0][0] + 0.237037037 * *k_oiyx[od][id][0][1] - 0.237037037 * *k_oiyx[od][id][0][2]);
						*kw[3][0][id][od] = (float)(+0.002821869 * *k_oiyx[od][id][0][0] + 0.005643739 * *k_oiyx[od][id][0][1] + 0.011287478 * *k_oiyx[od][id][0][2]);
						*kw[4][0][id][od] = (float)(+0.002821869 * *k_oiyx[od][id][0][0] - 0.005643739 * *k_oiyx[od][id][0][1] + 0.011287478 * *k_oiyx[od][id][0][2]);
						*kw[5][0][id][od] = (float)(+3.792592593 * *k_oiyx[od][id][0][0] + 1.896296296 * *k_oiyx[od][id][0][1] + 0.948148148 * *k_oiyx[od][id][0][2]);
						*kw[6][0][id][od] = (float)(+3.792592593 * *k_oiyx[od][id][0][0] - 1.896296296 * *k_oiyx[od][id][0][1] + 0.948148148 * *k_oiyx[od][id][0][2]);
						*kw[7][0][id][od] = (float)(-11.558377425 * *k_oiyx[od][id][0][0] - 2.889594356 * *k_oiyx[od][id][0][1] - 0.722398589 * *k_oiyx[od][id][0][2]);
						*kw[8][0][id][od] = (float)(-11.558377425 * *k_oiyx[od][id][0][0] + 2.889594356 * *k_oiyx[od][id][0][1] - 0.722398589 * *k_oiyx[od][id][0][2]);
						*kw[9][0][id][od] = (float)(+1.000000000 * *k_oiyx[od][id][0][2]);

						*kw[0][1][id][od] = (float)(+16.000000000 * *k_oiyx[od][id][1][0]);
						*kw[1][1][id][od] = (float)(-0.237037037 * *k_oiyx[od][id][1][0] - 0.237037037 * *k_oiyx[od][id][1][1] - 0.237037037 * *k_oiyx[od][id][1][2]);
						*kw[2][1][id][od] = (float)(-0.237037037 * *k_oiyx[od][id][1][0] + 0.237037037 * *k_oiyx[od][id][1][1] - 0.237037037 * *k_oiyx[od][id][1][2]);
						*kw[3][1][id][od] = (float)(+0.002821869 * *k_oiyx[od][id][1][0] + 0.005643739 * *k_oiyx[od][id][1][1] + 0.011287478 * *k_oiyx[od][id][1][2]);
						*kw[4][1][id][od] = (float)(+0.002821869 * *k_oiyx[od][id][1][0] - 0.005643739 * *k_oiyx[od][id][1][1] + 0.011287478 * *k_oiyx[od][id][1][2]);
						*kw[5][1][id][od] = (float)(+3.792592593 * *k_oiyx[od][id][1][0] + 1.896296296 * *k_oiyx[od][id][1][1] + 0.948148148 * *k_oiyx[od][id][1][2]);
						*kw[6][1][id][od] = (float)(+3.792592593 * *k_oiyx[od][id][1][0] - 1.896296296 * *k_oiyx[od][id][1][1] + 0.948148148 * *k_oiyx[od][id][1][2]);
						*kw[7][1][id][od] = (float)(-11.558377425 * *k_oiyx[od][id][1][0] - 2.889594356 * *k_oiyx[od][id][1][1] - 0.722398589 * *k_oiyx[od][id][1][2]);
						*kw[8][1][id][od] = (float)(-11.558377425 * *k_oiyx[od][id][1][0] + 2.889594356 * *k_oiyx[od][id][1][1] - 0.722398589 * *k_oiyx[od][id][1][2]);
						*kw[9][1][id][od] = (float)(+1.000000000 * *k_oiyx[od][id][1][2]);

						*kw[0][2][id][od] = (float)(+16.000000000 * *k_oiyx[od][id][2][0]);
						*kw[1][2][id][od] = (float)(-0.237037037 * *k_oiyx[od][id][2][0] - 0.237037037 * *k_oiyx[od][id][2][1] - 0.237037037 * *k_oiyx[od][id][2][2]);
						*kw[2][2][id][od] = (float)(-0.237037037 * *k_oiyx[od][id][2][0] + 0.237037037 * *k_oiyx[od][id][2][1] - 0.237037037 * *k_oiyx[od][id][2][2]);
						*kw[3][2][id][od] = (float)(+0.002821869 * *k_oiyx[od][id][2][0] + 0.005643739 * *k_oiyx[od][id][2][1] + 0.011287478 * *k_oiyx[od][id][2][2]);
						*kw[4][2][id][od] = (float)(+0.002821869 * *k_oiyx[od][id][2][0] - 0.005643739 * *k_oiyx[od][id][2][1] + 0.011287478 * *k_oiyx[od][id][2][2]);
						*kw[5][2][id][od] = (float)(+3.792592593 * *k_oiyx[od][id][2][0] + 1.896296296 * *k_oiyx[od][id][2][1] + 0.948148148 * *k_oiyx[od][id][2][2]);
						*kw[6][2][id][od] = (float)(+3.792592593 * *k_oiyx[od][id][2][0] - 1.896296296 * *k_oiyx[od][id][2][1] + 0.948148148 * *k_oiyx[od][id][2][2]);
						*kw[7][2][id][od] = (float)(-11.558377425 * *k_oiyx[od][id][2][0] - 2.889594356 * *k_oiyx[od][id][2][1] - 0.722398589 * *k_oiyx[od][id][2][2]);
						*kw[8][2][id][od] = (float)(-11.558377425 * *k_oiyx[od][id][2][0] + 2.889594356 * *k_oiyx[od][id][2][1] - 0.722398589 * *k_oiyx[od][id][2][2]);
						*kw[9][2][id][od] = (float)(+1.000000000 * *k_oiyx[od][id][2][2]);
					}*/
				}
			}
		}
	}

	CPUWinogradTransformKernelsFused::CPUWinogradTransformKernelsFused(ConvolutionParams in_params, WinogradSizeFused F) :
		m_winogradSize(F)
	{
		params = in_params;
		uint32_t nIFM = (uint32_t)params.inputs[0].Feature().v;
		uint32_t nOFM = (uint32_t)params.output.Feature().v;
		m_nKernels = nIFM*nOFM;

		//InitalizeTransformMaps();
		//m_TileSize = F.first + F.second - 1;

	}

	void CPUWinogradTransformKernelsFused::Execute(void* input, std::size_t inputSize, void* output, std::size_t outputSize) const
	{
		/*uint32_t odepth = (uint32_t) .output.Feature().v;
		uint32_t idepth = (uint32_t)newParams.inputs[0].Feature().v;
		uint32_t rows = (uint32_t)newParams.inputs[0].Y().v;
		uint32_t cols = (uint32_t)newParams.inputs[0].X().v;*/

		TransformKernles(input, output);
		if (outputSize==0)
			std::cout << "MADE IT TO TRANSFORM! inputSize:" << inputSize << "-" << outputSize <<  std::endl;
	}

	uint32_t CPUWinogradTransformKernelsFused::GetOutputSizeInBytes() const
	{
		uint32_t TCOLS = winograd_size_fused;// (uint32_t)type;
		uint32_t KROWSW = 3; // Layer::KROWS;
		uint32_t KCOLSW = TCOLS;

		const uint32_t ODEPTH = (uint32_t)params.output.Feature().v;
		const uint32_t IDEPTH = (uint32_t)params.inputs[0].Feature().v;
		return (ODEPTH* IDEPTH* KROWSW* KCOLSW) * sizeof(float);
        //return m_TileSize *m_TileSize* m_nKernels * 4;
	}


	WeightsReorderParams ConvolutionKernel_WinogradFused::GetWeightReorderParams(const ConvolutionParams& newParams, WeightsType wType) const
	{
		WeightsReorderParams weights_reorder_params;
		auto reorder = CPUWinogradTransformKernelsFused(newParams, WinogradSizeFused(2, 3));
		weights_reorder_params.engine = WeightsReorderParams::Engine::CPU;
		weights_reorder_params.cpuKernel = std::make_shared<CPUWinogradTransformKernelsFused>(reorder);
		weights_reorder_params.newBufferSize = reorder.GetOutputSizeInBytes();
		weights_reorder_params.dtype = wType;
		return weights_reorder_params;
	}

	/*
	def idiv(n, d) :
		return (n + d - 1) // d
	def idiv_up(n, d) :
		return idiv(n, d) * d
	*/

	uint32_t idiv(uint32_t n, uint32_t d)
		{
			return (n/d + ((n%d)?1:0));
		}								
	uint32_t idiv_up(uint32_t n, uint32_t d)
		{
			 return idiv(n, d) * d;
		}	

	KernelsData ConvolutionKernel_WinogradFused::GetKernelsData(const Params& params, const OptionalParams& options) const
	{
		assert(params.GetType() == KernelType::CONVOLUTION && options.GetType() == KernelType::CONVOLUTION);
		const uint32_t numOfkernels = 1; // unfused
		KernelData kd = KernelData::Default<ConvolutionParams>(params, numOfkernels);
		ConvolutionParams& newParams = *static_cast<ConvolutionParams*>(kd.params.get());
		size_t inpSiz = 
			newParams.inputs[0].PhysicalSizeInBytes();
		size_t outSiz = 
			newParams.output.PhysicalSizeInBytes();

		kd.internalBufferSizes = { inpSiz*5, outSiz*5 }; // Set correct internal buffer size to match winograd expansion
		
		uint32_t ww = (uint32_t)newParams.weights.X().v;
		uint32_t wh = (uint32_t)newParams.weights.Y().v;
		if ((ww != 3) || (wh != 3) ||
			((uint32_t)newParams.convParams.stride.x != 1) ||
			((uint32_t)newParams.convParams.stride.y != 1) ||
			((uint32_t)newParams.convParams.filterSize.x != 3) ||
			((uint32_t)newParams.convParams.filterSize.y != 3) ||
			((uint32_t)newParams.convParams.padding.x > 1) ||
			((uint32_t)newParams.convParams.padding.y > 1)  ||
			((uint32_t)newParams.output.Feature().v % 32) ||
		    ((uint32_t)newParams.inputs[0].Feature().v % 32) )
		{
			return {  };
		}
		kd.estimatedTime = FORCE_PRIORITY_1;  //always use

		bool succeed = UpdateWeightsParams(
			newParams,
			options,
			{
				WeightsLayout::oiyx,
				WeightsLayout::yxio,
				WeightsLayout::iyxo,
				WeightsLayout::oyxi,
			},
			kd.weightsReorderParams);
		if (!succeed)
		{
			return{};
		}

		//return {};

		//if (!newParams.output.GetDims()[0].pad.before) return {};

		/*static uint32_t possibleWinogradCntr = 0;
		possibleWinogradCntr++;
		if (!((possibleWinogradCntr >= 12) && (possibleWinogradCntr <=14)))
			return {};*/

		std::shared_ptr<KernelString> kernel_string = std::make_shared<KernelString>();

		uint32_t input_pad_y = +(uint32_t)newParams.inputs[0].Y().pad.before + (uint32_t)newParams.inputs[0].Y().pad.after;
		uint32_t input_pad_x = +(uint32_t)newParams.inputs[0].X().pad.before + (uint32_t)newParams.inputs[0].X().pad.after;

		uint32_t odepth = (uint32_t)newParams.output.Feature().v;
		uint32_t idepth = (uint32_t)newParams.inputs[0].Feature().v;
		uint32_t rows = (uint32_t)newParams.inputs[0].Y().v + input_pad_y;
		uint32_t cols = (uint32_t)newParams.inputs[0].X().v + input_pad_x;

		//cout << "[OD:" << odepth << "," << "ID:" << idepth << ",ROWS:" << rows << ",COLS:" << cols << "]\n";

		uint32_t TROWS = rows; 
		//while ((TROWS - 3 /*Layer::KROWS*/ + 1) % 8) TROWS++; // Must be multiple of 8, check in kernel for overaccess during transform

		uint32_t TCOLS = winograd_size_fused; // (uint32_t)type;
		uint32_t KROWSW = 3; // Layer::KROWS;
		uint32_t KCOLSW = TCOLS;
		//uint32_t OCOLS = TCOLS - 2; //layer.OCOLS;
		//uint32_t STRIDE_X = TCOLS - 2; //layer.OCOLS;
		
		//uint32_t NR_TILES_X = (uint32_t)((cols - 3 /*KCOLS*/ + 1) / OCOLS);
		/*uint32_t padded_cols = cols;
		// while ((padded_cols - 3 /*Layer::KCOLS  + 1) % OCOLS) padded_cols++; 
		uint32_t NR_TILES_X = (uint32_t)(padded_cols / OCOLS); */


		std::string JitParams = "";

		uint32_t X_PADDING_BEFORE = (uint32_t)newParams.output.GetDims()[0].pad.before;
		uint32_t Y_PADDING_BEFORE = (uint32_t)newParams.output.GetDims()[1].pad.before;
		uint32_t X_PADDING_AFTER  = (uint32_t)newParams.output.GetDims()[0].pad.after;
		uint32_t Y_PADDING_AFTER  = (uint32_t)newParams.output.GetDims()[1].pad.after;

		JitParams += " -DH=" + std::to_string(rows);
		JitParams += " -DW=" + std::to_string(cols);
		JitParams += " -DP=" + std::to_string(rows -3 + 1 + Y_PADDING_BEFORE + Y_PADDING_AFTER);
		JitParams += " -DQ=" + std::to_string(cols -3 + 1 + X_PADDING_BEFORE + X_PADDING_AFTER);

		JitParams += " -DC=" + std::to_string(idepth);
		JitParams += " -DK=" + std::to_string(odepth);
		JitParams += " -DR=" + std::to_string(3);
		JitParams += " -DS=" + std::to_string(3);
		JitParams += " -DN=" + std::to_string(1);
		JitParams += " -Dpx=" + std::to_string(0);
		JitParams += " -Dpy=" + std::to_string(0);
		JitParams += " -Dsx=" + std::to_string(1);
		JitParams += " -Dsy=" + std::to_string(1);

		//PYTHON: extra_build_options += ' -DC4_up16=%d' % (idiv_up(C, 16)//4,)
		uint32_t C4_up16 = idiv_up(idepth, 16)/4;
		JitParams += " -DC4_up16=" + std::to_string(C4_up16);
		
		JitParams += " -DIDEPTH=" + std::to_string(idepth);
		JitParams += " -DODEPTH=" + std::to_string(odepth);
		JitParams += " -DTROWS=" + std::to_string(TROWS);
		JitParams += " -DTCOLS=" + std::to_string(TCOLS);
		JitParams += " -DKROWSW=" + std::to_string(KROWSW);
		JitParams += " -DKCOLSW=" + std::to_string(KCOLSW);

		//TODO: this assumes bfyx. should be better way to determine output padding
		JitParams += " -DX_PADDING_BEFORE=" + std::to_string(X_PADDING_BEFORE);
		JitParams += " -DX_PADDING_AFTER=" + std::to_string(X_PADDING_AFTER);
		JitParams += " -DY_PADDING_BEFORE=" + std::to_string(Y_PADDING_BEFORE);
		JitParams += " -DY_PADDING_AFTER=" + std::to_string(Y_PADDING_AFTER);

		JitParams += (!newParams.bias.empty())?" -DBIAS_TERM ":"" ;

		JitParams += " -DNL_M=" + std::to_string(newParams.activationParams.m);
		JitParams += " -DNL_N=" + std::to_string(newParams.activationParams.n);

		if (newParams.bias.size() != 0)
		{
			JitParams += " -DOUTPUT_SIZE_X=" + std::to_string((uint32_t)newParams.output.X().v);
			JitParams += " -DOUTPUT_SIZE_Y=" + std::to_string((uint32_t)newParams.output.Y().v);
			const bool sameDims = newParams.bias[0].SameDims(newParams.output);
			JitParams += " -DBIAS_PER_OUTPUT=" + std::to_string(sameDims);
		}

		kd.weightsReorderParams = GetWeightReorderParams(newParams, WeightsType::F32);

		// Kernel to tranform to Winograd //
		auto& kernel1 = kd.kernels[0];
		auto codes = (winograd_size_fused == 4) ? db.get("convolution_gpu_winograd.fused2x3") : // 4
			         (winograd_size_fused == 6) ? db.get("convolution_gpu_winograd.fused4x3") : // 6
			         (winograd_size_fused == 8) ? db.get("convolution_gpu_winograd.fused6x3") : // 8
			                                      db.get("convolution_gpu_winograd.fused8x3");  // 10
		if (codes.size())
		{
			auto entry_point = (winograd_size_fused == 4) ? "winograd_fused2x3" : // 4
				               (winograd_size_fused == 6) ? "winograd_fused4x3" : // 6
				               (winograd_size_fused == 8) ? "winograd_fused6x3" : // 8
				                                            "winograd_fused8x3";  // 10
			kernel_string->str = codes[0];
			kernel_string->jit = "  ";
			kernel_string->options = " -cl-mad-enable -cl-fast-relaxed-math -cl-std=CL2.0 " + JitParams;
			kernel_string->entry_point = entry_point;
			kernel_string->batch_compilation = true;
		}
		kernel1.kernelString = kernel_string;
		kernel1.arguments.push_back({ ArgumentDescriptor::Types::INPUT, 0 }); 
		kernel1.arguments.push_back({ ArgumentDescriptor::Types::WEIGHTS, 0 });
		if (!newParams.bias.empty())
			kernel1.arguments.push_back({ ArgumentDescriptor::Types::BIAS, 0 });
		kernel1.arguments.push_back({ ArgumentDescriptor::Types::OUTPUT, 0 });

		uint32_t P = rows-2;
		uint32_t Q = cols-2;
		//uint32_t C = idepth;
		uint32_t K = odepth;
		uint32_t N = 1;
		//uint32_t R = 3;
		//uint32_t S = 3;
		
		uint32_t global_step[3] = { 14, 4, 16 * 8 };
		if (winograd_size_fused == 8) global_step[1] = 6;
		uint32_t local_size[3] = { 8, 1, 8 };

		uint32_t gx = idiv(Q, global_step[0]) * local_size[0];
		uint32_t gy = idiv(P, global_step[1]) * local_size[1];
		uint32_t gz = idiv(N*K * 8, global_step[2]) * local_size[2];
		kernel1.workGroups.global = { gx, gy, gz };
		kernel1.workGroups.local  = { local_size[0],  local_size[1], local_size[2] };

		/*size_t gx = (cols / 14 + ((cols % 14) ? 1 : 0)) * 8;
		size_t gy = (rows / 4  + ((rows % 4 ) ? 1 : 0)) * 1;
		size_t gz = ( (odepth / 16) + ((odepth % 16) ? 1 : 0) ) * 8;
		kernel1.workGroups.global = { gx, gy, gz }; 
        kernel1.workGroups.local =  { 8,  1,  8  };*/

		return{ kd };
	}
}