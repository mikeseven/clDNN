#include "windows.h"

#include "CL/cl2.hpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <iterator>
#include <random>
//#include "src/memory_gpu.h"
//#include "api/instrumentation.h"

inline void checkErr(cl_int err, const char * name)
{
	if (err != CL_SUCCESS) {
		std::cerr << "ERROR: " << name << " (" << err << ")" << std::endl;
		exit(EXIT_FAILURE);
	}
}

#define PLATFORM 0
#define DEVICE_TYPE CL_DEVICE_TYPE_GPU

#define OUTPUT_SIZE_IN_FLOATS 10000


float sum_filter[] = { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };
float avg_filter[] = { 1.0 / 9.0, 1.0 / 9.0 , 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0 , 1.0 / 9.0 , 1.0 / 9.0, 1.0 / 9.0 , 1.0 / 9.0 };
float gaussien_filter[] = { 1.0 / 16, 1.0 / 8, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0 / 8, 1.0 / 16, 1.0 / 8, 1.0 / 16 };

float* manual_filters[] = { sum_filter, avg_filter, gaussien_filter };

typedef UINT uint;

void GenerateFilters(float* filters, UINT nKernels)
{
	for (UINT i = 0; i < ARRAYSIZE(manual_filters); i++)
	{
		memcpy(filters + i * 9, manual_filters[i], sizeof(float) * 9);
	}
	for (UINT i = ARRAYSIZE(manual_filters); i < nKernels; i++)
	{
		std::default_random_engine generator;
		const UINT max_rnd = 1000000;
		std::uniform_int_distribution<int> distribution(1, max_rnd);
		auto dice = std::bind(distribution, generator);
		for (UINT j = 0; j < 9; j++)
		{
			filters[i * 9 + j] = (dice() / float(max_rnd));
		}
	}
}

std::string BuildInlineFloats(float* floats, UINT nFloats)
{

	char buff[100];
	sprintf_s(buff, "%f", floats[0]);
	std::string s(buff);
	for (UINT i = 1; i < nFloats; i++)
	{
		sprintf_s(buff, ",%f", floats[i]);
		s += std::string(buff);
	}
	return s;
}

std::string generateBuildOptions(const bool bTransformedKernels, const bool bFastLoad, uint nIFM , uint nOFM, uint ImageW)
{
	/*std::string floatString = BuildInlineFloats(filter, nFloats);
	char buff[100];
	sprintf_s(buff, "%d", nFloats);
	std::string nFloatsStr = "-D NFLOATS=" + std::string(buff);
	std::string opt = "-D INLINE " + nFloatsStr;
	opt += " -D FLOATS=";
	opt += floatString;*/
	char buff[500];
	sprintf_s(buff, "-D OPTIONS -D TRANSFORMED_KERNELS=%d -D N_IFM=%d -D N_OFM=%d -D IMAGE_W=%d -D FAST_LOAD=%d", bTransformedKernels, nIFM, nOFM, ImageW, bFastLoad);
	std::string opt(buff);
	return opt;
}

//transform from yxfb -> bfyx
void interleaveImageData(float* input, float* output, size_t sz, size_t nF)
{
	for (size_t f = 0; f < nF; f++)
	{
		for (size_t x = 0; x < sz; x++)
		{
			for (size_t y = 0; y < sz; y++)
			{

				//TODO
				return;
			}
		}
	}
}

void mul(float* A, float* B, float* C, uint Wa, uint Ha, uint Wb, uint Hb)
{
	for (uint i = 0; i < Ha; i++)
	{
		for (uint j = 0; j < Wb; j++)
		{
			float sum = 0;
			for (uint t = 0; t < Wa; t++)
			{
				sum += A[i * Wa + t] * B[t * Wb + j];
			}
			C[i * Wb + j] = sum;
		}
	}
}

void Transpose(float* M, float* O, uint H, uint W)
{
	for (uint i = 0; i < W; i++)
	{
		for (uint j = 0; j < H; j++)
		{
			O[i * H + j] = M[j*W + i];
		}
	}
}

void TransformKernel(float* kerenl, float* TransformedKErnel)
{
	float G[] = { 1, 0, 0,
		0.5, 0.5, 0.5,
		0.5, -0.5, 0.5,
		0, 0, 1 };

	float Gt[12];


	float GK[12];

	Transpose(G, Gt, 4, 3);
	mul(G, kerenl, GK, 3, 4, 3, 3);
	mul(GK, Gt, TransformedKErnel, 3, 4, 4, 3);
}

class KernelData
{
public:
	KernelData(std::string name, bool bInline, uint input_sz, uint nIFM, uint nOFM, cl::NDRange localSize)
	{
		m_name = name;
		m_bInline = bInline;
		m_inputSize = input_sz;
		m_nIFM = nIFM;
		m_nOFM = nOFM;
		m_localSize = localSize;
	}

public:
	std::string m_name;
	bool m_bInline;
	uint m_inputSize;
	uint m_nIFM;
	uint m_nOFM;
	cl::NDRange m_localSize;
};

void TransformKernels(float* kernels, float* TransformedKernels, UINT nIFM, UINT nOFM)
{
	float K[9];
	float K_Transformed[16];
	for (uint oFM = 0; oFM < nOFM; oFM++)
	{
		for (uint iFM = 0; iFM < nIFM; iFM++)
		{
			for (uint i = 0; i < 9; i++)
			{
				K[i] = kernels[9 * oFM * nIFM + iFM * 9 + i];
			}

			TransformKernel(K, K_Transformed);

			for (uint i = 0; i < 16; i++)
			{
				TransformedKernels[16 * oFM*nIFM + iFM * 16 + i] = K_Transformed[i];
			}
			
		}
	}
}

void PadKernels(float* kernels, float* PaddedKernels, UINT nIFM, uint nOFM)
{
	float K[9];
	float K_Transformed[16];
	for (uint oFM = 0; oFM < nOFM; oFM++)
	{
		for (uint iFM = 0; iFM < nIFM; iFM++)
		{
			for (uint i = 0; i < 9; i++)
			{
				K[i] = kernels[9 * oFM * nIFM + iFM * 9 + i];
			}

			uint currIndex = 0;
			for (uint i = 0; i < 3; i++)
			{
				for (uint j = 0; j < 3; j++)
				{
					uint kernelIdx = i * 3 + j;
					PaddedKernels[16 * oFM*nIFM + iFM * 16 + currIndex] = K[kernelIdx];
					currIndex++;
				}
				PaddedKernels[16 * oFM*nIFM + iFM * 16 + currIndex] = 0;
				currIndex++;
			}
			PaddedKernels[16 * oFM*nIFM + iFM * 16 + currIndex++] = 0;
			PaddedKernels[16 * oFM*nIFM + iFM * 16 + currIndex++] = 0;
			PaddedKernels[16 * oFM*nIFM + iFM * 16 + currIndex++] = 0;
			PaddedKernels[16 * oFM*nIFM + iFM * 16 + currIndex++] = 0;
			
		}
	}
}

#define MAT_ACC_R(i,j,row) i*row + j
#define MAT_ACC_4(i,j) MAT_ACC_R(i,j,4)
#define MAT_ACC_3(i,j) MAT_ACC_R(i,j,3)
#define MAT_ACC(i,j) MAT_ACC_4(i,j)

void TransformFilter(float* K, float* K_Tag, float* K_Tag1)
{

	
		float G[] = { 1, 0, 0,
			0.5, 0.5, 0.5,
			0.5, -0.5, 0.5,
			0, 0, 1 };

		float Gt[12];


		float GK[12];

		Transpose(G, Gt, 4, 3);
		mul(G, K, GK, 3, 4, 3, 3);
		mul(GK, Gt, K_Tag, 3, 4, 4, 3);
	
		float S[12];
		for (uint j = 0; j < 3; j++)
		{
			S[MAT_ACC_3(0, j)] = K[MAT_ACC_3(0, j)];
			S[MAT_ACC_3(1, j)] = (K[MAT_ACC_3(0, j)] + K[MAT_ACC_3(1, j)] + K[MAT_ACC_3(2, j)])*0.5;
			S[MAT_ACC_3(2, j)] = (K[MAT_ACC_3(0, j)] - K[MAT_ACC_3(1, j)] + K[MAT_ACC_3(2, j)])*0.5;
			S[MAT_ACC_3(3, j)] = K[MAT_ACC_3(2, j)];
		}

		for (uint i = 0; i < 4; i++)
		{
			K_Tag1[MAT_ACC(i, 0)] = S[MAT_ACC_3(i, 0)];
			K_Tag1[MAT_ACC(i, 1)] = (S[MAT_ACC_3(i, 0)] + S[MAT_ACC_3(i, 1)] + S[MAT_ACC_3(i, 2)])*0.5;
			K_Tag1[MAT_ACC(i, 2)] = (S[MAT_ACC_3(i, 0)] - S[MAT_ACC_3(i, 1)] + S[MAT_ACC_3(i, 2)])*0.5;
			K_Tag1[MAT_ACC(i, 3)] = S[MAT_ACC_3(i, 2)];
		}
	
}

void TransformImageTile(float* Tile, float* Tile_Tag, float* Tile_Tag1)
{
	
		float B[] = { 1, 0, -1, 0,
			0, 1, 1, 0,
			0, -1, 1, 0,
			0, 1, 0, -1 };
		float Bt[16];
		float B_Tile[16];
		Transpose(B, Bt, 4, 4);
		mul(B, Tile, B_Tile, 4, 4, 4, 4);
		mul(B_Tile, Bt, Tile_Tag, 4, 4, 4, 4);
	
		float S[16];
		for (uint j = 0; j < 4; j++)
		{
			S[MAT_ACC(0, j)] = Tile[MAT_ACC(0, j)] - Tile[MAT_ACC(2, j)];
			S[MAT_ACC(1, j)] = Tile[MAT_ACC(1, j)] + Tile[MAT_ACC(2, j)];
			S[MAT_ACC(2, j)] = -Tile[MAT_ACC(1, j)] + Tile[MAT_ACC(2, j)];
			S[MAT_ACC(3, j)] = Tile[MAT_ACC(1, j)] - Tile[MAT_ACC(3, j)];
		}

		for (uint i = 0; i < 4; i++)
		{
			Tile_Tag1[MAT_ACC(i, 0)] = S[MAT_ACC(i, 0)] - S[MAT_ACC(i, 2)];
			Tile_Tag1[MAT_ACC(i, 1)] = S[MAT_ACC(i, 1)] + S[MAT_ACC(i, 2)];
			Tile_Tag1[MAT_ACC(i, 2)] = -S[MAT_ACC(i, 1)] + S[MAT_ACC(i, 2)];
			Tile_Tag1[MAT_ACC(i, 3)] = S[MAT_ACC(i, 1)] - S[MAT_ACC(i, 3)];
		}

	

}

void TestTransformTile()
{
	float Tile[16];
	float O[16];
	float ref[16];

	for (uint i = 0; i < 16; i++)
	{
		Tile[i] = float(i);
	}

	TransformImageTile(Tile, ref, O);
	int stop = 0;
}

void TestTransformKernel()
{
	float K[9];
	float O[16];
	float ref[16];

	for (uint i = 0; i < 9; i++)
	{
		K[i] = float(i);
	}

	TransformFilter(K, ref, O);
	int stop = 0;
}


void GenerateFloats(float* data, uint nFloats)
{
	std::default_random_engine generator;
	const UINT max_rnd = 1000000;
	std::uniform_int_distribution<int> distribution(1, max_rnd);
	auto dice = std::bind(distribution, generator);
	
	for (uint i = 0; i < nFloats; i++)
	{
		data[i] = (dice() / float(max_rnd));
	}
}

enum KernelTypes
{
	REGULAR,
	TRANSFORMED,
	PADDED,

	NUM_KERNEL_TYPES
};

int winograd_example(size_t input_sz, size_t output_sz, size_t iFM, size_t oFM, float* Kernels, float* Data, float* bias, float* ref_out, FILE* OutputFile)
{
	if (OutputFile)
	{
		//fprintf_s(OutputFile, "input_sz,output_sz,in_fm,out_fm");
		fprintf_s(OutputFile, "%d,%d,%d,%d\n", input_sz, output_sz, iFM, oFM);
	}

	cl_int err;
	const bool bTransformKernels = true;
	const bool bFastLoad = true;
	//TestTransformTile();
	//TestTransformKernel();
	cl::vector< cl::Platform > platformList;
	cl::Platform::get(&platformList);
	checkErr(platformList.size() != 0 ? CL_SUCCESS : -1, "cl::Platform::get");
	//std::cout << "Platform number is: " << platformList.size() << std::endl;
	for (UINT i = 0; i < platformList.size(); i++)
	{
		std::string platformVendor;
		std::string playformVersion;
		platformList[i].getInfo((cl_platform_info)CL_PLATFORM_VENDOR, &platformVendor);
		platformList[i].getInfo((cl_platform_info)CL_PLATFORM_VERSION, &playformVersion);
		//std::cout << "Platform is by: " << platformVendor << " Version: " << playformVersion << "\n";
		cl::vector < cl::Device > DeviceList;
		platformList[i].getDevices(CL_DEVICE_TYPE_ALL, &DeviceList);
		for (UINT j = 0; j < DeviceList.size(); j++)
		{
			std::string deviceType;
			std::string extensions;
			DeviceList[j].getInfo((cl_device_info)CL_DEVICE_NAME, &deviceType);
			DeviceList[j].getInfo((cl_device_info)CL_DEVICE_EXTENSIONS, &extensions);
			//std::cout << "Device: " << deviceType << "\n" << "Extensions: " << extensions << "\n";
		}
	}
	cl_context_properties cprops[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties)(platformList[PLATFORM])(), 0 };
	cl::Context context(
		DEVICE_TYPE,
		cprops,
		NULL,
		NULL,
		&err);

	checkErr(err, "Context::Context()");

	cl::vector<cl::Device> devices;
	devices = context.getInfo<CL_CONTEXT_DEVICES>();

	cl_mem_flags CreationFlagsOut = 0;
	cl_mem_flags CreationFlagsIn = 0;
	switch (DEVICE_TYPE)
	{
	case CL_DEVICE_TYPE_CPU:
		CreationFlagsOut = CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR;
		CreationFlagsIn = CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR;
		break;
	case CL_DEVICE_TYPE_GPU:
		//CreationFlagsOut = CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR;
		//CreationFlagsIn = CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR | CL_MEM_ALLOC_HOST_PTR;
		CreationFlagsIn = CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR;
		CreationFlagsOut = CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR;
		break;
	}

	//build the output buffer
	//outH is the memory buffer openCL uses
	std::vector<KernelData> clKernels;

	/*clKernels.push_back(KernelData("winograd_conv_2_3_subGroup_n", false, input_sz, iFM, oFM, cl::NDRange(1, 1, (oFM % 16 == 0 && oFM != 0) ? 16 : 1)));
	clKernels.push_back(KernelData("winograd_conv_2_3_subGroup_kernel_fixed_n", false, input_sz, iFM, oFM, cl::NDRange(4, 4, 1)));
	clKernels.push_back(KernelData("winograd_conv_2_3_subGroup", true, input_sz, iFM, oFM, cl::NDRange(1, 1, (oFM % 16 == 0 && oFM != 0) ? 16 : 1)));*/

	if (!ref_out)
	{
		clKernels.push_back(KernelData("winograd_conv_2_3_noInline_ioyx_bfxy", false, input_sz, iFM, oFM, cl::NDRange(1, 1, (oFM % 16 == 0 && oFM != 0) ? 16 : 1)));
		clKernels.push_back(KernelData("winograd_conv_2_3_noInline_ioyx_xyfb", false, input_sz, iFM, oFM, cl::NDRange(1, 1, (oFM % 16 == 0 && oFM != 0) ? 16 : 1)));
		clKernels.push_back(KernelData("winograd_conv_2_3_noInline_oiyx_bfxy", false, input_sz, iFM, oFM, cl::NDRange(1, 1, (oFM % 16 == 0 && oFM != 0) ? 16 : 1)));
		clKernels.push_back(KernelData("winograd_conv_2_3_noInline_oiyx_xyfb", false, input_sz, iFM, oFM, cl::NDRange(1, 1, (oFM % 16 == 0 && oFM != 0) ? 16 : 1)));

		clKernels.push_back(KernelData("winograd_conv_2_3_Inline_ioyx_bfxy", true, input_sz, iFM, oFM, cl::NDRange(1, 1, (oFM % 16 == 0 && oFM != 0) ? 16 : 1)));
		clKernels.push_back(KernelData("winograd_conv_2_3_Inline_ioyx_xyfb", true, input_sz, iFM, oFM, cl::NDRange(1, 1, (oFM % 16 == 0 && oFM != 0) ? 16 : 1)));
		clKernels.push_back(KernelData("winograd_conv_2_3_Inline_oiyx_bfxy", true, input_sz, iFM, oFM, cl::NDRange(1, 1, (oFM % 16 == 0 && oFM != 0) ? 16 : 1)));
		clKernels.push_back(KernelData("winograd_conv_2_3_Inline_oiyx_xyfb", true, input_sz, iFM, oFM, cl::NDRange(1, 1, (oFM % 16 == 0 && oFM != 0) ? 16 : 1)));

		clKernels.push_back(KernelData("winograd_conv_2_3_subGroup_kernel_fixed_n", false, input_sz, iFM, oFM, cl::NDRange(4, 4, 1)));

		clKernels.push_back(KernelData("winograd_conv_2_3_noInline_ioyx_bfxy_opt", false, input_sz, iFM, oFM, cl::NDRange(1, 1, (oFM % 16 == 0 && oFM != 0) ? 16 : 1)));
		clKernels.push_back(KernelData("winograd_conv_2_3_Inline_ioyx_bfxy_opt", false, input_sz, iFM, oFM, cl::NDRange(1, 1, (oFM % 16 == 0 && oFM != 0) ? 16 : 1)));
	}
	else
	{
		clKernels.push_back(KernelData("winograd_conv_2_3_noInline_oiyx_xyfb", false, input_sz, iFM, oFM, cl::NDRange(1, 1, (oFM % 16 == 0 && oFM != 0) ? 16 : 1)));
	}

	

	

	float** outF = new float*[clKernels.size()];
	size_t outputSz = output_sz * output_sz * oFM;
	cl::Buffer* outputBuffer = new cl::Buffer[clKernels.size()];
	for (UINT i = 0; i < clKernels.size(); i++)
	{
		outF[i] = new float[outputSz];
		ZeroMemory(outF[i], sizeof(float)*outputSz);
		outputBuffer[i] = cl::Buffer(
			context,
			CreationFlagsOut,
			outputSz * sizeof(float),
			NULL,
			&err);
	}


	//build InputBuffers


	float* TransformedKernels = new float[iFM * oFM*16];
	float* paddedKernels = new float[iFM*oFM * 16];

	TransformKernels(Kernels, TransformedKernels, iFM, oFM);
	PadKernels(Kernels, paddedKernels, iFM, oFM);

	cl::Buffer FilterBuffers[NUM_KERNEL_TYPES];
	uint KernelSZ[] = { 9, 16, 16 };
	float* kernelBuffer[] = { Kernels, TransformedKernels, paddedKernels };

	for (uint i = 0; i < NUM_KERNEL_TYPES; i++)
	{
		FilterBuffers[i] = cl::Buffer(
			context,
			CreationFlagsIn,
			iFM* oFM*KernelSZ[i]*sizeof(float),
			kernelBuffer[i],
			&err
		);
	}

	size_t ImageSZ = input_sz * input_sz * iFM;
	cl::Buffer imageBuffer(
		context,
		CreationFlagsIn,
		ImageSZ * sizeof(float),
		Data,
		&err
	);

	size_t BiasSZ = oFM;
	cl::Buffer biasBuffer(
		context,
		CreationFlagsIn,
		BiasSZ * sizeof(float),
		bias,
		&err
	);

	checkErr(err, "clBuffer Output");

	std::ifstream file("winograd.cl");
	checkErr(file.is_open() ? CL_SUCCESS : -1, "lesson1_kernel.cl");
	std::string prog(
		std::istreambuf_iterator<char>(file),
		(std::istreambuf_iterator<char>()));

	//build programs require a list of devices to build the program for
	cl::Program program(context, prog);
	std::string buildOptions = generateBuildOptions(bTransformKernels, bFastLoad, iFM, oFM, input_sz);
	//buildOptions += " -s \"path_to_cl_file\"";
	err = program.build(devices, buildOptions.c_str());
	if (err != 0)
	{
		std::string errors;
		program.getBuildInfo(devices[0], CL_PROGRAM_BUILD_LOG, &errors);
		int stop = 0;
	}
	checkErr(err, "Program::build()");
	
	

	//we have dim - 2 / 2 Tiles on specific dim
	cl::NDRange workSize =cl::NDRange(ceil((input_sz) / 2.0), ceil((input_sz) / 2.0), oFM);
	cl::CommandQueue queue(context, devices[0], CL_QUEUE_PROFILING_ENABLE, &err);

	cl_uint frequency;
	cl_uint maxNumComputeUnits;
	err = devices[0].getInfo(CL_DEVICE_MAX_COMPUTE_UNITS, &maxNumComputeUnits);
	err = devices[0].getInfo(CL_DEVICE_MAX_CLOCK_FREQUENCY, &frequency);
	double maxGPUClockFreqInGHz = frequency / 1000.0;
	double peakGPUPerf = maxNumComputeUnits * maxGPUClockFreqInGHz * 16;   // peak GFLOP

	for (UINT i = 0; i < clKernels.size(); i++)
	{
		cl::Kernel kernel = cl::Kernel(program, clKernels[i].m_name.c_str(), &err);
		checkErr(err, "Kernel::Kernel()");
		err = kernel.setArg(0, imageBuffer);
		err = kernel.setArg(1, FilterBuffers[2]);
		err = kernel.setArg(2, outputBuffer[i]);
		err = kernel.setArg(3, biasBuffer);
		if (!clKernels[i].m_bInline)
		{
			err = kernel.setArg(4, (UINT)input_sz);
			err = kernel.setArg(5, (UINT)iFM);
			err = kernel.setArg(6, (UINT)oFM);
		}

		checkErr(err, "Kernel::setArg()");

		//build command queue
		
		checkErr(err, "CommandQueue::CommandQueue()");

		cl::Event event;
		std::chrono::time_point<std::chrono::steady_clock> start, end;
		start = std::chrono::steady_clock::now();
		
		
		err = queue.enqueueNDRangeKernel(
			kernel,
			cl::NullRange,
			workSize,
			clKernels[i].m_localSize,
			NULL,
			&event);
		checkErr(err, "ComamndQueue::enqueueNDRangeKernel()");
		//wait for kernel to finish
		event.wait();
		cl_ulong startCL, endCL;
		event.getProfilingInfo(CL_PROFILING_COMMAND_START, &startCL);
		event.getProfilingInfo(CL_PROFILING_COMMAND_END, &endCL);

		end = std::chrono::steady_clock::now();
		std::chrono::time_point<std::chrono::steady_clock>::duration elapsed_seconds = end - start;
		double timeMS = (double)elapsed_seconds.count() * 1e-6;
		double gflops = (1/*batchSize*/) *
			oFM * iFM *
			3 * 3 * /*KernelSize*/
			output_sz * output_sz * 2.0 / 1000.0 / 1000.0 / timeMS;
		char buff_output[1000];
		sprintf_s(buff_output, "PERF: kernel (%s), %4.5lf, %4.2lf,  %4.2lf, \t#conv_time (ms), gflops, efficiency(%%)\n",
			clKernels[i].m_name.c_str(),
			timeMS, gflops, gflops / peakGPUPerf * 100);
		
		if (OutputFile)
		{
			fprintf_s(OutputFile, "%s,%4.5lf,%4.2lf,%4.2lf\n", clKernels[i].m_name.c_str(), timeMS, gflops, gflops / peakGPUPerf * 100);
		}

		std::string output_string(buff_output);
		std::cerr << "timer Winograd " << output_string << std::endl;
		if (ref_out)
		{
			err = queue.enqueueReadBuffer(
				outputBuffer[i],
				CL_TRUE,
				0,
				outputSz * sizeof(float),
				outF[i]);
		}
	}

	if (ref_out)
	{
		for (size_t i = 0; i < outputSz; i++)
		{
			if (abs(outF[0][i] - ref_out[i]) / abs(ref_out[i]) > 1e-5)
			{
				//std::cerr << "functional issue in winograd" << std::endl;
				int stop = 0;
				stop++;
			}
		}
	}

	int stop = 0;
	stop++;

	//map the output buffer

	return EXIT_SUCCESS;
}

void runWinograd()
{
	unsigned input_sz[] = { 16, 360, 360, 96 };
	unsigned output_sz[] = { 16, 360, 360, 96 };
	unsigned ifm[] = { 192, 32, 32, 32 };
	unsigned ofm[] = { 192, 32, 64, 32 };

	FILE* outputFile = NULL;
	fopen_s(&outputFile, "winograd_res.txt", "w");

	for (uint i = 0; i < 1; i++)
	{
		uint KerenlSizeInFloats = ifm[i] * ofm[i] * 9;
		uint DataSizeInFloats = input_sz[i] * output_sz[i] * ifm[i];
		uint biasSizeInFloats = ofm[i];
		float* Kernels = new float[KerenlSizeInFloats];
		float* data = new float[DataSizeInFloats];
		float* bias = new float[biasSizeInFloats];

		GenerateFloats(Kernels, KerenlSizeInFloats);
		GenerateFloats(data, DataSizeInFloats);
		GenerateFloats(bias, biasSizeInFloats);

		winograd_example(input_sz[i], output_sz[i], ifm[i], ofm[i], Kernels, data, bias, NULL, outputFile);

	}

	system("pause");
}