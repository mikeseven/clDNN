//performa A*B=C
//C dimensions are HaxWb
//all matrix are row major

#define OPTIMIZED_TRANSFORM 1


#ifndef OPTIONS
#define TRANSFORMED_KERNELS 0
#define N_IFM 1
#define N_OFM 1
#define IMAGE_W 4
#define FAST_LOAD 1
#endif

typedef enum MemoryType_t
{
	MEMORY_IOYX,
	MEMORY_OIYX,
	MEMORY_BFYX,
	MEMORY_YXFB,
}MemoryType;

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


void mulElemetWise(float* A, float* B, float* C, uint Ha, uint Wa)
{
	for (uint i = 0; i < Ha; i++)
	{
		for (uint j = 0; j < Wa; j++)
		{
			C[i*Wa + j] = A[i*Wa + j] * B[i*Wa + j];
		}
	}
}

void add(float* A, float* B, float* C, uint Ha, uint Wa)
{
	for (uint i = 0; i < Ha; i++)
	{
		for (uint j = 0; j < Wa; j++)
		{
			C[i*Wa + j] = A[i*Wa + j] + B[i*Wa + j];
		}
	}
}

//perform 4x4 mad O = A*B+C element wise
void madElementWise(float* A, float* B, float* C, float* O)
{ 
	for (uint i = 0; i < 16; i++)
	{ 
		O[i] = A[i] * B[i] + C[i];
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

#define MAT_ACC_R(i,j,row) i*row + j
#define MAT_ACC_4(i,j) MAT_ACC_R(i,j,4)
#define MAT_ACC_3(i,j) MAT_ACC_R(i,j,3)
#define MAT_ACC(i,j) MAT_ACC_4(i,j)

void TransformFilter(float* K, float* K_Tag)
{

	if (!OPTIMIZED_TRANSFORM)
	{
		//assuming K is a 3x3 matrix
		float G[] = { 1, 0, 0,
			0.5, 0.5, 0.5,
			0.5, -0.5, 0.5,
			0, 0, 1 };

		float Gt[12];


		float GK[12];

		Transpose(G, Gt, 4, 3);
		mul(G, K, GK, 3, 4, 3, 3);
		mul(GK, Gt, K_Tag, 3, 4, 4, 3);
	}
	else
	{ 
		//assuming K is a 4x4 
		float S[12];
		for (uint j = 0; j < 3; j++)
		{
			S[MAT_ACC_3(0, j)] = K[MAT_ACC_R(0, j, 4)];
			S[MAT_ACC_3(1, j)] = (K[MAT_ACC_R(0, j, 4)] + K[MAT_ACC_R(1, j, 4)] + K[MAT_ACC_R(2, j, 4)])*0.5;
			S[MAT_ACC_3(2, j)] = (K[MAT_ACC_R(0, j, 4)] - K[MAT_ACC_R(1, j, 4)] + K[MAT_ACC_R(2, j, 4)])*0.5;
			S[MAT_ACC_3(3, j)] = K[MAT_ACC_R(2, j, 4)];
		}

		for (uint i = 0; i < 4; i++)
		{
			K_Tag[MAT_ACC(i, 0)] = S[MAT_ACC_3(i, 0)];
			K_Tag[MAT_ACC(i, 1)] = (S[MAT_ACC_3(i, 0)] + S[MAT_ACC_3(i, 1)] + S[MAT_ACC_3(i, 2)])*0.5;
			K_Tag[MAT_ACC(i, 2)] = (S[MAT_ACC_3(i, 0)] - S[MAT_ACC_3(i, 1)] + S[MAT_ACC_3(i, 2)])*0.5;
			K_Tag[MAT_ACC(i, 3)] = S[MAT_ACC_3(i, 2)];
		}
	}
}

void TransformImageTile(float* Tile, float* Tile_Tag)
{

	if (!OPTIMIZED_TRANSFORM)
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
	}
	else
	{ 
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
			Tile_Tag[MAT_ACC(i, 0)] = S[MAT_ACC(i, 0)] - S[MAT_ACC(i, 2)];
			Tile_Tag[MAT_ACC(i, 1)] = S[MAT_ACC(i, 1)] + S[MAT_ACC(i, 2)];
			Tile_Tag[MAT_ACC(i, 2)] = -S[MAT_ACC(i, 1)] + S[MAT_ACC(i, 2)];
			Tile_Tag[MAT_ACC(i, 3)] = S[MAT_ACC(i, 1)] - S[MAT_ACC(i, 3)];
		}

	}

}

void TransformMatrix(float* M, float* oM)
{
	float A[] = { 1, 1, 1, 0,
		0, 1, -1, -1 };

	float At[8];
	float A_M[8];

	Transpose(A, At, 2, 4);
	mul(A, M, A_M, 4, 2, 4, 4);
	mul(A_M, At, oM, 4, 2, 2, 4);

}

void LoadKernel_oiyx(const __global float* Kernel, float* K, uint iFM, uint nIFM, uint oFM)
{
	if (!FAST_LOAD)
	{
		//kernel will be linear row_major
		for (uint i = 0; i < 9; i++)
		{
			K[i] = Kernel[9 * oFM * nIFM + iFM * 9 + i];
		}
	}
	else 
	{ 
		//assuming padded kernel
		uint base = 16 * oFM * nIFM + iFM * 16;
		float16* pK = (float16*)K;
		const __global float16* pKernel = (const __global float16*)(Kernel + base);
		*pK = *pKernel;
	}
}

void LoadKernel_ioyx(const __global float* Kernel, float* K, uint iFM, uint nOFM, uint oFM)
{
	if (!FAST_LOAD)
	{
		//kernel will be linear row_major
		for (uint i = 0; i < 9; i++)
		{
			K[i] = Kernel[9 * iFM * nOFM + oFM * 9 + i];
		}
	}
	else if(0)
	{ 
		//assuming swizzled kernel were layout is x^[ofm]_[spatial]
		//x^0_0...x^0_9..x^16_0...x^16_9 so well have reduced load and nicely local layout
		uint iFMStart = 9 * iFM * nOFM;
		uint oFMStart = (oFM / 16) * 9 * 16; // this could be group_id_2
		uint base = iFMStart + oFMStart;
		for (uint k = 0; k < 9; k++) {
			float i0;
			i0 = as_float(intel_sub_group_block_read((const __global uint*)(Kernel + base + k * 16)));
			K[k] = intel_sub_group_shuffle(i0, get_sub_group_local_id());
		}
	}
	else
	{ 
		uint base = 9 * iFM * nOFM + oFM * 9;
		float16* pK = (float16*)K;
		const __global float16* pKernel = (const __global float16*)(Kernel + base);
		*pK = *pKernel;
	}
}

void LoadKernel_oiyx_Transformed(const __global float* Kernel, float* K, uint iFM, uint nIFM, uint oFM)
{ 
	if (!FAST_LOAD)
	{
		for (uint i = 0; i < 16; i++)
		{
			K[i] = Kernel[16 * oFM * nIFM + iFM * 16 + i];
		}
	}
	else
	{ 
		uint base = 16 * oFM * nIFM + iFM * 16;
		float16* pK = (float16*)K;
		const __global float16* pKernel = (const __global float16*)(Kernel + base);
		*pK = *pKernel;
	}

}

void LoadTile_yxfb(const __global float* Image, float* Tile, uint Wi, uint iFM, uint nFM)
{
	uint TopLeftX = get_global_id(0);
	uint TopLeftY = get_global_id(1);
	const uint strideX = nFM;
	const uint strideY = Wi * nFM;
	//assuming row major
	uint TopLeftIndex = TopLeftY * strideY * 2 + TopLeftX * 2 * strideX + iFM;
	for (uint y = 0; y < 4; y++)
	{
		for (uint x = 0; x < 4; x++)
		{
			bool bBorder = ((TopLeftY * 2 + y) >= Wi) || ((TopLeftX * 2 + x) >= Wi);
			uint index = TopLeftIndex + y * strideY + x * strideX;
			Tile[y * 4 + x] = !bBorder ? Image[index] : 0;
		}
	}
}

void LoadTile_bfyx(const __global float* Image, float* Tile, uint Wi, uint iFM, uint nFM)
{
	if (!FAST_LOAD)
	{
		uint TopLeftX = get_global_id(0);
		uint TopLeftY = get_global_id(1);
		const uint strideX = 1;
		const uint strideY = Wi;
		const uint strideZ = Wi*Wi;
		//assuming row major
		uint TopLeftIndex = TopLeftY * strideY * 2 + TopLeftX * 2 * strideX + iFM * strideZ;
		for (uint y = 0; y < 4; y++)
		{
			for (uint x = 0; x < 4; x++)
			{
				bool bBorder = ((TopLeftY * 2 + y) >= Wi) || ((TopLeftX * 2 + x) >= Wi);
				uint index = TopLeftIndex + y * strideY + x * strideX;
				Tile[y * 4 + x] = !bBorder ? Image[index] : 0;
			}
		}
	}
	else if(1)
	{ 
		uint TopLeftX = get_global_id(0);
		uint TopLeftY = get_global_id(1);
		const uint strideX = 1;
		const uint strideY = Wi;
		const uint strideZ = Wi*Wi;
		uint TopLeftIndex = TopLeftY * strideY * 2 + TopLeftX * 2 * strideX + iFM * strideZ;
		//assuming row major
		float i0;
		i0 = as_float(intel_sub_group_block_read((const __global uint*)(Image + TopLeftIndex)));
		for (uint i = 0; i < 16; i++)
			Tile[i] = intel_sub_group_shuffle(i0, i);
	}
	else
	{ 
		uint TopLeftX = get_global_id(0);
		uint TopLeftY = get_global_id(1);
		const uint strideX = 1;
		const uint strideY = Wi;
		const uint strideZ = Wi*Wi;
		uint TopLeftIndex = TopLeftY * strideY * 2 + TopLeftX * 2 * strideX + iFM * strideZ;
		float16* pB = (float16*)Tile;
		const __global float16* pImage = (const __global float16*)(Image + TopLeftIndex);
		*pB = *pImage;
	}
}

void WriteToOutput_yxfb(float* O, __global float* Output, uint OutputW, uint oFM, uint nFM, float bias)
{
	uint TopLeftX = get_global_id(0);
	uint TopLeftY = get_global_id(1);
	const uint strideX = nFM;
	const uint strideY = OutputW * nFM;
	const uint strideZ = 1;
	for (uint y = 0; y < 2; y++)
	{
		for (uint x = 0; x < 2; x++)
		{
			uint X = (TopLeftX * 2 + x) ;
			uint Y = (TopLeftY * 2 + y);
			uint Z = oFM;
			if (X < OutputW && Y < OutputW)
			{
				uint index = Z * strideZ + Y * strideY + X * strideX;
				Output[index] = O[y * 2 + x] + bias;
				
			}

		}
	}
}

float LoadBias(__global float* bias, uint oFM)
{
	return bias[oFM];
}


//assuming our data layout here is
// (X_0^0,Y_0^0)_0 (X_1^0, y_1^0)_0 ... (X_0^1,Y_0^1)_0 .... (X_0^0,Y_0^0)_1 i.e first the feature map 0 of iamge 0 then feature map 1 of image 0 then image 1

void winograd_conv_2_3_internal_optimized(const __global float* Image, __global float* Kernel, __global float* Output, __global float* Biases, uint ImageW, uint nIFM, uint nOFM, MemoryType kernelLayout,
	MemoryType ImageLayout, bool TransformedKernels)
{
	float O[4];
	float result[16] = { 0 };
	float bias;

	//some nice constants to have
	uint global_idx = get_global_id(2)* get_global_size(1) * get_global_size(0) + get_global_id(1) * get_global_size(0) + get_global_id(0);
	uint local_idx = get_local_id(2)* get_local_size(1) * get_local_size(0) + get_local_id(1) * get_local_size(0) + get_local_id(0);
	uint group_idx = get_group_id(2)* get_num_groups(1) * get_num_groups(0) + get_group_id(1) * get_num_groups(0) + get_group_id(0);
	uint X = get_global_id(0);
	uint Y = get_global_id(1);
	uint Z = get_global_id(2);

	
	bias = LoadBias(Biases, Z);

	for (uint iFM = 0; iFM < nIFM; iFM++)
	{
		float K_Tag[16];
		float K[9];
		float B[16];
		
		LoadTile_bfyx(Image, B, ImageW, iFM, nIFM);
		if (TransformedKernels)
		{
			LoadKernel_ioyx(Kernel, K_Tag, iFM, nOFM, Z);
			TransformFilter(K_Tag, K_Tag);
		}
		else
		{ 
			LoadKernel_oiyx_Transformed(Kernel, K_Tag, iFM, nIFM, Z);
		}

		TransformImageTile(B, B);

		for (uint i = 0; i < 16; i++)
		{
			B[i] = sub_group_broadcast(B[i], 0);
			
		}

		madElementWise(K_Tag, B, result, result);
	}
	
	TransformMatrix(result, O);

	WriteToOutput_yxfb(O, Output, ImageW, Z, nOFM, bias);
}

__kernel
void winograd_conv_2_3_subGroup_kernel_fixed_n(const __global float* Image, __global float* Kernel, __global float* Output, __global float* Biases, uint ImageW, uint nIFM, uint nOFM)
{
	//our data 
	//todo rename
	float B_Tag[16];
	float B[16];
	float K_Tag[16];
	float K[9];
	float K_Tag_B_Tag[16];
	float O[4];
	float result[16] = { 0 };
	float bias = 0;

	//some nice constants to have
	uint global_idx = get_global_id(2)* get_global_size(1) * get_global_size(0) + get_global_id(1) * get_global_size(0) + get_global_id(0);
	uint local_idx = get_local_id(2)* get_local_size(1) * get_local_size(0) + get_local_id(1) * get_local_size(0) + get_local_id(0);
	uint group_idx = get_group_id(2)* get_num_groups(1) * get_num_groups(0) + get_group_id(1) * get_num_groups(0) + get_group_id(0);
	uint X = get_global_id(0);
	uint Y = get_global_id(1);
	uint Z = get_global_id(2);

	bias = LoadBias(Biases, Z);
	
	for (uint iFM = 0; iFM < nIFM; iFM++)
	{
		if (local_idx == 0)
		{
			if (TRANSFORMED_KERNELS)
			{
				LoadKernel_oiyx_Transformed(Kernel, K_Tag, iFM, nIFM, Z);
			}
			else
			{
				LoadKernel_ioyx(Kernel, K, iFM, nIFM, Z);
				
				TransformFilter(K, K_Tag);
			}	
		}
		
		for (uint i = 0; i < 16; i++)
		{
			K_Tag[i] = sub_group_broadcast(K_Tag[i], 0);

		}

		LoadTile_yxfb(Image, B, ImageW, iFM, nIFM);

		TransformImageTile(B, B_Tag);




		/*if (global_idx == 0 && iFM == 1)
		{
		for (uint i = 0; i < 16; i++)
		{
		Output[i] = K[i];
		}

		}*/
		mulElemetWise(K_Tag, B_Tag, K_Tag_B_Tag, 4, 4);

		add(result, K_Tag_B_Tag, result, 4, 4);


	}

	TransformMatrix(result, O);

	WriteToOutput_yxfb(O, Output, ImageW, Z, nOFM, bias);
}



void winograd_conv_2_3_internal(const __global float* Image, __global float* Kernel, __global float* Output, __global float* Biases, uint ImageW, uint nIFM, uint nOFM, MemoryType kernelLayout,
	MemoryType ImageLayout, bool TransformedKernels)
{ 
	//our data 
	//todo rename
	float B_Tag[16];
	float B[16];
	float K_Tag[16];
	float K[9] = { 0 };
	float K_Tag_B_Tag[16];
	float O[4];
	float result[16] = { 0 };
	float bias = 0;

	//some nice constants to have
	uint global_idx = get_global_id(2)* get_global_size(1) * get_global_size(0) + get_global_id(1) * get_global_size(0) + get_global_id(0);
	uint local_idx = get_local_id(2)* get_local_size(1) * get_local_size(0) + get_local_id(1) * get_local_size(0) + get_local_id(0);
	uint group_idx = get_group_id(2)* get_num_groups(1) * get_num_groups(0) + get_group_id(1) * get_num_groups(0) + get_group_id(0);
	uint X = get_global_id(0);
	uint Y = get_global_id(1);
	uint Z = get_global_id(2);

	if (local_idx == 0)
	{
		bias = LoadBias(Biases, Z);
	}

	bias = sub_group_broadcast(bias, 0);

	for (uint iFM = 0; iFM < nIFM; iFM++)
	{
		if (local_idx == 0)
		{
			switch (ImageLayout)
			{
			case MEMORY_YXFB:
				LoadTile_yxfb(Image, B, ImageW, iFM, nIFM);
				break;
			case MEMORY_BFYX:
				LoadTile_bfyx(Image, B, ImageW, iFM, nIFM);
				break;
			}
			
			TransformImageTile(B, B_Tag);

		}


		for (uint i = 0; i < 16; i++)
		{
			B_Tag[i] = sub_group_broadcast(B_Tag[i], 0);

		}


		if (TransformedKernels)
		{
			LoadKernel_oiyx_Transformed(Kernel, K_Tag, iFM, nIFM, Z);
		}
		else
		{
			switch (kernelLayout)
			{ 
			case MEMORY_IOYX:
				LoadKernel_ioyx(Kernel, K, iFM, nOFM, Z);
				break;
			case MEMORY_OIYX:
				LoadKernel_oiyx(Kernel, K_Tag, iFM, nIFM, Z);
				break;
			}
			
			TransformFilter(K_Tag, K_Tag);
		}

	
		mulElemetWise(K_Tag, B_Tag, K_Tag_B_Tag, 4, 4);
		add(result, K_Tag_B_Tag, result, 4, 4);

	}

	TransformMatrix(result, O);

	WriteToOutput_yxfb(O, Output, ImageW, Z, nOFM, bias);
}

__kernel
void winograd_conv_2_3_noInline_ioyx_bfxy(const __global float* Image, __global float* Kernel, __global float* Output, __global float* Biases, uint ImageW, uint nIFM, uint nOFM)
{
	winograd_conv_2_3_internal(Image, Kernel, Output, Biases, ImageW, nIFM, nOFM, MEMORY_IOYX, MEMORY_BFYX, TRANSFORMED_KERNELS);
}

__kernel
void winograd_conv_2_3_noInline_ioyx_xyfb(const __global float* Image, __global float* Kernel, __global float* Output, __global float* Biases, uint ImageW, uint nIFM, uint nOFM)
{ 
	winograd_conv_2_3_internal(Image, Kernel, Output, Biases, ImageW, nIFM, nOFM, MEMORY_IOYX, MEMORY_YXFB, TRANSFORMED_KERNELS);
}

__kernel
void winograd_conv_2_3_noInline_oiyx_bfxy(const __global float* Image, __global float* Kernel, __global float* Output, __global float* Biases, uint ImageW, uint nIFM, uint nOFM)
{ 
	winograd_conv_2_3_internal(Image, Kernel, Output, Biases, ImageW, nIFM, nOFM, MEMORY_OIYX, MEMORY_BFYX, TRANSFORMED_KERNELS);
}

__kernel
void winograd_conv_2_3_noInline_oiyx_xyfb(const __global float* Image, __global float* Kernel, __global float* Output, __global float* Biases, uint ImageW, uint nIFM, uint nOFM)
{ 
	winograd_conv_2_3_internal(Image, Kernel, Output, Biases, ImageW, nIFM, nOFM, MEMORY_OIYX, MEMORY_YXFB, TRANSFORMED_KERNELS);
}

__kernel
void winograd_conv_2_3_Inline_ioyx_bfxy(const __global float* Image, __global float* Kernel, __global float* Output, __global float* Biases)
{
	winograd_conv_2_3_internal(Image, Kernel, Output, Biases, IMAGE_W, N_IFM, N_OFM, MEMORY_IOYX, MEMORY_BFYX, TRANSFORMED_KERNELS);
}

__kernel
void winograd_conv_2_3_Inline_ioyx_xyfb(const __global float* Image, __global float* Kernel, __global float* Output, __global float* Biases)
{
	winograd_conv_2_3_internal(Image, Kernel, Output, Biases, IMAGE_W, N_IFM, N_OFM, MEMORY_IOYX, MEMORY_YXFB, TRANSFORMED_KERNELS);
}

__kernel
void winograd_conv_2_3_Inline_oiyx_bfxy(const __global float* Image, __global float* Kernel, __global float* Output, __global float* Biases)
{
	winograd_conv_2_3_internal(Image, Kernel, Output, Biases, IMAGE_W, N_IFM, N_OFM, MEMORY_OIYX, MEMORY_BFYX, TRANSFORMED_KERNELS);
}

__kernel
void winograd_conv_2_3_Inline_oiyx_xyfb(const __global float* Image, __global float* Kernel, __global float* Output, __global float* Biases)
{
	winograd_conv_2_3_internal(Image, Kernel, Output, Biases, IMAGE_W, N_IFM, N_OFM, MEMORY_OIYX, MEMORY_YXFB, TRANSFORMED_KERNELS);
}

__kernel
void winograd_conv_2_3_noInline_ioyx_bfxy_opt(const __global float* Image, __global float* Kernel, __global float* Output, __global float* Biases, uint ImageW, uint nIFM, uint nOFM)
{

	winograd_conv_2_3_internal_optimized(Image, Kernel, Output, Biases, ImageW, nIFM, nOFM, MEMORY_IOYX, MEMORY_BFYX, TRANSFORMED_KERNELS);
}

__kernel
void winograd_conv_2_3_Inline_ioyx_bfxy_opt(const __global float* Image, __global float* Kernel, __global float* Output, __global float* Biases, uint ImageW, uint nIFM, uint nOFM)
{
	winograd_conv_2_3_internal_optimized(Image, Kernel, Output, Biases, IMAGE_W, N_IFM, N_OFM, MEMORY_IOYX, MEMORY_BFYX, TRANSFORMED_KERNELS);
}


__kernel
void LoadTileSubGroup(__global float* input, __global float* o)
{
	float B[16];
	float B_Tag[16];
	uint local_idx = get_local_id(2)* get_local_size(1) * get_local_size(0) + get_local_id(1) * get_local_size(0) + get_local_id(0);
	uint base = get_global_id(2) * 16;
	if (get_sub_group_local_id() == 0 && false)
	{
		
		for (uint i = 0; i < 16; i++)
		{ 
			if (0) {
				float16* Bt = (float16*)B;
				__global float16* inputt = (__global float16*)(input + base);
				Bt[0] = inputt[0];
			}
			
		}

	}
	else {
		float i0;
		i0 = as_float(intel_sub_group_block_read((const __global uint*)(input + base)));
		for (uint i = 0; i < 16; i++)
			B[i] = intel_sub_group_shuffle(i0, i);
	}

	for (uint i = 0; i < 16; i++)
	{
		B[i] = sub_group_broadcast(B[i], 0);

	}

	for (uint i = 0; i < 16; i++)
	{
		o[i] = B[i] + get_local_id(0);

	}
}

__kernel
void LoadKernelSubGroup(__global float* input, __global float* o)
{
	float B[16];
	float B_Tag[16];
	uint local_idx = get_local_id(2)* get_local_size(1) * get_local_size(0) + get_local_id(1) * get_local_size(0) + get_local_id(0);
	uint base = get_group_id(2) * 16 * 9 ;
	if (get_sub_group_local_id() == 0 && false)
	{

		for (uint i = 0; i < 16; i++)
		{
			if (0) {
				float16* Bt = (float16*)B;
				__global float16* inputt = (__global float16*)(input + base);
				Bt[0] = inputt[0];
			}

		}

	}
	else {
		for (uint k = 0; k < 9; k++) {
			float i0;
			i0 = as_float(intel_sub_group_block_read((const __global uint*)(input + base + k * 16)));
			B[k] = intel_sub_group_shuffle(i0, get_sub_group_local_id());
		}
	}

	for (uint i = 0; i < 16; i++)
	{
		B[i] = sub_group_broadcast(B[i], 0);

	}

	for (uint i = 0; i < 16; i++)
	{
		o[i] = B[i] + get_local_id(0);

	}
}