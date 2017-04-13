/*
// Copyright (c) 2017 Intel Corporation
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
#include "api_impl.h"
#include <immintrin.h>

namespace cldnn{

    CLDNN_API half_impl::half_impl(float value)
    {
#define TO_M128i(a) (*(__m128i*)&(a))
#define TO_M128(a) (*(__m128*)&(a))

        static const uint32_t DWORD_SIGNMASK = 0x80000000;
        static const uint32_t DWORD_MINFP16 = 0x38800000;
        static const uint32_t DWORD_MAXFP16 = 0x477fe000;
        static const uint32_t DWORD_FP16_2_POW_10 = (1 << 10);
        static const uint32_t DWORD_FP16_EXPBIAS_NO_HALF = 0xc8000000;
        static const uint32_t WORD_MAXFP16 = 0x7BFF;

        static const __m128i IVec4SignMask = _mm_set1_epi32(DWORD_SIGNMASK);
        static const __m128i IVec4MinNormalFp16 = _mm_set1_epi32(DWORD_MINFP16);
        static const __m128i IVec4MaxNormalFp16 = _mm_set1_epi32(DWORD_MAXFP16);
        static const __m128i IVec4OnePow10 = _mm_set1_epi32(DWORD_FP16_2_POW_10);
        static const __m128i IVec4ExpBiasFp16 = _mm_set1_epi32(DWORD_FP16_EXPBIAS_NO_HALF);
        static const __m128i IVec4MaxFp16InWords = _mm_set1_epi32(WORD_MAXFP16);

        static const __m128 FVec4MaxNormalFp16 = TO_M128(IVec4MaxNormalFp16);
        static const __m128 FVec4MinNormalFp16 = TO_M128(IVec4MinNormalFp16);
        static const __m128i IVec4InfF32 = _mm_set1_epi32(0x7f800000); //inf in in hex representation
        static const __m128i IVec4InfF16 = _mm_set1_epi32(0x00007c00);

        static const __m128 FVec4MaxFp16InWords = TO_M128(IVec4MaxFp16InWords);

        __m128 Src = _mm_set1_ps(value);

        // Remove the sign bit from the source
        __m128 AbsSrc = _mm_andnot_ps(TO_M128(IVec4SignMask), Src);

        // Create a mask to identify the DWORDs that are smaller than the minimum normalized fp16 number
        __m128 CmpToMinFp16Mask = _mm_cmplt_ps(AbsSrc, FVec4MinNormalFp16);

        // Create a mask to identify the DWORDs that are larger than the maximum normalized fp16 number
        __m128 CmpToMaxFp16Mask = _mm_cmpgt_ps(AbsSrc, FVec4MaxNormalFp16);
        __m128i CmpToInfMask = _mm_cmpeq_epi32(TO_M128i(AbsSrc), IVec4InfF32);
        // Create a mask with the minimum normalized fp16 number in the DWORDs that are smaller than it
        __m128 MaskOfMinFp16 = _mm_and_ps(CmpToMinFp16Mask, FVec4MinNormalFp16);

        __m128i MaskOf2POW10 = _mm_and_si128(TO_M128i(CmpToMinFp16Mask), IVec4OnePow10);
        __m128 ResultPS = _mm_add_ps(AbsSrc, MaskOfMinFp16);
        __m128i Result = TO_M128i(ResultPS);

        // We need to move from a 127 biased domain to a 15 biased domain. This means subtracting 112 from the exponent. We will add '-112'
        // to the exponent but since the exponent is shifted 23 bits to the left we need to shift '-112' 23 bits to the left as well.
        // This gives us 0xC8000000. We are going to shift the mantissa 13 bits to the right (moving from 23 bits mantissa to 10).
        Result = _mm_add_epi32(Result, IVec4ExpBiasFp16);

        // Shift the mantissa to go from 23 bits to 10 bits
        Result = _mm_srli_epi32(Result, 13);

        Result = _mm_sub_epi16(Result, MaskOf2POW10);

        ResultPS = _mm_blendv_ps(TO_M128(Result), FVec4MaxFp16InWords, CmpToMaxFp16Mask);
        Result = TO_M128i(ResultPS);
        //infinity preserving blending
        Result = _mm_blendv_epi8(Result, IVec4InfF16, CmpToInfMask);

        __m128i iPackedResult = _mm_packs_epi32(Result, Result);

        // iSignMask = mask of the sign bits of the source 4 dwords
        __m128i iSignMask = _mm_and_si128(TO_M128i(Src), IVec4SignMask);

        // Pack the sign mask to 4 words
        __m128i iSignInWords = _mm_packs_epi32(iSignMask, iSignMask);

        iPackedResult = _mm_or_si128(iPackedResult, iSignInWords);
        _data = (uint16_t)_mm_extract_epi16(iPackedResult, 0);
    }

}