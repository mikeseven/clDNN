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

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "helper_defines.h"

namespace neural
{
    const char helper_defines[] = R"__CC(
    #define DOT_PRODUCT_8( _result, _rowA, colB )    \
    {   \
            _result.s0 = mad( _rowA, intel_sub_group_shuffle( colB, 0 ), _result.s0 );  \
            _result.s1 = mad( _rowA, intel_sub_group_shuffle( colB, 1 ), _result.s1 );  \
            _result.s2 = mad( _rowA, intel_sub_group_shuffle( colB, 2 ), _result.s2 );  \
            _result.s3 = mad( _rowA, intel_sub_group_shuffle( colB, 3 ), _result.s3 );  \
            _result.s4 = mad( _rowA, intel_sub_group_shuffle( colB, 4 ), _result.s4 );  \
            _result.s5 = mad( _rowA, intel_sub_group_shuffle( colB, 5 ), _result.s5 );  \
            _result.s6 = mad( _rowA, intel_sub_group_shuffle( colB, 6 ), _result.s6 );  \
            _result.s7 = mad( _rowA, intel_sub_group_shuffle( colB, 7 ), _result.s7 );  \
    }
    #define ADD_BIAS_8( _result, _biasVal) \
    { \
        _result.s0 += intel_sub_group_shuffle( _biasVal, 0 ); \
        _result.s1 += intel_sub_group_shuffle( _biasVal, 1 ); \
        _result.s2 += intel_sub_group_shuffle( _biasVal, 2 ); \
        _result.s3 += intel_sub_group_shuffle( _biasVal, 3 ); \
        _result.s4 += intel_sub_group_shuffle( _biasVal, 4 ); \
        _result.s5 += intel_sub_group_shuffle( _biasVal, 5 ); \
        _result.s6 += intel_sub_group_shuffle( _biasVal, 6 ); \
        _result.s7 += intel_sub_group_shuffle( _biasVal, 7 ); \
    }
    )__CC";

    const char helper_undefines[] = R"__CC(
    #undef ADD_BIAS_8
    #undef DOT_PRODUCT_8
    )__CC";
}