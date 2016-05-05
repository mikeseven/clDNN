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

#include "api/neural.h"
#include "tests\test_utils\test_utils.h"
void example_lrn_forward() {

    const uint32_t px = 2, py = 2, pb = 1, pf = 7, psize = 3;

    using namespace neural;
    auto input = memory::create({ engine::reference, memory::format::yxfb_f32,{ pb,{ px, py }, pf }, true });
    auto output = memory::create({ engine::reference, memory::format::yxfb_f32,{ pb,{ px, py }, pf }, true});
    auto output_oracle = memory::create({ engine::reference, memory::format::yxfb_f32,{ pb,{ px, py }, pf }, true });

    std::initializer_list<float> input_oracle_init = {
         -1.0f, -0.5f,  0.0f,  0.5f,  1.0f,  1.5f,  2.0f,    // b=0, x=0, y=0
         -2.0f, -1.7f, -1.2f, -0.7f, -0.2f,  0.3f,  0.8f,    // b=0, x=1, y=0
          0.1f,  0.4f,  0.9f,  1.4f,  1.9f,  2.4f,  2.9f,    // b=0, x=0, y=1
        -10.0f, -8.0f, -7.5f, -7.0f, -6.5f, -6.0f, -5.5f };  // b=0, x=1, y=1

    std::initializer_list<float> output_oracle_init = {
        -0.54433f, -0.27217f,  0.00000f,  0.27217f,  0.32366f,  0.30814f,  0.45266f,    // b=0, x=0, y=0
        -0.42484f, -0.31845f, -0.32025f, -0.30941f, -0.13928f,  0.19550f,  0.53034f,    // b=0, x=1, y=0
         0.08889f,  0.23964f,  0.32244f,  0.31267f,  0.28876f,  0.26604f,  0.37728f,    // b=0, x=0, y=1
        -0.21721f, -0.13945f, -0.15913f, -0.16455f, -0.17056f, -0.17725f, -0.23420f };  // b=0, x=1, y=1

    // input.as<const memory&>().fill<float>();
    tests::set_values(input, input_oracle_init);
    tests::set_values(output_oracle, output_oracle_init);

    float    pk = 1.f;
    float    palpha = 1.0f; //0.00002f;
    float    pbeta = 0.75f;

    auto lrn = normalization::response::create({ engine::reference, output, input, psize, padding::zero, pk, palpha, pbeta});

    execute({ lrn });
}
