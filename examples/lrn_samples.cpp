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
#include "memory_utils.h"

void example_lrn_forward() {

    using namespace neural;
    auto input = memory::create({ engine::cpu, memory::format::yxfb_f32,{ 8, {10, 10}, 3 }, true });
    auto output = memory::create({ engine::cpu, memory::format::yxfb_f32,{ 8, {10, 10}, 3 }, true});
    fill<float>(input.as<const memory&>());
    float pk = 1.f;
    uint32_t psize = 5;
    float palpha = 0.00002f;
    float pbeta = 0.75f;
    auto lrn = normalization::response::create({ engine::cpu, output, input, psize, padding::zero, pk, palpha, pbeta});

    execute({ lrn });
}
