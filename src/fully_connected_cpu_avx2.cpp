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
#include "fully_connected.h"
#include "fully_connected_cpu_avx2.h"
#include "api/neural.h"


namespace neural {


	fully_connected_forward_cpu_avx2::fully_connected_forward_cpu_avx2(fully_connected &arg)
		: is_an_implementation(neural::type_id<fully_connected_forward_cpu_avx2>())
		, outer(arg)
	{
	
	};

	fully_connected_forward_cpu_avx2::~fully_connected_forward_cpu_avx2()
	{

	}


	
namespace {
	struct attach {
		attach() {
			fully_con_implementation_map::instance().insert({ std::make_tuple(engine::cpu, memory::format::xb_f32, memory::format::xb_f32), fully_connected_forward_cpu_avx2::create });
			fully_con_implementation_map::instance().insert({ std::make_tuple(engine::cpu, memory::format::x_f32,  memory::format::x_f32),  fully_connected_forward_cpu_avx2::create });
		}
		~attach() {}
	};

#ifdef __GNUC__
	__attribute__((visibility("default"))) //todo meybe dll_sym?
#elif _MSC_VER
#   pragma section(".nn_init$m", read, write)
#endif
	attach attach_impl;

}

}

