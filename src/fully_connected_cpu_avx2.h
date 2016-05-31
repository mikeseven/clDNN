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

#include "fully_connected.h"

namespace neural {
	struct fully_connected_forward_cpu_avx2 : is_an_implementation {
		fully_connected_forward_cpu_avx2(fully_connected &arg);
		~fully_connected_forward_cpu_avx2();

		static is_an_implementation *create(fully_connected &arg) { return new fully_connected_forward_cpu_avx2(arg); };
		std::vector<task> work();// { return fully_connected_ptr->work(); };

		std::unique_ptr<is_an_implementation> fully_connected_ptr;
		const fully_connected &outer;
	};

}


