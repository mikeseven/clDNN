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

/*! @page tutorial clDNN Tutorial  
* @section intro Introduction
*  This section contains chapter of tutorial demonstrating how to work with clDNN. If you are new in clDNN, we recommend to start with
*  "clDNN documentation" that describes API. We assume that user is familiar with C++ or C, Deep learining terminology i.e. primitive kinds.
*  
* @subpage c1 
* @subpage c2
* @subpage c3
* @subpage c4
* @subpage c5
*
*/
#include <api\CPP\engine.hpp>
#include <api\CPP\topology.hpp>


cldnn::engine   chapter_1(); // Engine, layout, tensor, memory, data and input
cldnn::topology chapter_2(cldnn::engine&); // Primitives and topology
void            chapter_3(cldnn::engine&, cldnn::topology&); // Network and execution
void            chapter_4(cldnn::engine&, cldnn::topology&); // Hidden layers access
void            chapter_5(cldnn::engine&, cldnn::topology&); // Other building options

int main(int argc, char* argv[])
{
    cldnn::engine eng = chapter_1();
    cldnn::topology topology = chapter_2(eng);
    chapter_3(eng, topology);
    chapter_4(eng, topology);
    chapter_5(eng, topology);
    return 0;
}
