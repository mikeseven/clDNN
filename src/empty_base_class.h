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

namespace neural{
    // This class is used for inheritance when common parent class is needed. It is used e.g. for convolution jit implementation where
    // class is locally defined in .cpp file but I want to keep pointer indicating on object of this class. Pointer is declared is .h file.
    // It doesn't know about class existing in .cpp file. (Smart) Pointer is used only to keep object alive untill end of program, not to access
    // any fields/methods.
    struct empty_base_class {
        empty_base_class(){};
        virtual ~empty_base_class(){};
    };
}