///*
//// Copyright (c) 2016 Intel Corporation
////
//// Licensed under the Apache License, Version 2.0 (the "License");
//// you may not use this file except in compliance with the License.
//// You may obtain a copy of the License at
////
////      http://www.apache.org/licenses/LICENSE-2.0
////
//// Unless required by applicable law or agreed to in writing, software
//// distributed under the License is distributed on an "AS IS" BASIS,
//// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//// See the License for the specific language governing permissions and
//// limitations under the License.
//*/
//
/////////////////////////////////////////////////////////////////////////////////////////////////////
//#pragma once
//#include <cstdint>
//#include "cldnn_defs.h"
//#include "compounds.h"
//#include "memory.hpp"
//#include "primitive.hpp"
//
//namespace cldnn
//{
//
//
//
//struct depth_concat_desc : primitive_desc_base<primitive_types::depth_concat>
//{
//
//    depth_concat_desc(array_ref<primitive_id> inputs)
//        :primitive_desc_base(inputs)
//    {}
//};
//
//template<> struct primitive_type_traits<primitive_types::reorder>        { typedef reorder_desc primitive_type;         };
//template<> struct primitive_type_traits<primitive_types::mean_substract> { typedef mean_substract_desc primitive_type;  };
//template<> struct primitive_type_traits<primitive_types::activation>     { typedef activation_desc primitive_type;      };
//template<> struct primitive_type_traits<primitive_types::convolution>    { typedef convolution_desc primitive_type;     };
//template<> struct primitive_type_traits<primitive_types::fully_connected>{ typedef fully_connected_desc primitive_type; };
//template<> struct primitive_type_traits<primitive_types::pooling>        { typedef pooling_desc primitive_type;         };
//template<> struct primitive_type_traits<primitive_types::normalization>  { typedef normalization_desc primitive_type;   };
//template<> struct primitive_type_traits<primitive_types::softmax>        { typedef softmax_desc primitive_type;         };
//template<> struct primitive_type_traits<primitive_types::depth_concat>   { typedef depth_concat_desc primitive_type;    };
//
//}
