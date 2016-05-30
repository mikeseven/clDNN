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
#include <map>

namespace neural {

type_traits* typeid_register(size_t size, bool is_float, const std::string& str){
    
    static std::map<std::string, std::shared_ptr<type_traits>> register_map;

    auto it = register_map.find(str);
    if( register_map.end() != it )
        return it->second.get();

    std::shared_ptr<type_traits> tt_ptr = std::make_shared<type_traits>(0, size, is_float, str.c_str());
    *const_cast<size_t *>(&tt_ptr->id) = reinterpret_cast<size_t>(tt_ptr.get());

    register_map.emplace(str, tt_ptr);

    return tt_ptr.get();
}

}