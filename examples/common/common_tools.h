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

#include "api/neural.h"

#include <string>
#include <vector>


std::vector<std::string> get_directory_images(const std::string &images_path);
std::vector<std::string> get_directory_weights(const std::string &images_path);
void load_images_from_file_list(const std::vector<std::string> &images_list, neural::primitive &memory); 
