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

#include "neural.h"

// TODO!!! - optimize weights based on HW
class Weights_optimizer
{
private:
    bool enabled;
    std::vector<neural::primitive> primitives;
    bool needs_optimization(const neural::primitive& prim, neural::file::weights_type type);
public:
    Weights_optimizer(bool enabled);
    neural::primitive create_weights_from_file(const std::string& path, neural::file::weights_type type);
    void optimize(const neural::worker& worker);
};