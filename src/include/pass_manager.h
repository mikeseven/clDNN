/*
// Copyright (c) 2018 Intel Corporation
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


#include "program_impl.h"
#include "layout_optimizer.h"

namespace cldnn
{
    class base_pass
    {
    public:
        virtual void run(program_impl &p) = 0;
    };

    class trim_to_outputs : base_pass
    {
    public:
        virtual void run(program_impl &p) override;
    };

    class propagate_constants : base_pass
    {
    public:
        virtual void run(program_impl &p) override;
    };

    class remove_redundant_reorders : base_pass
    {
    public:
        virtual void run(program_impl &p) override;
    };

    class reorder_inputs : base_pass
    {
    public:
        reorder_inputs(layout_optimizer& lo_ref);
        virtual void run(program_impl &p) override;
        virtual void run(program_impl &p, layout_optimizer& lo);
    private:
        layout_optimizer& _lo;
    };

    class post_optimize_weights : base_pass
    {
    public:
        post_optimize_weights(layout_optimizer& lo_ref);
        virtual void run(program_impl &p) override;
        virtual void run(program_impl &p, layout_optimizer& lo);
    private:
        layout_optimizer& _lo;
    };
}