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

    class prepare_buffer_fusing : base_pass
    {
    public:
        virtual void run(program_impl &p) override;    
    };

    class eltwise_shrinking : base_pass
    {
    public:
        virtual void run(program_impl &p) override;
    };

    class eltwise_remove_stride : base_pass
    {
    public:
        virtual void run(program_impl &p) override;
    private:    
        void conv_stride_extend(program_impl & p, program_node & node, cldnn::tensor & tensor);
    };

    class prepare_primitive_fusing : base_pass
    {
    public:
        virtual void run(program_impl &p) override;
    private:
        void fuse_skip_layers(program_impl &p, program_node* node);
        void fuse_conv_bn_scale(program_impl &p, program_node* node);
    };

    class prepare_depthwise_sep_opt : base_pass
    {
    public:
        virtual void run(program_impl &p) override;
    };

    class prep_opt_depthwise_sep_post : base_pass
    {
    public:
        virtual void run(program_impl &p) override;
        template <typename T>
        void optimize_depthwise_sep_pre(T& node);
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

    class pre_optimize_bias : base_pass
    {
    public:
        pre_optimize_bias(layout_optimizer& lo_ref);
        virtual void run(program_impl &p) override;
        virtual void run(program_impl &p, layout_optimizer& lo);
        template <typename T>
        void optimize_bias(T& node, layout_optimizer& lo, program_impl& p);
    private:
        layout_optimizer& _lo;
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
        template <typename T>
        void optimize_weights(T& node, layout_optimizer& lo, program_impl &p);
        virtual void run(program_impl &p) override;
        virtual void run(program_impl &p, layout_optimizer& lo);
    private:
        layout_optimizer& _lo;
    };
}