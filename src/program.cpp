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

#include "program_impl.h"
#include "primitive_inst.h"
#include "layout_optimizer.h"

#include "primitive_type.h"
#include "api/CPP/activation.hpp"
#include "api/CPP/convolution.hpp"
#include "api/CPP/deconvolution.hpp"
#include "api/CPP/data.hpp"
#include "api/CPP/eltwise.hpp"
#include "api/CPP/input_layout.hpp"
#include "api/CPP/pooling.hpp"
#include "api/CPP/proposal.hpp"
#include "api/CPP/prior_box.hpp"
#include "api/CPP/reorder.hpp"
#include "api/CPP/detection_output.hpp"
#include "api/CPP/proposal.hpp"
#include "api/CPP/roi_pooling.hpp"
#include "api/CPP/memory.hpp"

#include "internal_primitive.h"
#include "internal_primitive_type_base.h"
#include "convolution_inst.h"
#include "concatenation_inst.h"
#include "crop_inst.h"
#include "deconvolution_inst.h"
#include "detection_output_inst.h"
#include "prior_box_inst.h"
#include "reorder_inst.h"

#include "kernel_selector_helper.h"
#include "sliding_window_utils.h"

namespace cldnn
{

CLDNN_DEFINE_INTERNAL_PRIM(connector)
CLDNN_DEFINE_SIMPLE_PRIM_INST(connector)

namespace {

    //helper function for selecting function basing on the type of the given primitive
    //this is the termination case for parameter pack recurrence, see overload below for logic
    template <class... T>
    void do_for_types(program_node&)
    {
        return;
    }

    //helper function for selecting function basing on the type of the given primitive
    //this function should be explicitly given set of types and implicitly set of functions.
    //both sets should have equal size. First function will be called if type of the given primitive
    //will match first explicitly given type, second will be called if it matches second explicitly given
    //type etc.
    //Functions given as arguments should themselves take std::shared_ptr<const T> as argument
    //where T is the type that should be match if this function should be called
    //
    //example:
    // do_for_types<
    //      convolution,
    //      pooling
    //  >(primitive,
    //      [](typed_program_node<convolution>&){ do something if 'primitive' is a convolution },
    //      [](typed_program_node<pooling>&)    { do something if 'primitive' is a pooling }
    //  );
    template <class T, class... RestOfT, class Func, class... RestOfFuncs>
    decltype(static_cast<void>(std::declval<Func>()(std::declval<typed_program_node<T>&>()))) do_for_types(
        program_node& node,
        Func const& func,
        RestOfFuncs const&... rest)
    {
        if (node.type() == T::type_id())
            func(node.as<T>());
        else
            do_for_types<RestOfT...>(node, rest...);
    }

    template <class T>
    struct single_element_container
    {
        single_element_container(T& t) : elem(&t)
        {}

        constexpr size_t size() const { return 1; }
        auto begin() const { return single_element_container(elem); }
        auto end() const { return single_element_container(nullptr); }
        auto& operator ++() { elem = nullptr; return *this; }
        bool operator !=(single_element_container const& sec) { return elem != sec.elem; }

        decltype(auto) operator *() { return *elem; }

    private:
        single_element_container(T* t) : elem(t)
        {}

        T* elem;
    };

    //helper function which creates single-element array if it's given anything
    //other than std::vector.
    //It should be used in generic code when there's a need to force vector usage
    //in foreach loop over variable which can in one context be a vector or a scalar
    //in another.
    //example:
    // T t;
    // for (auto& string : wrap_if_single(t.dump()))
    //depending on type T, t.dump() may return either std::string or std::vector<std::string>,
    //to ensure compatibility between these cases, wrap_if_single will create single-element
    //container in case t.dump() would return plain std::string.
    //
    // T& case -> returns container which holds T&
    template <class T>
    auto wrap_if_single(T& t)
    {
        return single_element_container<T>(t);
    }

    //helper function which creates single-element array if it's given anything
    //other than std::vector.
    // T const& case -> returns container which holds T const&
    template <class T>
    auto wrap_if_single(T const& t)
    {
        return single_element_container<T const>(t);
    }

    //helper function which creates single-element array if it's given anything
    //other than std::vector.
    // T&& case -> returns container which holds new instance of T created by moving given param
    template <class T>
    auto wrap_if_single(T&& t)
    {
        static_assert(meta::always_false_v<T>, "Wrapping temporary object into single_element_container is an error (requires valid reference)");
        return single_element_container<T>(t);
    }

    //helper function which creates single-element array if it's given anything
    //other than std::vector.
    // std::vector case -> does not wrap, returns t as-is
    decltype(auto) wrap_if_single(primitive::fixed_size_vector_ref const& t)
    {
        return t;
    }

    //helper function for merging the weights/biases buffers on cpu side for depthwise separable convolution optimization
    void merge_buffers(engine_impl::ptr engine, program_node &node, layout target_layout, size_t begin_offset, size_t end_offset)
    {
        memory_impl* data_to_allocate = engine->allocate_buffer(target_layout);

        for (size_t i = begin_offset; i < end_offset; i++)
        {
            auto& weights = node.get_dependency(i).as<data>();
            pointer<char> src_ptr(weights.get_primitive()->mem);
            auto mem_size = weights.get_primitive()->mem.size();
            pointer<char> dst_ptr(memory(api_cast(data_to_allocate), true));
            std::copy(src_ptr.begin(), src_ptr.end(), dst_ptr.begin() + (i - begin_offset)*mem_size);
        }

        for(size_t i = 0; i < end_offset - begin_offset - 1; i++)
            node.remove_dependency(begin_offset + 1);

        auto& data_node = node.get_dependency(begin_offset).as<data>();
        const_cast<data&>(*data_node.get_primitive()).mem = api_cast(data_to_allocate);
    }

    //helper function for getting target layout used in depthwise sep optimization
    layout get_weights_layout(typed_program_node<cldnn::data> &data_node, int32_t split)
    {
        auto& mem_layout = const_cast<data&>(*data_node.get_primitive()).mem.get_layout();

        return layout(mem_layout.data_type, mem_layout.format, { split * mem_layout.size.batch[0], mem_layout.size.feature[0], mem_layout.size.spatial[0], mem_layout.size.spatial[1] });
    }
}

program_impl::program_impl(engine_impl::ptr engine, topology_impl const& topology, build_options const& options)
    : engine(engine), options(options), output_size_handling_enabled(true)
{
    init_graph(topology);
    pre_optimize_graph();
    compile_graph();
    post_optimize_graph();

    engine->compile_program(*this);
}

// TODO: Remove once we will get full support for input/output padding in all primitive implementations.
void program_impl::analyze_output_size_handling_need()
{
    bool handling_needed = false;

    // Calculate output size and compare with specified.
    for (const auto& node : processing_order)
    {
        if (node->is_type<convolution>())
        {
            auto& prim_node = node->as<convolution>();
            const auto& prim = prim_node.get_primitive();

            if (!prim->with_output_size)
                continue;

            tensor specified_output_range({0, 0, prim->output_size.spatial[0], prim->output_size.spatial[1]}, 1);

            auto filter_size = prim_node.weights(0).get_output_layout().size;

            auto calc_output_range = calc_sliding_window_output_range<swor_mode::all>(
                prim_node.input().get_output_layout().size,
                filter_size, prim->input_offset, prim->stride, prim->dilation, true, 1);

            if (specified_output_range != calc_output_range)
                handling_needed = true;
        }
        else if (node->is_type<deconvolution>())
        {
            auto& prim_node = node->as<deconvolution>();
            const auto& prim = prim_node.get_primitive();

            if (!prim->with_output_size)
                continue;

            tensor specified_output_range({0, 0, prim->output_size.spatial[0], prim->output_size.spatial[1]}, 1);

            auto filter_size = prim_node.weights(0).get_output_layout().size;

            auto calc_output_range = calc_sliding_window_needed_input_range(
                prim_node.input().get_output_layout().size,
                filter_size, prim->input_offset, prim->stride, {1, 1, 1, 1}, true, 1);

            if (specified_output_range != calc_output_range)
                handling_needed = true;
        }
        else if (node->is_type<pooling>())
        {
            auto& prim_node = node->as<pooling>();
            const auto& prim = prim_node.get_primitive();

            if (!prim->with_output_size)
                continue;

            tensor specified_output_range({0, 0, prim->output_size.spatial[0], prim->output_size.spatial[1]}, 1);

            // TODO: Check compatibility of output size calculation (with caffe).
            auto calc_output_range = calc_sliding_window_output_range<swor_mode::exceed_once_data>(
                prim_node.input().get_output_layout().size,
                prim->size, prim->input_offset, prim->stride, {1, 1, 1, 1}, true, 1);

            if (specified_output_range != calc_output_range)
                handling_needed = true;
        }
    }

    output_size_handling_enabled = handling_needed;
}

std::list<std::shared_ptr<program_node>> program_impl::get_nodes() const
{
    std::list<std::shared_ptr<program_node>> ret;

    for (auto& node : processing_order)
        ret.push_back(nodes_map.at(node->id()));

    //does not expose connector at the end of a graph
    if (!ret.empty() && ret.back()->is_type<connector>())
        ret.pop_back();

    return ret;
}

void program_impl::init_graph(topology_impl const& topology)
{
    auto const& topo_map = topology.get_primitives();
    for (auto const& prim : topo_map)
    {
        auto& n = get_or_create(prim.second);
        inputs.push_back(&n);
    }

    for (auto itr = inputs.begin(); itr != inputs.end(); )
    {
        auto node_itr = itr++;
        auto& node = (*node_itr);
        auto deps = node->get_primitive()->dependecies();
        if (deps.empty())
            continue;

        //add pointers to node's dependencies
        for (auto& dep : deps)
        {
            try {
                auto dep_node = nodes_map.at(dep);
                node->dependencies.push_back(dep_node.get());
                dep_node->users.push_back(node);
            }
            catch(...) {
                throw std::runtime_error("Program doesn't contain primitive: " + dep +
                    " that is input to: " + node->get_primitive()->id);
            }
        }

        //primitive has dependencies so remove it from 'inputs'
        inputs.erase(node_itr);
    }

    set_outputs();
    calc_processing_order();
    mark_constants();
    mark_data_flow();
    calc_dominators();
}

void program_impl::pre_optimize_graph()
{
    trim_to_outputs();

    if (get_engine()->configuration().enable_parallelisation)
        reorder_nodes_for_parallel_execution();

    analyze_output_size_handling_need();

    if (options.get<build_option_type::optimize_data>()->enabled())
    {
        layout_optimizer lo(engine, true, output_size_handling_enabled);
        reorder_inputs(lo);
        // this code should move to post compilation after kernel selector will support handling reorder bias
        pre_optimize_bias(lo);
    }

    prepare_padding();
    prepare_depthwise_sep_opt();

    //try to fuse buffers (i.e. depth_concat in bfyx format) after padding calculations
    if (options.get<build_option_type::optimize_data>()->enabled())
    {
        prepare_buffer_fusing();
    }
}

void program_impl::post_optimize_graph()
{
    layout_optimizer lo(engine);
    post_optimize_weights(lo);
    //prepare_padding(); - TODO: padding should be prepare according to the kernels needs
}

void program_impl::compile_graph()
{
    for (auto& node : processing_order)
    {
        if (!node->is_type<internal_primitive>())
        {
            node->get_output_layout();
            node->selected_impl = node->type()->choose_impl(*engine, *node);
        }
    }
}

void program_impl::set_outputs()
{
    auto outputs_option = options.get<build_option_type::outputs>();

    // in debug mode select all primitives as output
    if (options.get<build_option_type::debug>()->enabled())
    {
        for (auto& node : nodes_map)
        {
            //do not add cldnn::data as output
            if (node.second->type() == data::type_id())
                continue;

            node.second->set_output(true);
            outputs.push_back(node.second.get());
        }

        return;
    }

    if (!outputs_option->outputs.empty())
    {
        for (auto const& output : outputs_option->outputs)
        {
            auto o_node = nodes_map.at(output);
            o_node->set_output(true);
            outputs.push_back(o_node.get());
        }
    }
    else
    {
        for (auto& node : nodes_map)
            if (node.second->is_endpoint())
            {
                node.second->set_output(true);
                outputs.push_back(node.second.get());
            }
    }
}

void program_impl::calc_processing_order()
{
    processing_order.clear();

    //run dfs to sort nodes topologically
    for (auto input : inputs)
    {
        if (input->is_marked())
            continue;

        input->mark();
        std::list<std::pair<program_node*, std::list<program_node*>::const_iterator>> stack = { std::make_pair(input, input->users.begin()) };

        while (!stack.empty()) //imitate call stack
        {
        new_frame:
            auto& frame = stack.back();

            while (frame.second != frame.first->users.end())
            {
                auto successor = *frame.second;
                ++frame.second;

                if (!successor->is_marked())
                {
                    successor->mark();

                    //recurrence call
                    stack.push_back(std::make_pair(successor, successor->users.begin()));
                    goto new_frame;
                }
            }

            //we have finished processing one node so add it to the processing queue
            processing_order.push_front(frame.first);
            frame.first->processing_itr = processing_order.begin();

            //return from call
            stack.pop_back();
        }
    }

    uint32_t idx = 0;
    for (auto& node : processing_order)
    {
        node->processing_num = ++idx;
        node->unmark();
    }
}

void program_impl::mark_constants()
{
    for (auto& node : processing_order)
    {
        if (node->is_type<data>() || node->is_type<prior_box>())
        {
            node->constant = true;
            continue;
        }
        else if (node->dependencies.empty())
            continue;

        node->constant = true;
        for (auto& dep : node->get_dependencies())
        {
            if (!dep->constant)
            {
                node->constant = false;
                break;
            }
        }
    }
}

void program_impl::mark_data_flow()
{
    std::list<program_node*> stack;
    for (auto const& node : processing_order)
    {
        if (node->is_endpoint() && !node->constant)
        {
            stack.push_back(node);
            node->data_flow = true;
            node->mark();
        }
    }

    while (!stack.empty())
    {
        auto node = stack.front();
        stack.pop_front();

        size_t dep_idx = 0;
        size_t inputs_count = (node->is_type<internal_primitive>() ? node->get_dependencies().size() : node->get_primitive()->input.size());
        //TODO: remove this hack after addition of constants propagation pass
        if (node->is_type<detection_output>() || node->is_type<proposal>())
            inputs_count = 2; //ignore third input as it is related to prior boxes (i.e. concat of prior-boxes)

        for (auto dep : node->get_dependencies())
        {
            bool data_flow = (dep_idx < inputs_count && !dep->constant);
            ++dep_idx;
            if (!data_flow)
                continue;

            dep->data_flow = data_flow;

            if (dep->is_marked())
                continue;

            stack.push_back(dep);
            dep->mark();
        }
    }

    for (auto& node : processing_order)
    {
        node->main_branch = node->data_flow;
        assert(!node->constant || !node->data_flow); //node which is constant cannot be marked as data flow
        node->unmark();
    }
}

void program_impl::calc_dominators()
{
    if (nodes_map.empty())
        return;

    //Algorithm per: Keith D. Cooper, Timothy J. Harvey, and Ken Kennedy "A Simple, Fast Dominance Algorithm"
    //url: http://www.hipersoft.rice.edu/grads/publications/dom14.pdf

    //Please note that in our representation we only care for immidiate dominators which are not direct predecessors.

    //Althought we don't know the input node (since we cannot distinguish image input from weights/biases), we can simply trim our graph
    //for this analysis so it'd start with the first primitive which is used by more than one user. There are few assumptions here, though:
    // 1. weights and biases cannot be shared (they have to have unique user)
    // 2. the network should have exactly one input (OOO execution will not be enabled for independent paths, at the beginning, if more at than one input is provided)
    //please note that it could be also possible to create dummy super-source so we could guarantee that the graph will have only one input,
    //however such approach would also require us to analys all weights and biases which is not really what we want to do, hence the simplification described above.

    //When it comes to the multiple endpoints, we can create dummy node to gather all of them. Super-sink is not a problem since the number of endpoints
    //is much less than inputs, due to weights and biases. Also unique endpoint is handy when dealing with dominators frontier and determinating
    //if the node lies within a 'main' branch of the network.

    //...so firstly create super-sink, find all endpoints
    {
        std::list<program_node*> endpoints;
        for (auto const& node : processing_order)
            if (node->is_endpoint())
                endpoints.push_back(node);

        assert(endpoints.size() > 0 && "Network without endpoints?");

        //if more than one endpoint, create sink
        if (endpoints.size() > 1)
        {
            std::shared_ptr<program_node> node = std::make_shared<connector_node>(*this);
            node->data_flow = true;
            nodes_map.insert(std::make_pair(node->id(), node));

            for (auto const& endpoint : endpoints)
            {
                endpoint->users.push_back(node.get());
                node->dependencies.push_back(endpoint);
            }

            node->processing_itr = processing_order.insert(processing_order.end(), node.get());
            node->processing_num = static_cast<uint32_t>(processing_order.size()) + 1;
        }
    }

    //As mentioned, we want to find 'first' node with at least two users, where by 'first' we meant such node that every other can be reached from it (except for w/b),
    //the first one from the set of topologically sorted nodes should have this property.
    //Also, we take iterator to it rather than a node itself since, accoring to the algorithm, we will need to process nodes in reversed-postorder, which is equivalent for ordering produced by
    //topological sorting.
    auto root_itr = processing_order.begin();
    while (root_itr != processing_order.end())
    {
        if ((*root_itr)->get_users().size() > 1)
            break;

        ++root_itr;
    }

    //there are no splits so simply end (for each n, idom(n) e { dpred(n) })
    if (root_itr == processing_order.end())
        return;

    auto root = *root_itr;
    root->dominator = root; //its not valid accordingly to the definition of 'program_node::dominator' field, but it's required by the algorithm - at the end this field should be reverted to nullptr
    bool changed = true;

    const auto intersects = [](program_node* n1, program_node* n2) -> program_node*
    {
        assert(n1 != nullptr);
        assert(n2 != nullptr);
        while (n1->processing_num != n2->processing_num)
        {
            //please note: we use reverse-postorder numbering so conditions are swapped in regard to the original algorithm (which uses postorder here)
            while (n1->processing_num > n2->processing_num)
            {
                n1 = n1->dominator;
                assert(n1 != nullptr);
            }
            while (n1->processing_num < n2->processing_num)
            {
                n2 = n2->dominator;
                assert(n2 != nullptr);
            }
        }

        return n1;
    };

    while (changed)
    {
        changed = false;

        //for all nodes, in reverse postorder (except root)
        auto itr = root_itr;
        ++itr;
        while (itr != processing_order.end())
        {
            auto node = *(itr++);
            if (!node->is_in_data_flow()) //eliminate helper nodes
                continue;

            //pick first (processed) predecessor
            auto pred_itr = node->get_dependencies().begin();
            while (pred_itr != node->get_dependencies().end() && (*pred_itr)->dominator == nullptr)
                ++pred_itr;
            assert(pred_itr != node->get_dependencies().end()); //we are processing nodes in reverse postorder, so at least one our predecessor should have been processed at this point

            auto new_idom = *pred_itr;

            //new_idom <- first predecessor
            //for all other predecessors of node (note: it's not clear if authors mean DIRECT predecessors but I've assumed so, I makes more sense when looking at the examples)
            pred_itr = node->get_dependencies().begin();
            while (pred_itr != node->get_dependencies().end())
            {
                auto pred = *(pred_itr++);
                if (pred->dominator != nullptr) //doms[pred] already calculated
                    new_idom = intersects(pred, new_idom);
            }

            if (node->dominator != new_idom)
            {
                node->dominator = new_idom;
                changed = true;
            }
        }
    }

    //change meaningless idoms to nullptr (i.e. idom(node) == node or idom(node) e { dpred(node) })
    //process items in reverse order to guarantee not-null dominators when checking node's predecessors (needed for dominance frontier)
    auto ritr = processing_order.rbegin();
    while (ritr != processing_order.rend())
    {
        auto node = *(ritr++);
        if (!node->is_in_data_flow())
            continue;
        else if (node->dominator == node)
            node->dominator = nullptr;
        else if (node->get_dependencies().size() == 1)
            node->dominator = nullptr;
        else if (std::find(node->get_dependencies().begin(), node->get_dependencies().end(), node->dominator) != node->get_dependencies().end())
            node->dominator = nullptr;
        else //if dominator's not trivial, check its frontier to determinate if it lies on a 'main' branch
        {
            if (!node->dominator->joint)
                node->dominator->joint = node;

            for (auto dep : node->get_dependencies())
            {
                while (dep != node->dominator)
                {
                    if (!dep->data_flow)
                        break;

                    dep->main_branch = false;
                    dep = dep->dominator;
                    assert(dep != nullptr);
                }
            }
        }

        if (node == root)
            break;
    }
}

void program_impl::trim_to_outputs()
{
    backward_bfs(nullptr, [this](program_node& node) {
        if (!node.is_marked() && !node.is_type<internal_primitive>())
        {
            processing_order.erase(node.processing_itr);
            optimized_out.push_back(node.id());

            if (node.is_input())
            {
                inputs.remove(&node);
            }
        }
    });

    for (auto const& opt : optimized_out)
    {
        nodes_map.erase(opt);
    }
}

void program_impl::reorder_nodes_for_parallel_execution()
{
    if (processing_order.empty())
        return;

    //note: during computations perfomed by this function, both program_node::processing_itr and program_node::processing_num might be invalidated

    //firstly, move all helpers at the beginning of processing queue to prevent them from being parallelised
    std::list<program_node*> old_order;
    std::swap(old_order, processing_order);
    assert(processing_order.empty() && !old_order.empty());

    const auto push_back = [this](program_node* node)
    {
        processing_order.push_back(node);
        node->processing_itr = --processing_order.end();
        node->processing_num = static_cast<uint32_t>(processing_order.size());
    };

    auto itr = old_order.begin();
    while (itr != old_order.end())
    {
        auto* node = (*itr);
        node->processing_num = 0;
        if (!node->is_in_data_flow())
        {
            push_back(node);
            itr = old_order.erase(itr);
        }
        else
            ++itr;
    }

    //now identify all splits and try to reorder nodes
    itr = old_order.begin();
    while (itr != old_order.end())
    {
        auto* split = (*itr);
        if (!split->is_split_point())
        {
            push_back(split);
            ++itr;
            continue;
        }

        //the node is a split point, reorder all nodes between node and node->get_joint() in queue so they can be run in a parallel way
        auto joint = split->get_joint();

        //a nice thing: we can use already topologically sorted nodes to calculated maximum distance from the source for each node in a range (split, joint)
        //then we can simply sort them so nodes which are in the same distance will be next to each other in the resulting list
        auto sub_itr = itr;
        while (*sub_itr != joint)
        {
            auto node = (*sub_itr++);
            for (auto& user : node->get_users())
                user->processing_num = std::max(user->processing_num, node->processing_num + 1);
        }

        //bucket sort nodes basing on their distance from source
        std::vector<std::list<program_node*>> dist_map;
        dist_map.resize(joint->processing_num);
        sub_itr = itr;
        while (*sub_itr != joint)
        {
            auto node = (*sub_itr++);
            dist_map[node->processing_num].push_back(node);
        }

        //insert sorted nodes to a resulting list in order of their distance
        for (auto& dist : dist_map)
            for (auto& node : dist)
                push_back(node);

        itr = sub_itr;
        assert(*itr == joint);
        joint->processing_num = 0;
    }
}

void program_impl::reorder_inputs(layout_optimizer& lo)
{
    //first pass to set layout optimization_attributes for topology
    for (auto& p : nodes_map)
    {
        auto& prim = *p.second;
        if (prim.type() == cldnn::convolution::type_id())
        {
            if (prim.as<convolution>().get_primitive()->split() > 1)
                lo.set_optimization_attribute(layout_optimizer::optimization_attributes_type::splitted_convolution, 1);
        }

        //list of layers that do not support yxfb or perform worse than bfyx
        if (prim.type() == cldnn::detection_output::type_id() || prim.type() == cldnn::proposal::type_id() ||
            prim.type() == cldnn::roi_pooling::type_id() || prim.type() == cldnn::deconvolution::type_id())
            lo.set_optimization_attribute(layout_optimizer::optimization_attributes_type::bfyx_only_layer, 1);
    }

    const auto reorder_input = [this, &lo](typed_program_node<convolution>& conv_node)
    {
        auto conv_prim = conv_node.get_primitive();
        auto& input_node = conv_node.get_dependency(0);
        auto weights_layout = conv_node.weights(0).get_output_layout();

        std::shared_ptr<reorder> new_input = nullptr;

        if (input_node.type() == data::type_id())
        {
            new_input = lo.add_weights_for_optimization(input_node.as<data>().typed_desc(),
                layout_optimizer::data_type::input,
                conv_prim,
                weights_layout).first;
        }
        else if (input_node.type() == input_layout::type_id())
        {
            new_input = lo.get_reorder(
                input_node.as<input_layout>().get_primitive()->layout,
                input_node.id(),
                layout_optimizer::data_type::input,
                conv_prim,
                weights_layout).first;
        }
        else if (input_node.type() == reorder::type_id()) //convolution's input is a reorder
        {
            auto reorder_prim = input_node.as<reorder>().typed_desc();
            auto& reorder_input = input_node.get_dependency(0);
            auto reorder_layout = reorder_input.get_output_layout();
            reorder_layout.data_type = reorder_prim->output_data_type;
            new_input = lo.get_reorder(
                reorder_layout,
                reorder_prim->id,
                layout_optimizer::data_type::input,
                conv_prim,
                weights_layout).first;

            if (new_input) //output format is not optimal
            {
                auto reorder_input_layout = reorder_input.get_output_layout();

                auto opt_layout = layout(new_input->output_data_type, new_input->output_format, reorder_input_layout.size);
                if (reorder_input_layout == opt_layout) //reorder 'breaks' optimal format
                {
                    if (reorder_prim->subtract_per_feature.empty() &&
                        reorder_prim->mean.empty() &&
                        !reorder_prim->output_padding) //just plain reorder
                    {
                        conv_node.replace_dependency(0, reorder_input);
                        new_input = nullptr;
                    }
                    else //change reorder's output layout
                    {
                        reorder_prim->output_format = opt_layout.format;
                        reorder_prim->output_data_type = opt_layout.data_type;
                        new_input = nullptr;
                    }
                }
                else //current reorder gives bad output, simply change it
                {
                    reorder_prim->output_format = opt_layout.format;
                    reorder_prim->output_data_type = opt_layout.data_type;
                    new_input = nullptr;
                }
            }

            input_node.recalc_output_layout();
        }
        else
        {
            new_input = lo.get_reorder(
                input_node.get_output_layout(),
                input_node.id(),
                layout_optimizer::data_type::input,
                conv_prim,
                weights_layout).first;
        }

        if (new_input)
        {
            add_intermediate(new_input, conv_node, 0);
            conv_node.recalc_output_layout();
        }
    };

    const auto reorder_input_detection_output = [this, &lo](typed_program_node<detection_output>& detection_output_node)
    {
        auto detection_output_prim = detection_output_node.get_primitive();
         
        for (size_t i = 0; i < detection_output_node.get_dependencies().size(); i++)
        {
            auto& input = detection_output_node.get_dependency(i);
            std::shared_ptr<reorder> new_input = lo.get_reorder(
                input.get_output_layout(),
                input.id(),
                layout_optimizer::data_type::input,
                detection_output_prim).first;

            if (new_input)
            {
                add_intermediate(new_input, detection_output_node, i);
            }
        }
    };

    for (auto& prim : processing_order)
    {
        //there's an assumption that only convolution will take data/input_layout as input
        //exception to that rule would be a convolution which takes a reorder as input - see reoder_input above
        do_for_types<convolution, detection_output>(*prim,
            reorder_input,                  //case for convolution
            reorder_input_detection_output  //case for detection-output
            );
    }
}

void program_impl::pre_optimize_bias(layout_optimizer& lo)
{
    std::list<program_node*> outputs_to_recalc;

    //lambda function which finds weights primitive with given pimitive_id and adds it to weights_optimizer
    //this function is reused in all cases (convolution weights, convolution bias, fc weights and fc bias) and does
    //some basic sanity checks about existence of the primitive and it's type. throws std::logic_error
    const auto add_bias = [this, &lo, &outputs_to_recalc](program_node& bias, auto& node, layout const& output_layout, size_t dep_idx)
    {
        const auto bias_type = layout_optimizer::data_type::bias;
        if (bias.type() == data::type_id())
        {
            lo.add_weights_for_optimization(
                bias.as<data>().typed_desc(),
                bias_type,
                node.get_primitive(),
                output_layout);
            outputs_to_recalc.push_back(&bias);
        }
        else if (bias.type() == input_layout::type_id())
        {
            auto reorder = lo.get_reorder(
                bias.as<input_layout>().typed_desc()->layout,
                bias.id(),
                bias_type,
                node.get_primitive(),
                output_layout);

            if (reorder.first)
                this->add_intermediate(reorder.first, node, dep_idx);
        }
        else
            throw std::logic_error("Optimization of weights which are neither of type cldnn::data nor cldnn::input_layout!");
    };

    //generic lambda function which prepares given primitive for weights optimization
    //it deduces the type of weights from the type of the argument and calls 'add_weights' for all
    //weights and biases used by given primitive.
    //argument should match few requirements:
    // - it should be of a form 'typed_program_node<T>&'
    // - both 'T.weights' and 'T.bias' should be either of type 'primitive_id' or 'std::vector<primitive_id>'
    const auto prep_opt = [this, &add_bias, &outputs_to_recalc](auto& node) -> void
    {
        auto output_layout = node.get_output_layout();

        auto weights_offset = node.get_primitive()->input.size();
        auto bias_offset = weights_offset + wrap_if_single(node.get_primitive()->weights).size();
        for (auto i = weights_offset; i < bias_offset; ++i)
        {
            outputs_to_recalc.push_back(&node.get_dependency(i));
        }
        for (auto i = bias_offset; i < node.get_dependencies().size(); ++i)
        {
            add_bias(node.get_dependency(i), node, output_layout, i);
        }
    };

    for (auto& p : nodes_map)
    {
        auto& prim = *p.second;
        if (prim.type() == convolution::type_id())
        {
            prep_opt(prim.as<convolution>());
        }
        else if (prim.type() == deconvolution::type_id())
        {
            prep_opt(prim.as<deconvolution>());
        }
        else if (prim.type() == fully_connected::type_id())
        {
            prep_opt(prim.as<fully_connected>());
        }
    }

    //all optimizing primitives has been added and inputs for all primitives has been updated.
    //run reorders and replace cldnn::data::mem
    lo.optimize();

    for (auto dnode : outputs_to_recalc)
        dnode->recalc_output_layout();
}

void program_impl::prepare_depthwise_sep_opt()
{
    const auto prepare_depthwise_sep_opt = [this](auto& node) -> void
    {
        //enable optimization only when split = IFM and split >= 16
        if (!(node.get_dependency(0).get_output_layout().size.feature[0] == node.get_primitive()->split()) ||
            !(node.get_primitive()->split() >= 16))
            return;

        //make sure the weights and biases are data type and 
        //are not reused in other primitives as they will be overriden with concatenated ones
        for (size_t i = 1; i < node.get_dependencies().size(); i++)
        {
            auto& weights_or_biases = node.get_dependency(i);
            if (weights_or_biases.get_users().size() > 1 || weights_or_biases.type() != data::type_id())
                return;
        }

        node.set_depthwise_sep_opt(true);
    };

    //depthiwise separated convolution/deconvolution optimization
    for (auto& p : nodes_map)
    {
        auto& prim = *p.second;
        do_for_types<deconvolution, convolution>(prim,
            prepare_depthwise_sep_opt,   //case for deconvolution
            prepare_depthwise_sep_opt    //case for convolution
            );
    }
}

void program_impl::post_optimize_weights(layout_optimizer& lo)
{
    //lambda function which finds weights primitive with given pimitive_id and adds it to weights_optimizer
    //this function is reused in all cases (convolution weights, convolution bias, fc weights and fc bias) and does
    //some basic sanity checks about existence of the primitive and it's type. throws std::logic_error
    const auto add_weights = [this, &lo](program_node const& weights, auto& node, size_t dep_idx)
    {
        auto wtype = weights.type();
        auto* impl = node.get_selected_impl().get();
        auto output_layout = node.get_output_layout();
        const auto weights_type = layout_optimizer::data_type::weights;
        if (wtype == data::type_id())
        {
            lo.add_weights_for_optimization(
                impl->_kernel_data.weightsReorderParams,
                weights.as<data>().typed_desc(),
                weights_type);
        }
        else if (wtype == input_layout::type_id())
        {
            auto reorders = lo.get_generic_layer(
                impl->_kernel_data.weightsReorderParams,
                weights.as<input_layout>().typed_desc()->id,
                output_layout,
                weights_type);

            for (auto& reorder : reorders)
            {
                this->add_intermediate(reorder.first, node, dep_idx);
            }
        }
    };

    //generic lambda function which prepares given primitive for weights optimization
    //it deduces the type of weights from the type of the argument and calls 'add_weights' for all
    //weights used by given primitive.
    //argument should match few requirements:
    // - it should be of a form 'typed_program_node<T>&'
    // - 'T.weights' should be either of type 'primitive_id' or 'std::vector<primitive_id>'
    const auto prep_opt = [this, &add_weights](auto& node) -> void
    {
        auto weights_offset = node.get_primitive()->input.size();
        auto bias_offset = weights_offset + wrap_if_single(node.get_primitive()->weights).size();
        for (auto i = weights_offset; i < bias_offset; i++)
        {
            add_weights(node.get_dependency(i), node, i);
        }
    };

    for (auto& p : nodes_map)
    {
        auto& prim = *p.second;
        if (prim.type() == convolution::type_id())
        {
            prep_opt(prim.as<convolution>());
        }
        else if (prim.type() == deconvolution::type_id())
        {
            prep_opt(prim.as<deconvolution>());
        }
        else if (prim.type() == fully_connected::type_id())
        {
            prep_opt(prim.as<fully_connected>());
        }
    }

    //all optimizing primitives has been added and inputs for all primitives has been updated.
    //run reorders now
    lo.optimize();

    const auto prep_opt_depthwise_sep = [this](auto& node) -> void
    {
        if (!node.get_depthwise_sep_opt())
            return;

        auto weights_offset = node.get_primitive()->input.size();
        auto bias_offset = weights_offset + wrap_if_single(node.get_primitive()->weights).size();

        const auto& weights_layout = node.get_dependency(1).get_output_layout();
        const auto& split = node.get_primitive()->split();

        //concatenate weights
        {
            //if weights were optimized it is needed to use the sizes after optimization
            auto target_layout = get_weights_layout(node.get_dependency(1), split);
            merge_buffers(engine, node, target_layout, weights_offset, bias_offset);
        }

        //concatenate biases
        if (node.get_primitive()->bias.size() != 0)
        {
            auto target_layout = layout(weights_layout.data_type, cldnn::format::bfyx, { 1, 1, weights_layout.size.batch[0] * split, 1 });
            merge_buffers(engine, node, target_layout, weights_offset + 1, bias_offset + 1);
        }

        //override node split, as only one kernel will be executed
        node.set_split(1);
    };

    //depthiwise separated convolution/deconvolution optimization
    for (auto& p : nodes_map)
    {
        auto& prim = *p.second;
        do_for_types<deconvolution, convolution>(prim,
            prep_opt_depthwise_sep,   //case for deconvolution
            prep_opt_depthwise_sep    //case for convolution
            );
    }
}

void program_impl::apply_needed_padding(program_node& node, program_node& prev_node,
                                                const padding& needed_padding)
{
    auto target_layout = prev_node.get_output_layout();

    // Short circuit if padding did not change.
    if (target_layout.data_padding == needed_padding)
        return;

    // Special handling for input nodes.
    if (prev_node.is_type<input_layout>())
    {
        target_layout.data_padding = needed_padding;

        auto r_prim = std::make_shared<reorder>("reorder_" + prev_node.id(), prev_node.id(), target_layout);
        add_intermediate(r_prim, node, 0);
        return;
    }

    prev_node.merge_output_padding(needed_padding);
}

void program_impl::prepare_padding()
{
    if (output_size_handling_enabled)
    {
        // Prepare upper padding for primitives that support output_size parameter.
        for (const auto& node : processing_order)
        {
            if (node->is_type<convolution>())
            {
                auto& prim_node = node->as<convolution>();
                const auto& prim = prim_node.get_primitive();

                if (!prim->with_output_size)
                    continue;

                auto filter_size = prim_node.weights(0).get_output_layout().size;

                auto needed_padding = calc_sliding_window_needed_input_padding(
                    prim_node.input().get_output_layout(),
                    prim->output_size, filter_size, prim->input_offset, prim->stride, prim->dilation, false, 1);

                apply_needed_padding(prim_node, prim_node.input(), needed_padding);
            }
            else if (node->is_type<deconvolution>())
            {
                auto& prim_node = node->as<deconvolution>();
                const auto& prim = prim_node.get_primitive();

                if (!prim->with_output_size)
                    continue;

                auto filter_size = prim_node.weights(0).get_output_layout().size;

                auto needed_padding = calc_sliding_window_needed_input_padding(
                    prim_node.input().get_output_layout(),
                    prim->output_size, filter_size, prim->input_offset, prim->stride, {1, 1, 1, 1}, true, 1);

                apply_needed_padding(prim_node, prim_node.input(), needed_padding);
            }
            else if (node->is_type<pooling>())
            {
                auto& prim_node = node->as<pooling>();
                const auto& prim = prim_node.get_primitive();

                if (!prim->with_output_size)
                    continue;

                // NOTE: Currently there is no pooling implementation/pooling mode which does not check input data range.
                // There is no need to add padding requirements on pooling inputs.
                //auto needed_padding = calc_sliding_window_needed_input_padding(
                //    prim_node.input().get_output_layout(),
                //    prim->output_size, prim->size, prim->input_offset, prim->stride, {1, 1, 1, 1}, false, 1);
                auto needed_padding = prim_node.input().get_output_layout().data_padding;

                apply_needed_padding(prim_node, prim_node.input(), needed_padding);
            }
        }
    }

    // Prepare optimized padding for bfyx convolution.
    for (auto& pair : nodes_map)
    {
        if (pair.second->type() != convolution::type_id())
            continue;

        auto& node = pair.second->as<convolution>();
        if (node.get_dependencies().empty())
            continue;

        auto conv = node.get_primitive();
        auto& conv_input_node = node.get_dependency(0);
        auto conv_layout = node.get_output_layout();

        // right now output padding optimization is only available for bfyx format and data type = float32
        if (conv_layout.format != cldnn::format::bfyx)
        {
            continue;
        }

        // Calculating input padding needed for convolution
        auto& filter_node = node.as<convolution>().weights(0);
        auto filter_prim = filter_node.get_primitive();

        layout filter_layout = filter_node.get_output_layout();
        
        // convolution have only one input primitive
        auto prev_prim_output_layout = conv_input_node.get_output_layout();

        // Compute initial required paddings for primitive used as input for convolution.
        auto input_offset = conv->input_offset;
        auto stride = conv->stride;
        auto dilation = conv->dilation;

        auto input_limit_x = input_offset.spatial[0] + (conv_layout.size.spatial[0] - 1) * stride.spatial[0] + (filter_layout.size.spatial[0] - 1) * dilation.spatial[0] + 1;
        auto input_limit_y = input_offset.spatial[1] + (conv_layout.size.spatial[1] - 1) * stride.spatial[1] + (filter_layout.size.spatial[1] - 1) * dilation.spatial[1] + 1;

        auto left_padding = std::max(-input_offset.spatial[0], 0);
        auto top_padding = std::max(-input_offset.spatial[1], 0);
        auto right_padding = std::max(input_limit_x - prev_prim_output_layout.size.spatial[0], 0);
        auto bottom_padding = std::max(input_limit_y - prev_prim_output_layout.size.spatial[1], 0);

        // Adjust right padding, so entire buffer size in X dimension is properly aligned.
        // TODO: NOTE: Will be reenabled with next check-in once heuristic for line-aligned algorithm will be added.
        //auto needed_buffer_size_x = static_cast<cldnn::tensor::value_type>(
        //    round_up_to(left_padding + prev_prim_output_layout.size.spatial[0] + right_padding, 16));
        //right_padding = needed_buffer_size_x - left_padding - prev_prim_output_layout.size.spatial[0];

        cldnn::padding needed_padding({ 0, 0, left_padding, top_padding }, { 0, 0, right_padding, bottom_padding }, 0);
        needed_padding = padding::max(prev_prim_output_layout.data_padding, needed_padding);

        apply_needed_padding(node, conv_input_node, needed_padding);
    }
}

void program_impl::prepare_buffer_fusing()
{
    bool is_debug = options.get<build_option_type::debug>()->enabled();
    for (auto& node : processing_order)
    {
        do_for_types<concatenation>(*node, [this, is_debug](concatenation_node& node)
        {
            //if any of this node's inputs is used by more than one primitive do not fuse buffers,
            //also, if an input is marked as network output, prevent optimizations which would affect a form of its output (unless debug flag is set)
            // todo: in future, if this case is problem, it can be optimized further to enable buffer fusing
            //       per single input rather than all/none
            // + restrict input types to pooling, convolution and activation only due to problems with output padding on b and f
            for (auto const& input : node.get_dependencies())
                if (input->get_users().size() > 1 ||
                    (!input->is_type<pooling>() && !input->is_type<convolution>() && !input->is_type<activation>()) ||
                    (input->is_output() && !is_debug))
                    return;

            // buffer fusing should not be performed if one of inputs produces padded output since
            // it could break desired memory alignment. On the other hand, if this node uses all inputs
            // exclusively (see check above) they should not have output padding set since concatenation
            // does not ask for any.
            assert(!node.has_padded_dependency());
            if (node.has_padded_dependency())
                return;

            auto concat_axis = node.get_primitive()->axis;
            auto padd = node.get_output_layout().data_padding;

            //calculate lower and upper paddding so they sum up to the buffer size
            // at the beginning lower padd points to the starting position of the output data
            //
            //   |--- lower padd ---| ------------------ upper padd -----------------------|
            //   |-- output padd ---| ----- input1 ------|----- input2 -----|-- out padd --|
            tensor lower_padd = padd.lower_size();
            tensor upper_padd = padd.upper_size();

            upper_padd.raw[concat_axis] = node.get_output_layout().get_buffer_size().raw[concat_axis] - lower_padd.raw[concat_axis];

            for (auto const& input : node.get_dependencies())
            {
                auto input_lenght = input->get_output_layout().size.raw[concat_axis];

                // shrink upper pad so it points at the end of the input's buffer
                //
                //   |--- lower padd ---|                    |---------- upper padd -----------|
                //   |-- output padd ---| ----- input1 ------|----- input2 -----|-- out padd --|
                upper_padd.raw[concat_axis] -= input_lenght;

                // set new padding for input
                input->set_output_padding(padding(lower_padd.sizes(), upper_padd.sizes()));

                // move lower padd further
                //
                //   |-------------- lower padd -------------|---------- upper padd -----------|
                //   |-- output padd ---| ----- input1 ------|----- input2 -----|-- out padd --|

                lower_padd.raw[concat_axis] += input_lenght;
            }
            
            node.can_be_optimized(true);
        });

        // zero copy 
        do_for_types<crop>(*node, [this, is_debug](crop_node& node)
        {
            //if the node is marked as network output, prevent optimizations which would affect a form of its output, unless debug flag is set
            if (node.is_output() && !is_debug)
                return;

            //connector mights have been added at the end of the network, if that is a case ignore it
            if (node.get_users().size() == 1 && node.get_users().front()->is_type<connector>())
                return;

            if (node.get_dependencies().size() == 1 &&
                node.get_users().size() > 0)
            {
                // optimization is avaiable for croping across depth(features) only
                // if output padding has defined padding accross featuers already it wouldn't 
                // work because it expect to have zeros in the padded area.
                auto format = node.get_output_layout().format;
                auto crop_prim = node.get_primitive();
                auto input_layout = node.get_dependency(0).get_output_layout();
                auto in_place_layout = node.get_output_layout();
                auto out_padd = node.get_output_layout().data_padding;
                if (format == format::bfyx &&
                    crop_prim->reference_input.batch[0] == input_layout.size.batch[0] &&
                    crop_prim->reference_input.spatial[0] == input_layout.size.spatial[0] &&
                    crop_prim->reference_input.spatial[1] == input_layout.size.spatial[1] &&
                    out_padd.lower_size().feature[0] == 0 &&
                    out_padd.upper_size().feature[0] == 0)
                {
                    //  Regular crop
                    //  crop input buffer
                    //  |___________data____________|
                    //  
                    //  crop output buffer
                    //  |-------->| offsets[f]  |<--|
                    //            |_____data____|
                    //             <------------>
                    //           reference size
                    //
                    //  Inplace crop
                    //  crop output buffer
                    //  |_low_pad_|__data_size__|___|<-upper pad

                    node.set_output_padding(padding(
                    { out_padd.lower_size().batch[0], crop_prim->offsets.feature[0], out_padd.lower_size().spatial[0], out_padd.lower_size().spatial[1] },
                    { out_padd.upper_size().batch[0], in_place_layout.size.feature[0] - crop_prim->offsets.feature[0] - crop_prim->reference_input.feature[0],
                        out_padd.upper_size().spatial[0], out_padd.upper_size().spatial[1] }));
                    node.can_be_optimized(true);
                }
            }
        });

        do_for_types<reorder>(*node, [this](reorder_node& node)
        {
            auto const& input = node.get_dependencies();
            //Optimization only available in case of layers that support different input and output formats.
            //todo: new api needs to be created to read such caps
            if (!input[0]->is_type<pooling>())
                return;

            if (input[0]->get_users().size() != 1)
                return;

            input[0]->set_output_layout(node.get_output_layout());

            node.can_be_optimized(true);
        });
    }
}

program_node& program_impl::get_or_create(std::shared_ptr<primitive> prim)
{
    auto itr = nodes_map.lower_bound(prim->id);
    if (itr != nodes_map.end() && itr->first == prim->id)
        return *itr->second;

    auto new_node = prim->type->create_node(*this, prim);
    nodes_map.insert(itr, { prim->id, new_node });
    return *new_node;
}

void program_impl::add_intermediate(program_node& node, program_node& next, size_t prev_idx)
{
    auto& prev = next.get_dependency(prev_idx);
    //firstly add connection, later replace dependency, so 'prev' won't become dangling and therefore removed
    add_connection(prev, node);
    next.replace_dependency(prev_idx, node);
    node.processing_itr = processing_order.insert(next.processing_itr, &node);
}

void program_impl::remove_if_dangling(program_node& node)
{
    if (!node.users.empty() || !node.dependencies.empty() || node.is_output())
        return;

    processing_order.erase(node.processing_itr);
    optimized_out.push_back(node.id());
    nodes_map.erase(node.id());
    
    if(node.is_input())
        inputs.remove(&node);
}

void program_impl::forward_bfs(std::function<void(program_node&)> const& mark_func, std::function<void(program_node&)> const& unmark_func) const
{
    if (!mark_func && !unmark_func)
        return;

    std::list<const std::list<program_node*>*> stack = { &inputs };
    while (!stack.empty())
    {
        auto nodes_list = stack.front();
        stack.pop_front();

        for (auto node : *nodes_list)
        {
            if (!node->is_marked())
            {
                node->mark();
                if (mark_func)
                    mark_func(*node);
                if (!node->get_users().empty())
                    stack.push_back(&node->get_users());
            }
        }
    }

    for (auto& node : nodes_map)
    {
        if (unmark_func)
            unmark_func(*node.second);
        node.second->unmark();
    }
}

void program_impl::backward_bfs(std::function<void(program_node&)> const& mark_func, std::function<void(program_node&)> const& unmark_func) const
{
    if (!mark_func && !unmark_func)
        return;

    std::list<const std::vector<program_node*>*> stack = { &outputs };
    while (!stack.empty())
    {
        auto nodes_list = stack.front();
        stack.pop_front();

        for (auto node : *nodes_list)
        {
            if (!node->is_marked())
            {
                node->mark();
                if (mark_func)
                    mark_func(*node);
                if (!node->get_dependencies().empty())
                    stack.push_back(&node->get_dependencies());
            }
        }
    }

    for (auto& node : nodes_map)
    {
        if (unmark_func)
            unmark_func(*node.second);
        node.second->unmark();
    }
}

}