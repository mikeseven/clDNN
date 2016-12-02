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
#include <iostream>
#include <string>
#include <memory>
#include <chrono>
#include <api/engine.hpp>
#include <api/primitives/reorder.hpp>
#include <api/primitives/convolution.hpp>
#include <api/primitives/pooling.hpp>
#include <api/primitives/normalization.hpp>
#include <api/primitives/fully_connected.hpp>
#include <api/primitives/softmax.hpp>

// Commented out next line just to show proper types in example
//using namespace cldnn;

// Stubs of helper function and classes needed for later code
namespace helpers {

    struct file
    {
        enum weights_type {
            bias,
            convolution,
            fully_connected,
            mean
        };
    };

    std::string join_path(const std::string& base, const std::string& relative);

    cldnn::memory create_memory_from_file(cldnn::engine& engine, const std::string& path, file::weights_type type);
    void log_memory_to_file(const cldnn::memory& mem, const std::string& prefix);
    void print_profiling_table(std::ostream& os, const cldnn::network& network);

    template<typename T>
    void load_images_from_file_list(const std::vector<std::string>& images_in_batch, cldnn::memory mem);
    struct html
    {
        html(const std::string& file_name, const std::string& title);
        void batch(const cldnn::memory & mem,
            const std::string& categories_file,
            const std::vector<std::string>& image_names);
    };

    uint32_t get_gpu_batch_size(uint32_t batch_size);
    std::vector<std::string> get_directory_images(const std::string& img_dir);
    struct executable_info
    {
        std::string dir() const;
    };
    std::shared_ptr<const executable_info> get_executable_info();
    namespace instrumentation
    {
        template<class ClockTy = std::chrono::steady_clock>
        struct timer {
            typedef typename ClockTy::duration val_type;

            timer();
            val_type uptime();
        };

        template<class Rep, class Period>
        std::string to_string(const std::chrono::duration<Rep, Period> val);
    }
}


// Building AlexNet network with loading weights & biases from file
void define_alexnet(cldnn::topology& alexnet, cldnn::engine& engine, const cldnn::layout& input_layout, const std::string& weights_dir)
{
    // [227x227x3xB] convolution->relu->pooling->lrn [1000xB]
    std::cout << "Building Alexnet topology started" << std::endl;
    helpers::instrumentation::timer<> timer_build;

    alexnet.add_input("input", input_layout);
    //OR
    cldnn::input_layout input("input", input_layout);
    alexnet.add_primitive(input);
    //OR
    alexnet.add_primitive<cldnn::input_layout>({ "input", input_layout });

    // create conversion to yxfb format and subtract mean values
    cldnn::memory imagenet_mean_mem = helpers::create_memory_from_file(engine, helpers::join_path(weights_dir, "imagenet_mean.nnd"), helpers::file::mean);
    alexnet.add_data("imagenet_mean", imagenet_mean_mem);
    //OR
    cldnn::data imagenet_mean("imagenet_mean", imagenet_mean_mem);
    alexnet.add_primitive(imagenet_mean);

    cldnn::reorder reordered_input1("reordered_input", "input", cldnn::format::yxfb, "imagenet_mean");
    //OR
    cldnn::reorder reordered_input2("reordered_input", input, cldnn::format::yxfb, imagenet_mean);
    //OR
    alexnet.add_primitive<cldnn::reorder>({ "reordered_input", input, cldnn::format::yxfb, "imagenet_mean" });
    alexnet.add_primitive(cldnn::reorder("reordered_input", input, cldnn::format::yxfb, imagenet_mean));

    // create convolution
    cldnn::memory conv1_weights = helpers::create_memory_from_file(engine, helpers::join_path(weights_dir, "conv1_weights.nnd"), helpers::file::convolution);
    alexnet.add_data("conv1_weights", conv1_weights);
    cldnn::memory conv1_biases = helpers::create_memory_from_file(engine, helpers::join_path(weights_dir, "conv1_biases.nnd"), helpers::file::bias);
    alexnet.add_data("conv1_biases", conv1_biases);

    cldnn::convolution conv1(
        "conv1",
        "reordered_input",
        { "conv1_weights" },
        { "conv1_biases" },
        { cldnn::format::xy, {0, 0} },
        { cldnn::format::xy, {4, 4} },
        true
    );
    alexnet.add_primitive(conv1);

    //create pooling
    cldnn::pooling pool1(
        "pool1",
        "conv1",
        cldnn::pooling_mode::max,
        { cldnn::format::xy, {2, 2} },
        { cldnn::format::xy, {3, 3} }
    );
    alexnet.add_primitive(pool1);

    //learn
    cldnn::normalization lrn1{"lrn1", "pool1", 5, 1.0f, 0.00002f, 0.75f };
    alexnet.add_primitive(lrn1);


    //convolution with split
    cldnn::memory conv2_g1_weights = helpers::create_memory_from_file(engine, helpers::join_path(weights_dir, "conv2_g1_weights.nnd"), helpers::file::convolution);
    alexnet.add_data("conv2_g1_weights", conv2_g1_weights);
    cldnn::memory conv2_g1_biases = helpers::create_memory_from_file(engine, helpers::join_path(weights_dir, "conv2_g1_biases.nnd"), helpers::file::bias);
    alexnet.add_data("conv2_g1_biases", conv2_g1_biases);
    cldnn::memory conv2_g2_weights = helpers::create_memory_from_file(engine, helpers::join_path(weights_dir, "conv2_g2_weights.nnd"), helpers::file::convolution);
    alexnet.add_data("conv2_g2_weights", conv2_g2_weights);
    cldnn::memory conv2_g2_biases = helpers::create_memory_from_file(engine, helpers::join_path(weights_dir, "conv2_g2_biases.nnd"), helpers::file::bias);
    alexnet.add_data("conv2_g2_biases", conv2_g2_biases);

    cldnn::convolution conv2_group2(
        "conv2_group2",
        "lrn1",
        { "conv2_g1_weights", "conv2_g2_weights" },
        { "conv2_g1_biases", "conv2_g2_biases" },
        cldnn::tensor{ cldnn::format::xy, {-2, -2}},
        cldnn::tensor{ cldnn::format::xy, {1,1}},
        true,
        0.0f
    );
    alexnet.add_primitive(conv2_group2);

    //pooling #2
    cldnn::pooling pool2{
        "pool2",
        "conv2_group2",
        cldnn::pooling_mode::max,
        { cldnn::format::xy, {2, 2} },
        { cldnn::format::xy, {3, 3} }
    };
    alexnet.add_primitive(pool2);
    //OR
    // pool1.input = { "conv2_group2" };
    // alexnet.add_primitive("pool2", pool1);


    //learn #2
    cldnn::normalization lrn2{ "lrn2", "pool2", 5, 1.0f, 0.00002f, 0.75f };
    alexnet.add_primitive(lrn2);
    //OR
    // lrn1.input = { "pool2" };
    // alexnet.add_primitive("lrn2", lrn1);

    // convolution #3
    cldnn::memory conv3_weights = helpers::create_memory_from_file(engine, helpers::join_path(weights_dir, "conv3_weights.nnd"), helpers::file::convolution);
    alexnet.add_data("conv3_weights", conv3_weights);
    cldnn::memory conv3_biases = helpers::create_memory_from_file(engine, helpers::join_path(weights_dir, "conv3_biases.nnd"), helpers::file::bias);
    alexnet.add_data("conv3_biases", conv3_biases);

    cldnn::convolution conv3{
        "conv3",
        "lrn2",
        {"conv3_weights"},
        {"conv3_biases"},
        { cldnn::format::xy, {-1, -1}},
        { cldnn::format::xy, {1, 1}},
        true
    };
    alexnet.add_primitive(conv3);

    // convolution #4
    cldnn::memory conv4_g1_weights = helpers::create_memory_from_file(engine, helpers::join_path(weights_dir, "conv4_g1_weights.nnd"), helpers::file::convolution);
    alexnet.add_data("conv4_g1_weights", conv4_g1_weights);
    cldnn::memory conv4_g1_biases = helpers::create_memory_from_file(engine, helpers::join_path(weights_dir, "conv4_g1_biases.nnd"), helpers::file::bias);
    alexnet.add_data("conv4_g1_biases", conv4_g1_biases);
    cldnn::memory conv4_g2_weights = helpers::create_memory_from_file(engine, helpers::join_path(weights_dir, "conv4_g2_weights.nnd"), helpers::file::convolution);
    alexnet.add_data("conv4_g2_weights", conv4_g2_weights);
    cldnn::memory conv4_g2_biases = helpers::create_memory_from_file(engine, helpers::join_path(weights_dir, "conv4_g2_biases.nnd"), helpers::file::bias);
    alexnet.add_data("conv4_g2_biases", conv4_g2_biases);

    cldnn::convolution conv4_group2{
        "conv4_group2",
        conv3,
        { "conv4_g1_weights", "conv4_g2_weights" },
        { "conv4_g1_biases", "conv4_g2_biases" },
        { cldnn::format::xy, {-1, -1} },
        { cldnn::format::xy, {1, 1} },
        true
    };
    alexnet.add_primitive(conv4_group2);

    // convolution #5
    cldnn::memory conv5_g1_weights = helpers::create_memory_from_file(engine, helpers::join_path(weights_dir, "conv5_g1_weights.nnd"), helpers::file::convolution);
    alexnet.add_data("conv5_g1_weights", conv5_g1_weights);
    cldnn::memory conv5_g1_biases = helpers::create_memory_from_file(engine, helpers::join_path(weights_dir, "conv5_g1_biases.nnd"), helpers::file::bias);
    alexnet.add_data("conv5_g1_biases", conv5_g1_biases);
    cldnn::memory conv5_g2_weights = helpers::create_memory_from_file(engine, helpers::join_path(weights_dir, "conv5_g2_weights.nnd"), helpers::file::convolution);
    alexnet.add_data("conv5_g2_weights", conv5_g2_weights);
    cldnn::memory conv5_g2_biases = helpers::create_memory_from_file(engine, helpers::join_path(weights_dir, "conv5_g2_biases.nnd"), helpers::file::bias);
    alexnet.add_data("conv5_g2_biases", conv5_g2_biases);

    alexnet.add_primitive<cldnn::convolution>({
        "conv5_group2",
        "conv4_group2",
        { "conv5_g1_weights", "conv5_g2_weights" },
        { "conv5_g1_biases", "conv5_g2_biases" },
        { cldnn::format::xy,{ -1, -1 } },
        { cldnn::format::xy,{ 1, 1 } },
        true
    });

    //pooling #5
    cldnn::pooling pool5{
        "pool5",
        "conv5_group2",
        cldnn::pooling_mode::max,
        { cldnn::format::xy, {2, 2} },
        { cldnn::format::xy, {3, 3} }
    };
    alexnet.add_primitive(pool5);


    // fully connected #6
    cldnn::memory fc6_weights = helpers::create_memory_from_file(engine, helpers::join_path(weights_dir, "fc6_weights.nnd"), helpers::file::fully_connected);
    alexnet.add_data("fc6_weights", fc6_weights);
    cldnn::memory fc6_biases = helpers::create_memory_from_file(engine, helpers::join_path(weights_dir, "fc6_biases.nnd"), helpers::file::bias);
    alexnet.add_data("fc6_biases", fc6_biases);

    alexnet.add_primitive<cldnn::fully_connected>({
        "fc6",
        pool5,
        { "fc6_weights" },
        { "fc6_biases" },
        true
    });

    // fully connected #7
    cldnn::memory fc7_weights = helpers::create_memory_from_file(engine, helpers::join_path(weights_dir, "fc7_weights.nnd"), helpers::file::fully_connected);
    alexnet.add_data("fc7_weights", fc7_weights);
    cldnn::memory fc7_biases = helpers::create_memory_from_file(engine, helpers::join_path(weights_dir, "fc7_biases.nnd"), helpers::file::bias);
    alexnet.add_data("fc7_biases", fc7_biases);

    cldnn::fully_connected fc7{
        "fc7",
        "fc6",
        { "fc7_weights" },
        { "fc7_biases" },
        true
    };
    alexnet.add_primitive(fc7);

    // fully connected #7
    cldnn::memory fc8_weights = helpers::create_memory_from_file(engine, helpers::join_path(weights_dir, "fc8_weights.nnd"), helpers::file::fully_connected);
    alexnet.add_data("fc8_weights", fc8_weights);
    cldnn::memory fc8_biases = helpers::create_memory_from_file(engine, helpers::join_path(weights_dir, "fc8_biases.nnd"), helpers::file::bias);
    alexnet.add_data("fc8_biases", fc8_biases);

    cldnn::fully_connected fc8{
        "fc8",
        "fc7",
        { "fc8_weights" },
        { "fc8_biases" },
        true
    };
    alexnet.add_primitive(fc8);

    alexnet.add_primitive(cldnn::softmax{ "softmax", "fc8" });

    auto build_time = timer_build.uptime();
    std::cout << "Building Alexnet topology finished in " << helpers::instrumentation::to_string(build_time) << std::endl;
}

std::chrono::nanoseconds execute_network(cldnn::network& network, bool dump_hl, const std::string& topology, const std::string& output_primitive_key)
{
    std::cout << "Start execution" << std::endl;
    helpers::instrumentation::timer<> timer_execution;

    cldnn::event complete_event = network.execute({/*empty dependencies array*/});

    //GPU primitives scheduled in unblocked manner
    auto scheduling_time(timer_execution.uptime());

    // wait for network execution completion
    complete_event.wait();

    auto execution_time(timer_execution.uptime());
    std::cout << topology << " scheduling finished in " << helpers::instrumentation::to_string(scheduling_time) << std::endl;
    std::cout << topology << " execution finished in " << helpers::instrumentation::to_string(execution_time) << std::endl;
    if (dump_hl)
    {
        cldnn::array_ref<cldnn::primitive_id> primitives = network.primitive_keys();
        for (auto& p : primitives)
        {
            const cldnn::memory& out = network.get_output(p);
            helpers::log_memory_to_file(out, p.c_str());
        }
        // for now its enough. rest will be done when we have equals those values
    }
    else
    {
        const cldnn::memory& output = network.get_output(output_primitive_key);
        helpers::log_memory_to_file(output, "final_result");
    }

    helpers::print_profiling_table(std::cout, network);

    return std::chrono::duration_cast<std::chrono::nanoseconds>(execution_time);
}

void alexnet(uint32_t batch_size, std::string img_dir, const std::string& weights_dir, bool dump_hl, bool profiling, bool optimize_weights, bool use_half)
{
    int32_t gpu_batch_size = helpers::get_gpu_batch_size(batch_size);
    if (gpu_batch_size != batch_size)
    {
        std::cout << "WARNING: This is not the optimal batch size. You have " << (gpu_batch_size - batch_size)
            << " dummy images per batch!!! Please use batch=" << gpu_batch_size << "." << std::endl;
    }

    auto img_list = helpers::get_directory_images(img_dir);
    if (img_list.empty())
        throw std::runtime_error("specified input images directory is empty (does not contain image data)");

    auto number_of_batches = (img_list.size() % batch_size == 0)
        ? img_list.size() / batch_size : img_list.size() / batch_size + 1;

    helpers::html output_file("alexnet", "alexnet run");

    cldnn::layout input_layout{
        use_half ? cldnn::data_types::f16 : cldnn::data_types::f32,
        { cldnn::format::byxf, { gpu_batch_size, 227, 227, 3}}
    };

    // build alexnet
    cldnn::topology alexnet_topology = cldnn::topology::create();
    cldnn::engine engine = cldnn::engine::create({ profiling });
    define_alexnet(alexnet_topology, engine, input_layout, weights_dir);
    cldnn::network alexnet = engine.build_network(alexnet_topology);

    cldnn::memory input = engine.allocate_memory(input_layout);

    std::vector<std::string> images_in_batch;
    auto images_list_iterator = img_list.begin();
    auto images_list_end = img_list.end();
    for (decltype(number_of_batches) batch = 0; batch < number_of_batches; batch++)
    {
        images_in_batch.clear();
        for (uint32_t i = 0; i < batch_size && images_list_iterator != images_list_end; ++i, ++images_list_iterator)
        {
            images_in_batch.push_back(*images_list_iterator);
        }

        // load croped and resized images into input
        if (use_half)
        {
            helpers::load_images_from_file_list<half_t>(images_in_batch, input);
        }
        else
        {
            helpers::load_images_from_file_list<float>(images_in_batch, input);
        }

        // execute alexnet

        alexnet.set_input_data("input", input);
        auto time = execute_network(alexnet, dump_hl, "alexnet", "softmax");

        auto time_in_sec = std::chrono::duration_cast<std::chrono::duration<double, std::chrono::seconds::period>>(time).count();
        output_file.batch(alexnet.get_output("softmax"), helpers::join_path(helpers::get_executable_info()->dir(), "names.txt"), images_in_batch);
        if (time_in_sec != 0.0)
        {
            std::cout << "Frames per second:" << static_cast<double>(batch_size) / time_in_sec << std::endl;
        }
    }
}

