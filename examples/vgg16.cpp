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



#include "common/common_tools.h"
#include "output_parser.h"
#include <iostream>
#include <string>
#include "api/instrumentation.h"

using namespace neural;

// AlexNet with weights & biases from file
std::vector<std::pair<primitive, std::string>> build_vgg16(const primitive& input, const primitive& output, const std::string& weights_dir)
{
	// [227x227x3xB] convolution->relu->pooling->lrn [1000xB]
	std::cout << "Building vgg16 started" << std::endl;
	instrumentation::timer<> timer_build;

	// create conversion to yxfb format and subtract mean values
	auto reordered_input = reorder::create(
	{
		memory::format::yxfb_f32,
		input.as<const memory&>().argument.size,
		input,
		file::create({ join_path(weights_dir, "imagenet_mean.nnd") })
	});

	auto conv1_1 = convolution::create(
	{
		memory::format::yxfb_f32,
		{
			reordered_input,
			file::create({ join_path(weights_dir, "conv1_1_weights.nnd") }),
			file::create({ join_path(weights_dir, "conv1_1_bias.nnd") }),
		},
		{ 0,{ -1, -1 }, 0 },
		{ 1,{ 1, 1 }, 1 },
		padding::zero,
		1,
		true });

	auto conv1_2 = convolution::create(
	{
		memory::format::yxfb_f32,
		{
			conv1_1,
			file::create({ join_path(weights_dir, "conv1_2_weights.nnd") }),
			file::create({ join_path(weights_dir, "conv1_2_bias.nnd") }),
		},
		{ 0,{ -1, -1 }, 0 },
		{ 1,{ 1, 1 }, 1 },
		padding::zero,
		1,
		true });


	auto pool1 = pooling::create(
	{
		pooling::mode::max,
		memory::format::yxfb_f32,
		conv1_2,
		{ 1,{ 2,2 },1 }, // strd
		{ 1,{ 2,2 },1 }, // kernel
		padding::zero
	});

	auto conv2_1 = convolution::create(
	{
		memory::format::yxfb_f32,
		{
			pool1,
			file::create({ join_path(weights_dir, "conv2_1_weights.nnd") }),
			file::create({ join_path(weights_dir, "conv2_1_bias.nnd") }),
		},
		{ 0,{ -1, -1 }, 0 },
		{ 1,{ 1, 1 }, 1 },
		padding::zero,
		1,
		true, // negative slope for RELU
	});

	auto conv2_2 = convolution::create(
	{
		memory::format::yxfb_f32,
		{
			conv2_1,
			file::create({ join_path(weights_dir, "conv2_2_weights.nnd") }),
			file::create({ join_path(weights_dir, "conv2_2_bias.nnd") }),
		},
		{ 0,{ -1, -1 }, 0 },
		{ 1,{ 1, 1 }, 1 },
		padding::zero,
		1,
		true, // negative slope for RELU
	});

	auto pool2 = pooling::create(
	{
		pooling::mode::max,
		memory::format::yxfb_f32,
		conv2_2,
		{ 1,{ 2,2 },1 }, // strd
		{ 1,{ 2,2 },1 }, // kernel
		padding::zero
	});

	auto conv3_1 = convolution::create(
	{
		memory::format::yxfb_f32,
		{
			pool2,
			file::create({ join_path(weights_dir, "conv3_1_weights.nnd") }),
			file::create({ join_path(weights_dir, "conv3_1_bias.nnd") }),
		},
		{ 0,{ -1, -1 }, 0 },
		{ 1,{ 1, 1 }, 1 },
		padding::zero,
		1,
		true,
	});

	auto conv3_2 = convolution::create(
	{
		memory::format::yxfb_f32,
		{
			conv3_1,
			file::create({ join_path(weights_dir, "conv3_2_weights.nnd") }),
			file::create({ join_path(weights_dir, "conv3_2_bias.nnd") }),
		},
		{ 0,{ -1, -1 }, 0 },
		{ 1,{ 1, 1 }, 1 },
		padding::zero,
		1,
		true,
	});

	auto conv3_3 = convolution::create(
	{
		memory::format::yxfb_f32,
		{
			conv3_2,
			file::create({ join_path(weights_dir, "conv3_3_weights.nnd") }),
			file::create({ join_path(weights_dir, "conv3_3_bias.nnd") }),
		},
		{ 0,{ -1, -1 }, 0 },
		{ 1,{ 1, 1 }, 1 },
		padding::zero,
		1,
		true,
	});

	auto pool3 = pooling::create(
	{
		pooling::mode::max,
		memory::format::yxfb_f32,
		conv3_3,
		{ 1,{ 2,2 },1 }, // strd
		{ 1,{ 2,2 },1 }, // kernel
		padding::zero
	});

	auto conv4_1 = convolution::create(
	{
		memory::format::yxfb_f32,
		{
			pool3,
			file::create({ join_path(weights_dir, "conv4_1_weights.nnd") }),
			file::create({ join_path(weights_dir, "conv4_1_bias.nnd") }),
		},
		{ 0,{ -1, -1 }, 0 },
		{ 1,{ 1, 1 }, 1 },
		padding::zero,
		1,
		true,
	});

	auto conv4_2 = convolution::create(
	{
		memory::format::yxfb_f32,
		{
			conv4_1,
			file::create({ join_path(weights_dir, "conv4_2_weights.nnd") }),
			file::create({ join_path(weights_dir, "conv4_2_bias.nnd") }),
		},
		{ 0,{ -1, -1 }, 0 },
		{ 1,{ 1, 1 }, 1 },
		padding::zero,
		1,
		true,
	});

	auto conv4_3 = convolution::create(
	{
		memory::format::yxfb_f32,
		{
			conv4_2,
			file::create({ join_path(weights_dir, "conv4_3_weights.nnd") }),
			file::create({ join_path(weights_dir, "conv4_3_bias.nnd") })
		},
		{ 0,{ -1, -1 }, 0 },
		{ 1,{ 1, 1 }, 1 },
		padding::zero,
		1,
		true,
	});

	auto pool4 = pooling::create(
	{
		pooling::mode::max,
		memory::format::yxfb_f32,
		conv4_3,
		{ 1,{ 2,2 },1 }, // strd
		{ 1,{ 2,2 },1 }, // kernel
		padding::zero
	});

	auto conv5_1 = convolution::create(
	{
		memory::format::yxfb_f32,
		{
			pool4,
			file::create({ join_path(weights_dir, "conv5_1_weights.nnd") }),
			file::create({ join_path(weights_dir, "conv5_1_bias.nnd") }),
		},
		{ 0,{ -1, -1 }, 0 },
		{ 1,{ 1,1 }, 1 },
		padding::zero,
		1,
		true,
	});

	auto conv5_2 = convolution::create(
	{
		memory::format::yxfb_f32,
		{
			conv5_1,
			file::create({ join_path(weights_dir, "conv5_2_weights.nnd") }),
			file::create({ join_path(weights_dir, "conv5_2_bias.nnd") }),
		},
		{ 0,{ -1, -1 }, 0 },
		{ 1,{ 1,1 }, 1 },
		padding::zero,
		1,
		true,
	});

	auto conv5_3 = convolution::create(
	{
		memory::format::yxfb_f32,
		{
			conv5_2,
			file::create({ join_path(weights_dir, "conv5_3_weights.nnd") }),
			file::create({ join_path(weights_dir, "conv5_3_bias.nnd") })
		},
		{ 0,{ -1, -1 }, 0 },
		{ 1,{ 1,1 }, 1 },
		padding::zero,
		1,
		true,
	});

	auto pool5 = pooling::create(
	{
		pooling::mode::max,
		memory::format::yxfb_f32,
		conv5_3,
		{ 1,{ 2,2 },1 }, // strd
		{ 1,{ 2,2 },1 }, // kernel
		padding::zero
	});

	auto fc6 = fully_connected::create(
	{
		memory::format::xb_f32,
		pool5,
		file::create({ join_path(weights_dir, "fc6_weights.nnd"), file::weights_type::fully_connected }),
		file::create({ join_path(weights_dir, "fc6_bias.nnd") }),
		true,
		0
	});

	auto fc7 = fully_connected::create(
	{
		memory::format::xb_f32,
		fc6,
		file::create({ join_path(weights_dir, "fc7_weights.nnd"), file::weights_type::fully_connected }),
		file::create({ join_path(weights_dir, "fc7_bias.nnd") }),
		true,
		0
	});

	auto fc8 = fully_connected::create(
	{
		memory::format::xb_f32,
		fc7,
		file::create({ join_path(weights_dir, "fc8_weights.nnd"), file::weights_type::fully_connected }),
		file::create({ join_path(weights_dir, "fc8_bias.nnd") }),
		true,
		0
	});

	auto softmax = normalization::softmax::create(
	{
		output,
		fc8
	});

	auto build_time = timer_build.uptime();
	std::cout << "Building VGG16 finished in " << instrumentation::to_string(build_time) << std::endl;

	return std::vector<std::pair<primitive, std::string>> {
		{ reordered_input, "reorder"},
		{ conv1_1, "conv1_1" },
		{ conv1_2, "conv1_2" },
		{ pool1, "pool1" },
		{ conv2_1, "conv2_1" },
		{ conv2_2, "conv2_2" },
		{ pool2, "pool2" },
		{ conv3_1, "conv3_1" },
		{ conv3_2, "conv3_2" },
		{ conv3_3, "conv3_3" },
		{ pool3, "pool3" },
		{ conv4_1, "conv4_1" },
		{ conv4_2, "conv4_2" },
		{ conv4_3, "conv4_3" },
		{ pool4, "pool4" },
		{ conv5_1, "conv5_1" },
		{ conv5_2, "conv5_2" },
		{ conv5_3, "conv5_3" },
		{ pool5, "pool5" },
		{ fc6, "fc6" },
		{ fc7, "fc7" },
		{ fc8, "fc8" },
		{ softmax, "softmax" }
	};
}


// AlexNet execution
std::chrono::nanoseconds execute_vgg16(const worker& worker, const std::vector<std::pair<primitive, std::string>>& primitives, const primitive& output, bool dump_hl)
{
	// we need this exact number of primitives(those are created in create_alexnet) 
	assert(primitives.size() == 23);

	std::cout << "Start execution" << std::endl;
	instrumentation::timer<> timer_execution;

	for (auto& p : primitives)
	{
		worker.execute(p.first.work());
	}

	//GPU primitives scheduled in unblocked manner
	auto scheduling_time(timer_execution.uptime());

	//OCL buffers mapping blocks until all primitives are completed
	output.as<const neural::memory&>().pointer<float>();

	auto execution_time(timer_execution.uptime());
	std::cout << "VGG16 scheduling finished in " << instrumentation::to_string(scheduling_time) << std::endl;
	std::cout << "VGG16 execution finished in " << instrumentation::to_string(execution_time) << std::endl;
	if (dump_hl)
	{
		instrumentation::logger::log_memory_to_file(primitives[0].first.input[0].primitive(), "input0");
		for (auto& p : primitives)
		{
			instrumentation::logger::log_memory_to_file(p.first, p.second);
		}
		// for now its enought. rest wil be done when we have equals those values
	}
	else
	{
		instrumentation::logger::log_memory_to_file(output, "final_result");
	}

	//print_profiling_table(std::cout, worker.as<worker_gpu&>().get_profiling_info());

	return std::chrono::duration_cast<std::chrono::nanoseconds>(execution_time);
}

void vgg16(uint32_t batch_size, std::string img_dir, const std::string& weights_dir, bool dump_hl, bool profiling)
{
	extern worker create_worker();
	extern uint32_t get_next_nearest_power_of_two(int number);
	extern uint32_t get_gpu_batch_size(int number);

	uint32_t gpu_batch_size = get_gpu_batch_size(batch_size);
	if (gpu_batch_size != batch_size)
	{
		std::cout << "WARNING: This is not the optimal batch size. You have " << (gpu_batch_size - batch_size)
			<< " dummy images per batch!!! Please use batch=" << gpu_batch_size << "." << std::endl;
	}
	gpu::configuration::get().enable_profiling = profiling;

	auto input = memory::allocate({ memory::format::byxf_f32,{ gpu_batch_size,{ 224, 224 }, 3, } });
	auto output = memory::allocate({ memory::format::xb_f32,{ gpu_batch_size,{ 1000 } } });

	auto img_list = get_directory_images(img_dir);
	if (img_list.empty())
		throw std::runtime_error("specified input images directory is empty (does not contain image data)");

	auto images_list_iterator = img_list.begin();
	auto images_list_end = img_list.end();

	auto number_of_batches = (img_list.size() % batch_size == 0)
		? img_list.size() / batch_size : img_list.size() / batch_size + 1;
	std::vector<std::string> image_in_batches;
	html output_file("VGG16", "VGG16 run");

	// build alexnet
	std::vector<std::pair<primitive, std::string>> alexnet_primitives = build_vgg16(input, output, weights_dir);

	// create worker
	worker worker = create_worker();

	for (decltype(number_of_batches) batch = 0; batch < number_of_batches; batch++)
	{
		image_in_batches.clear();
		for (uint32_t i = 0; i < batch_size && images_list_iterator != images_list_end; i++, images_list_iterator++)
			image_in_batches.push_back(*images_list_iterator);

		// load croped and resized images into input
		load_images_from_file_list(image_in_batches, input);

		// execute alexnet
		auto time = execute_vgg16(worker, alexnet_primitives, output, dump_hl);

		auto time_in_sec = std::chrono::duration_cast<std::chrono::duration<double, std::chrono::seconds::period>>(time).count();
		output_file.batch(output.as<const neural::memory&>(), join_path(get_executable_info()->dir(), "names.txt"), image_in_batches);
		if (time_in_sec != 0.0)
			std::cout << "Frames per second:" << (double)batch_size / time_in_sec << std::endl;
	}
}
