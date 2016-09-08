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
#include "memory_utils.h"
#include "implementation_map.h"
#include <algorithm>
#include <numeric>
#include <tuple>
#include <map>
#include <functional>
#include <atomic>

namespace neural
{

namespace
{

// maps of available strides for specific formats
static std::map< std::tuple<memory::format::type, bool>, std::tuple<std::vector<uint32_t>, std::vector<uint32_t>, std::vector<uint32_t>>> format_strides_map =
{
      // raw sizes: b,f, {x,y}                 spatial                                  spatial_stride         single_average_stride             batch_stride
    { std::make_tuple(memory::format::yxfb_f32, true),  std::make_tuple(std::vector<uint32_t>{0,1}, std::vector<uint32_t>{0},     std::vector<uint32_t>{}) },
    { std::make_tuple(memory::format::byxf_f32, true),  std::make_tuple(std::vector<uint32_t>{0},   std::vector<uint32_t>{},      std::vector<uint32_t>{1,2,3}) },
    { std::make_tuple(memory::format::fyxb_f32, true),  std::make_tuple(std::vector<uint32_t>{},    std::vector<uint32_t>{0},     std::vector<uint32_t>{}) },
    { std::make_tuple(memory::format::bfyx_f32, true),  std::make_tuple(std::vector<uint32_t>{},    std::vector<uint32_t>{2,3},   std::vector<uint32_t>{1,2,3}) },
    { std::make_tuple(memory::format::yxfb_f32, false), std::make_tuple(std::vector<uint32_t>{},    std::vector<uint32_t>{0},     std::vector<uint32_t>{}) },
    { std::make_tuple(memory::format::byxf_f32, false), std::make_tuple(std::vector<uint32_t>{},    std::vector<uint32_t>{},      std::vector<uint32_t>{1,2,3}) },
    { std::make_tuple(memory::format::fyxb_f32, false), std::make_tuple(std::vector<uint32_t>{},    std::vector<uint32_t>{0},     std::vector<uint32_t>{}) },
    { std::make_tuple(memory::format::bfyx_f32, false), std::make_tuple(std::vector<uint32_t>{},    std::vector<uint32_t>{2,3},   std::vector<uint32_t>{1,2,3}) }
};


template <class T>
struct batch_normalization_training_forward_reference : is_an_implementation {

    struct request_data
    {
        const normalization::batch_training_forward* primitive;
        mutable std::atomic_int* minibatch_counter;
    };

    const normalization::batch_training_forward &outer;
    std::atomic_int minibatch_counter;

    std::vector<request_data> requests;

    batch_normalization_training_forward_reference(normalization::batch_training_forward &arg)
        : is_an_implementation(neural::type_id<batch_normalization_training_forward_reference>())
        , outer(arg)
        , minibatch_counter(0) {}

    ~batch_normalization_training_forward_reference() {}

    task_group work()
    {
        requests.push_back({&outer, &minibatch_counter});
        return {{task{implementation, &requests[0]}}, schedule::single};
    }

    static is_an_implementation *create(normalization::batch_training_forward &arg) { return new batch_normalization_training_forward_reference(arg); };

    static void implementation(const void *ptr)
    {
        const auto request = reinterpret_cast<const request_data *>(ptr);
        const auto this_bn = static_cast<const normalization::batch_training_forward *>(request->primitive);

        if(this_bn->output().size() < 3 ||
           this_bn->output().size() > 5 ||
           this_bn->input().size() != 3)
            throw std::runtime_error("batch_normalization_training_forward_reference::implementation -> incorrect number of BatchNorm input/outputs.");

        const auto spatial = this_bn->argument.spatial;
        const T exp_avg_factor = static_cast<T>(this_bn->argument.exp_avg_factor);

        // Cast double->T.
        const T epsilon = static_cast<T>(this_bn->argument.epsilon);

        auto& input = this_bn->input_memory(0);
        auto& scale = this_bn->input_memory(1);
        auto& bias  = this_bn->input_memory(2);

        auto& output              = this_bn->argument.output[0].as<const memory&>();
        auto& current_mean        = this_bn->argument.output[1].as<const memory&>();
        auto& current_inv_std_dev = this_bn->argument.output[2].as<const memory&>();

        auto input_buffer = input.pointer<T>();
        auto scale_buffer = scale.pointer<T>();
        auto bias_buffer  = bias.pointer<T>();

        auto output_buffer              = output.pointer<T>();
        auto current_mean_buffer        = current_mean.pointer<T>();
        auto current_inv_std_dev_buffer = current_inv_std_dev.pointer<T>();

        if(output.argument.format != input.argument.format)
            throw std::runtime_error("batch_normalization_training_forward_reference::implementation -> io format doesn't match.");

        auto it = format_strides_map.find(std::make_tuple(input.argument.format, spatial));
        if(it==std::end(format_strides_map)) throw std::runtime_error("batch_normalization_training_forward_reference::implementation -> unknown BatchNorm format");

        auto data_w = input.argument.size.raw[2];
        auto data_h = input.argument.size.raw[3];
        auto data_c = input.argument.size.raw[1];
        auto data_n = input.argument.size.raw[0];

        auto spatial_location_stride = 1;
        auto element_stride = 1;
        auto batch_stride = 1;

        for (auto index : std::get<0>(it->second)) spatial_location_stride *= input.argument.size.raw[index];
        for (auto index : std::get<1>(it->second)) element_stride *= input.argument.size.raw[index];
        for (auto index : std::get<2>(it->second)) batch_stride *= input.argument.size.raw[index];

        const auto spatial_size = (spatial) ? data_w * data_h : 1;
        const auto num_averages = (spatial) ? data_c : data_c * data_w * data_h;
        const T inv_num_average_over = static_cast<T>(1.0 / (data_n * spatial_size));

        fill<T>(current_mean, 0);
        fill<T>(current_inv_std_dev, 0);

        auto compute_io_data_offset = [&element_stride, &batch_stride, &spatial_location_stride](int element, int batch, int spatial_location)
        {
            return spatial_location * spatial_location_stride + element * element_stride + batch * batch_stride;
        };

        // Compute spatial/batch average. (can be MT over element)
        for(uint32_t element = 0; element < num_averages; ++element)
            for(uint32_t batch = 0; batch < data_n; ++batch)
                for(uint32_t spatial_location = 0; spatial_location < spatial_size; ++spatial_location)
                    current_mean_buffer[element] +=
                        input_buffer[compute_io_data_offset(element, batch, spatial_location)] * inv_num_average_over;

        // Compute spatial/batch variance. (can be MT over element)
        for(uint32_t element = 0; element < num_averages; ++element)
            for(uint32_t batch = 0; batch < data_n; ++batch)
                for(uint32_t spatial_location = 0; spatial_location < spatial_size; ++spatial_location)
                    current_inv_std_dev_buffer[element] +=
                        std::pow(input_buffer[compute_io_data_offset(element, batch, spatial_location)] - current_mean_buffer[element], 2.0f) * inv_num_average_over;

        // Compute spatial/batch inverse standard deviation. (can be MT over element)
        for(uint32_t element = 0; element < current_inv_std_dev.count(); ++element)
            current_inv_std_dev_buffer[element] = std::pow(current_inv_std_dev_buffer[element] + epsilon, -0.5f);

        // Normalize data. (can be MT over element)
        for(uint32_t element = 0; element < num_averages; ++element)
            for(uint32_t batch = 0; batch < data_n; ++batch)
                for(uint32_t spatial_location = 0; spatial_location < spatial_size; ++spatial_location)
                    output_buffer[compute_io_data_offset(element, batch, spatial_location)] =
                          (input_buffer[compute_io_data_offset(element, batch, spatial_location)] - current_mean_buffer[element])
                        * current_inv_std_dev_buffer[element]
                        * scale_buffer[element]
                        + bias_buffer[element];

        // Compute and save moving averages.
        if(this_bn->output().size() > 3)
        {
            //auto& moving_mean = this_bn->output_memory(3);
            auto& moving_mean = this_bn->argument.output[3].as<const memory&>();
            auto moving_mean_buffer = moving_mean.pointer<T>();

            // For first run, set data to zero.
            if(*request->minibatch_counter == 0) fill<T>(moving_mean, 0);

            // Compute avg factor for moving average basing on number of already computed averages [factor<=0], or using user provided [factor>0] factor.
            T actual_exp_avg_factor = (exp_avg_factor > 0.0f) ? exp_avg_factor : 1.0f / (1.0f + *request->minibatch_counter);

            // Compute moving average. (can be MT over elements)
            for(uint32_t element = 0; element < moving_mean.count(); ++element)
                moving_mean_buffer[element] = current_mean_buffer[element] * actual_exp_avg_factor + moving_mean_buffer[element] * (1.0f - actual_exp_avg_factor);
        }

        if(this_bn->output().size() > 4)
        {
            //auto& moving_inv_std_dev = this_bn->output_memory(4);
            auto& moving_inv_std_dev = this_bn->argument.output[4].as<const memory&>();
            auto moving_inv_std_dev_buffer = moving_inv_std_dev.pointer<T>();

            // For first run, set data to zero.
            if(*request->minibatch_counter == 0) fill<T>(moving_inv_std_dev, 0);

            // Compute avg factor for moving average basing on number of already computed averages [factor<=0], or using user provided [factor>0] factor.
            T actual_exp_avg_factor = (exp_avg_factor > 0.0f) ? exp_avg_factor : 1.0f / (1.0f + *request->minibatch_counter);

            // Compute moving inv std dev. (can be MT over elements)
            for(uint32_t element = 0; element < moving_inv_std_dev.count(); ++element)
                moving_inv_std_dev_buffer[element] = current_inv_std_dev_buffer[element] * actual_exp_avg_factor + moving_inv_std_dev_buffer[element] * (1.0f - actual_exp_avg_factor);
        }
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-value"
#endif
        *request->minibatch_counter++;
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
    }
};


template <class T>
struct batch_normalization_inference_reference : is_an_implementation {

    struct request_data
    {
        const normalization::batch_inference* primitive;
    };

    const normalization::batch_inference &outer;

    std::vector<request_data> requests;

    batch_normalization_inference_reference(normalization::batch_inference &arg)
        : is_an_implementation(neural::type_id<batch_normalization_inference_reference>())
        , outer(arg) {}

    ~batch_normalization_inference_reference() {}

    task_group work()
    {
        requests.push_back({&outer});
        return {{task{implementation, &requests[0]}}, schedule::single};
    }

    static is_an_implementation *create(normalization::batch_inference &arg) { return new batch_normalization_inference_reference(arg); };

    static void implementation(const void *ptr)
    {
        const auto request = reinterpret_cast<const request_data *>(ptr);
        const auto this_bn = static_cast<const normalization::batch_inference *>(request->primitive);

        if(this_bn->input().size() != 5 ||
           this_bn->output().size() != 1)
            throw std::runtime_error("batch_normalization_inference_reference::implementation -> incorrect number of BatchNorm input/outputs.");

        const auto spatial = this_bn->argument.spatial;

        auto& input       = this_bn->input_memory(0);
        auto& scale       = this_bn->input_memory(1);
        auto& bias        = this_bn->input_memory(2);
        auto& mean        = this_bn->input_memory(3);
        auto& inv_std_dev = this_bn->input_memory(4);

        auto& output = this_bn->argument.output[0].as<const memory&>();

        auto input_buffer = input.pointer<T>();
        auto scale_buffer = scale.pointer<T>();
        auto bias_buffer = (bias.pointer<T>());
        auto mean_buffer = (mean.pointer<T>());
        auto inv_std_dev_buffer = (inv_std_dev.pointer<T>());

        auto output_buffer = (output.pointer<T>());


        if(output.argument.format != input.argument.format)
            throw std::runtime_error("batch_normalization_inference_reference::implementation -> io format doesn't match.");

        auto it = format_strides_map.find(std::make_tuple(input.argument.format, spatial));
        if(it==std::end(format_strides_map)) throw std::runtime_error("batch_normalization_inference_reference::implementation -> unknown BatchNorm format");

        auto data_w = input.argument.size.raw[2];
        auto data_h = input.argument.size.raw[3];
        auto data_c = input.argument.size.raw[1];
        auto data_n = input.argument.size.raw[0];

        auto spatial_location_stride = 1;
        auto element_stride = 1;
        auto batch_stride = 1;

        for (auto index : std::get<0>(it->second)) spatial_location_stride *= input.argument.size.raw[index];
        for (auto index : std::get<1>(it->second)) element_stride *= input.argument.size.raw[index];
        for (auto index : std::get<2>(it->second)) batch_stride *= input.argument.size.raw[index];

        const auto spatial_size = (spatial) ? data_w * data_h : 1;
        const auto num_averages = (spatial) ? data_c : data_c * data_w * data_h;

        auto compute_io_data_offset = [&element_stride, &batch_stride, &spatial_location_stride](int element, int batch, int spatial_location)
        {
            return spatial_location * spatial_location_stride + element * element_stride + batch * batch_stride;
        };

        // Normalize data. (can be MT over element)
        for(uint32_t element = 0; element < num_averages; ++element)
            for(uint32_t batch = 0; batch < data_n; ++batch)
                for(uint32_t spatial_location = 0; spatial_location < spatial_size; ++spatial_location)
                    output_buffer[compute_io_data_offset(element, batch, spatial_location)] =
                    (input_buffer[compute_io_data_offset(element, batch, spatial_location)] - mean_buffer[element])
                    * inv_std_dev_buffer[element]
                    * scale_buffer[element]
                    + bias_buffer[element];
    }
};


struct attach {
    attach() {
        implementation_map<normalization::batch_training_forward>::add({
        { std::make_tuple(engine::reference, memory::format::yxfb_f32, memory::format::yxfb_f32), batch_normalization_training_forward_reference<float>::create },
        { std::make_tuple(engine::reference, memory::format::byxf_f32, memory::format::byxf_f32), batch_normalization_training_forward_reference<float>::create },
        { std::make_tuple(engine::reference, memory::format::fyxb_f32, memory::format::fyxb_f32), batch_normalization_training_forward_reference<float>::create },
        { std::make_tuple(engine::reference, memory::format::bfyx_f32, memory::format::bfyx_f32), batch_normalization_training_forward_reference<float>::create },
        });

        implementation_map<normalization::batch_inference>::add({
        { std::make_tuple(engine::reference, memory::format::yxfb_f32, memory::format::yxfb_f32), batch_normalization_inference_reference<float>::create },
        { std::make_tuple(engine::reference, memory::format::byxf_f32, memory::format::byxf_f32), batch_normalization_inference_reference<float>::create },
        { std::make_tuple(engine::reference, memory::format::fyxb_f32, memory::format::fyxb_f32), batch_normalization_inference_reference<float>::create },
        { std::make_tuple(engine::reference, memory::format::bfyx_f32, memory::format::bfyx_f32), batch_normalization_inference_reference<float>::create }
        });
    }
    ~attach() {}
};

#ifdef __GNUC__
    __attribute__((visibility("default"))) //todo meybe dll_sym?
#elif _MSC_VER
#   pragma section(".nn_init$m", read, write)
#endif
    attach attach_impl;

} // namespace {


namespace normalization {
batch_training_forward::arguments::arguments( neural::engine::type engine, std::vector<primitive> output, std::vector<primitive_at> input, bool spatial, double exp_avg_factor, double epsilon)
    : engine(engine)
    , output(output)
    , input(input)
    , spatial(spatial)
    , exp_avg_factor(exp_avg_factor)
    , epsilon(epsilon) {}

// creates primitive with batch_normalization implementation that supports provided arguments
primitive batch_training_forward::create(batch_training_forward::arguments arg)
{
    return is_a_primitive::create<batch_training_forward>(arg);
}

batch_inference::arguments::arguments( neural::engine::type engine, std::vector<primitive> output, std::vector<primitive_at> input, bool spatial)
    : engine(engine)
    , output(output)
    , input(input)
    , spatial(spatial) {}

// creates primitive with batch_normalization implementation that supports provided arguments
primitive batch_inference::create(batch_inference::arguments arg)
{
    return is_a_primitive::create<batch_inference>(arg);
}

} // namespace normalization

} // namespace neural