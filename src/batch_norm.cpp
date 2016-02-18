#include "neural.h"
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

template <class T>
struct batch_normalization_training_forward_reference : is_an_implementation {

    struct request_data
    {
        const batch_normalization_training_forward* primitive;
        mutable std::atomic_int* minibatch_counter;
    };

    const batch_normalization_training_forward &outer;
    std::atomic_int minibatch_counter;

    std::vector<request_data> requests;

    batch_normalization_training_forward_reference(batch_normalization_training_forward &arg)
        : is_an_implementation(neural::type_id<batch_normalization_training_forward_reference>()) 
        , outer(arg)
        , minibatch_counter(0) {}

    ~batch_normalization_training_forward_reference() {}

    std::vector<task> work() 
    {
        requests.push_back({&outer, &minibatch_counter});
        return {task{implementation, &requests[0]}};
    }

    static is_an_implementation *create(batch_normalization_training_forward &arg) { return new batch_normalization_training_forward_reference(arg); };
    
    static void implementation(const void *ptr) 
    {
        const auto request = reinterpret_cast<const request_data *>(ptr);
        const auto this_bn = static_cast<const batch_normalization_training_forward *>(request->primitive);

        if(this_bn->output().size() < 3 || 
           this_bn->output().size() > 5 ||
           this_bn->input().size() != 3)
            throw std::runtime_error("Incorrect number of BatchNorm input/outputs.");

        const auto spatial = this_bn->argument.spatial;
        const T exp_avg_factor = static_cast<T>(this_bn->argument.exp_avg_factor);

        // Cast double->T.
        const T epsilon = static_cast<T>(this_bn->argument.epsilon);

        auto& input = this_bn->input_memory(0);
        auto& scale = this_bn->input_memory(1);
        auto& bias = this_bn->input_memory(2);

        auto& output = this_bn->output_memory(0);
        auto& current_mean = this_bn->output_memory(1);
        auto& current_inv_std_dev = this_bn->output_memory(2);

        auto input_buffer = static_cast<T*>(input.pointer);
        auto scale_buffer = static_cast<T*>(scale.pointer);
        auto bias_buffer = static_cast<T*>(bias.pointer);

        auto output_buffer = static_cast<T*>(output.pointer);
        auto current_mean_buffer = static_cast<T*>(current_mean.pointer);
        auto current_inv_std_dev_buffer = static_cast<T*>(current_inv_std_dev.pointer);

        if(output.argument.format != input.argument.format)
            throw std::runtime_error("Output format doesn't match.");

        const auto format = input.argument.format;
        const auto data_size = memory::traits(format).type->size;

        auto data_w = 0;
        auto data_h = 0;
        auto data_c = 0;
        auto data_n = 0;

        auto spatial_location_stride = 0;
        auto element_stride = 0;
        auto batch_stride = 0;

        if(format == memory::format::yxfb_f32 ||
           format == memory::format::yxfb_f64)
        {
            data_h = input.argument.size[0];
            data_w = input.argument.size[1];
            data_c = input.argument.size[2];
            data_n = input.argument.size[3];

            spatial_location_stride = data_n * data_c;
            element_stride = data_n;
            batch_stride = 1;
        }
        else if(format == memory::format::xyfb_f32 ||
                format == memory::format::xyfb_f64)
        {
            data_w = input.argument.size[0];
            data_h = input.argument.size[1];
            data_c = input.argument.size[2];
            data_n = input.argument.size[3];

            spatial_location_stride = data_n * data_c;
            element_stride = data_n;
            batch_stride = 1;
        }
        else if(format == memory::format::fyxb_f32 ||
                format == memory::format::fyxb_f64)
        {
            data_c = input.argument.size[0];
            data_h = input.argument.size[1];
            data_w = input.argument.size[2];
            data_n = input.argument.size[3];

            spatial_location_stride = data_n;
            element_stride = data_n * data_w * data_h;
            batch_stride = 1;
        }
        else if(format == memory::format::fxyb_f32 ||
                format == memory::format::fxyb_f64)
        {
            data_c = input.argument.size[0];
            data_w = input.argument.size[1];
            data_h = input.argument.size[2];
            data_n = input.argument.size[3];

            spatial_location_stride = data_n;
            element_stride = data_n * data_w * data_h;
            batch_stride = 1;
        }
        else if(format == memory::format::byxf_f32 ||
                format == memory::format::byxf_f64)
        {
            data_n = input.argument.size[0];
            data_h = input.argument.size[1];
            data_w = input.argument.size[2];
            data_c = input.argument.size[3];

            spatial_location_stride = data_c;
            element_stride = 1;
            batch_stride = data_c * data_w * data_h;
        }
        else if(format == memory::format::bxyf_f32 ||
                format == memory::format::bxyf_f64)
        {
            data_n = input.argument.size[0];
            data_w = input.argument.size[1];
            data_h = input.argument.size[2];
            data_c = input.argument.size[3];

            spatial_location_stride = data_c;
            element_stride = 1;
            batch_stride = data_c * data_w * data_h;
        }
        else if(format == memory::format::bfyx_f32 ||
                format == memory::format::bfyx_f64)
        {
            data_n = input.argument.size[0];
            data_c = input.argument.size[1];
            data_h = input.argument.size[2];
            data_w = input.argument.size[3];

            spatial_location_stride = 1;
            element_stride = data_w * data_h;
            batch_stride = data_c * data_w * data_h;
        }
        else if(format == memory::format::bfxy_f32 ||
                format == memory::format::bfxy_f64)
        {
            data_n = input.argument.size[0];
            data_c = input.argument.size[1];
            data_w = input.argument.size[2];
            data_h = input.argument.size[3];

            spatial_location_stride = 1;
            element_stride = data_w * data_h;
            batch_stride = data_c * data_w * data_h;
        }
        else
        {
            throw std::runtime_error("Unknown BatchNorm format.");
        }

        const auto spatial_size = (spatial) ? data_w * data_h : 1;
        const auto num_averages = (spatial) ? data_c : data_c * data_w * data_h;
        const T inv_num_average_over = static_cast<T>(1.0 / (data_n * spatial_size));

        memset(current_mean_buffer, 0, current_mean.count() * data_size);
        memset(current_inv_std_dev_buffer, 0, current_inv_std_dev.count() * data_size);

        auto compute_io_data_offset = [&element_stride, &batch_stride, &spatial_location_stride](int element, int batch, int spatial_location)
        {
            return spatial_location * spatial_location_stride + element * element_stride + batch * batch_stride;
        };

        // Compute spatial/batch average. (can be MT over element)
        for(int element = 0; element < num_averages; ++element)
            for(int batch = 0; batch < data_n; ++batch)
                for(int spatial_location = 0; spatial_location < spatial_size; ++spatial_location)
                    current_mean_buffer[element] += 
                        input_buffer[compute_io_data_offset(element, batch, spatial_location)] * inv_num_average_over;

        // Compute spatial/batch variance. (can be MT over element)
        for(int element = 0; element < num_averages; ++element)
            for(int batch = 0; batch < data_n; ++batch)
                for(int spatial_location = 0; spatial_location < spatial_size; ++spatial_location)
                    current_inv_std_dev_buffer[element] += 
                        std::pow(input_buffer[compute_io_data_offset(element, batch, spatial_location)] - current_mean_buffer[element], 2.0f) * inv_num_average_over;

        // Compute spatial/batch inverse standard deviation. (can be MT over element)
        for(int element = 0; element < current_inv_std_dev.count(); ++element)
            current_inv_std_dev_buffer[element] = std::pow(current_inv_std_dev_buffer[element] + epsilon, -0.5f);
          
        // Normalize data. (can be MT over element)
        for(int element = 0; element < num_averages; ++element)
            for(int batch = 0; batch < data_n; ++batch)
                for(int spatial_location = 0; spatial_location < spatial_size; ++spatial_location)
                    output_buffer[compute_io_data_offset(element, batch, spatial_location)] = 
                          (input_buffer[compute_io_data_offset(element, batch, spatial_location)] - current_mean_buffer[element])
                        * current_inv_std_dev_buffer[element]
                        * scale_buffer[element]
                        + bias_buffer[element];

        // Compute and save moving averages.
        if(this_bn->output().size() > 3)
        {
            auto& moving_mean = this_bn->output_memory(3);
            auto moving_mean_buffer = static_cast<T*>(moving_mean.pointer);

            // For first run, set data to zero.
            if(*request->minibatch_counter == 0) memset(moving_mean_buffer, 0, moving_mean.count() * data_size);

            // Compute avg factor for moving average basing on number of already computed averages [factor<=0], or using user provided [factor>0] factor.
            T actual_exp_avg_factor = (exp_avg_factor > 0.0f) ? exp_avg_factor : 1.0f / (1.0f + *request->minibatch_counter);

            // Compute moving average. (can be MT over elements)
            for(int element = 0; element < moving_mean.count(); ++element)
                moving_mean_buffer[element] = current_mean_buffer[element] * actual_exp_avg_factor + moving_mean_buffer[element] * (1.0f - actual_exp_avg_factor);
        }

        if(this_bn->output().size() > 4)
        {
            auto& moving_inv_std_dev = this_bn->output_memory(4);
            auto moving_inv_std_dev_buffer = static_cast<T*>(moving_inv_std_dev.pointer);

            // For first run, set data to zero.
            if(*request->minibatch_counter == 0) memset(moving_inv_std_dev_buffer, 0, moving_inv_std_dev.count() * data_size);

            // Compute avg factor for moving average basing on number of already computed averages [factor<=0], or using user provided [factor>0] factor.
            T actual_exp_avg_factor = (exp_avg_factor > 0.0f) ? exp_avg_factor : 1.0f / (1.0f + *request->minibatch_counter);

            // Compute moving inv std dev. (can be MT over elements)
            for(int element = 0; element < moving_inv_std_dev.count(); ++element)
                moving_inv_std_dev_buffer[element] = current_inv_std_dev_buffer[element] * actual_exp_avg_factor + moving_inv_std_dev_buffer[element] * (1.0f - actual_exp_avg_factor);
        }

        *request->minibatch_counter++;
    }
};

template <class T>
struct batch_normalization_training_backward_reference : is_an_implementation {

    struct request_data
    {
        const batch_normalization_training_backward* primitive;
    };

    const batch_normalization_training_backward &outer;

    std::vector<request_data> requests;

    batch_normalization_training_backward_reference(batch_normalization_training_backward &arg)
        : is_an_implementation(neural::type_id<batch_normalization_training_backward_reference>()) 
        , outer(arg) {}

    ~batch_normalization_training_backward_reference() {}

    std::vector<task> work() 
    {
        requests.push_back({&outer});
        return {task{implementation, &requests[0]}};
    }

    static is_an_implementation *create(batch_normalization_training_backward &arg) { return new batch_normalization_training_backward_reference(arg); };

    static void implementation(const void *ptr) 
    {
        const auto request = reinterpret_cast<const request_data *>(ptr);
        const auto this_bn = static_cast<const batch_normalization_training_backward *>(request->primitive);

        if(this_bn->output().size() != 3 ||
           this_bn->input().size() != 6)
            throw std::runtime_error("Incorrect number of BatchNorm input/outputs.");

        const auto spatial = this_bn->argument.spatial;

        auto& forward_input = this_bn->input_memory(0);
        auto& forward_scale = this_bn->input_memory(1);
        // auto& forward_bias = this_bn->input_memory(2); // not required
        auto& output_grad = this_bn->input_memory(3);
        auto& current_mean = this_bn->input_memory(4);
        auto& current_inv_std_dev = this_bn->input_memory(5);

        auto& input_grad = this_bn->output_memory(0);
        auto& scale_grad = this_bn->output_memory(1);
        auto& bias_grad = this_bn->output_memory(2);

        auto forward_input_buffer         = static_cast<T*>(forward_input.pointer);
        auto forward_scale_buffer         = static_cast<T*>(forward_scale.pointer);
        //auto forward_bias_buffer          = static_cast<T*>(forward_bias.pointer); // not required
        auto output_grad_buffer           = static_cast<T*>(output_grad.pointer);
        auto current_mean_buffer          = static_cast<T*>(current_mean.pointer);
        auto current_inv_std_dev_buffer   = static_cast<T*>(current_inv_std_dev.pointer);

        auto input_grad_buffer = static_cast<T*>(input_grad.pointer);
        auto scale_grad_buffer = static_cast<T*>(scale_grad.pointer);
        auto bias_grad_buffer  = static_cast<T*>(bias_grad.pointer);

        if(output_grad.argument.format != input_grad.argument.format)
            throw std::runtime_error("Output format doesn't match.");

        const auto format = input_grad.argument.format;
        const auto data_size = memory::traits(format).type->size;

        auto data_w = 0;
        auto data_h = 0;
        auto data_c = 0;
        auto data_n = 0;

        auto spatial_location_stride = 0;
        auto element_stride = 0;
        auto batch_stride = 0;

        if(format == memory::format::yxfb_f32 ||
           format == memory::format::yxfb_f64)
        {
            data_h = input_grad.argument.size[0];
            data_w = input_grad.argument.size[1];
            data_c = input_grad.argument.size[2];
            data_n = input_grad.argument.size[3];

            spatial_location_stride = data_n * data_c;
            element_stride = data_n;
            batch_stride = 1;
        }
        else if(format == memory::format::xyfb_f32 ||
                format == memory::format::xyfb_f64)
        {
            data_w = input_grad.argument.size[0];
            data_h = input_grad.argument.size[1];
            data_c = input_grad.argument.size[2];
            data_n = input_grad.argument.size[3];

            spatial_location_stride = data_n * data_c;
            element_stride = data_n;
            batch_stride = 1;
        }
        else if(format == memory::format::fyxb_f32 ||
                format == memory::format::fyxb_f64)
        {
            data_c = input_grad.argument.size[0];
            data_h = input_grad.argument.size[1];
            data_w = input_grad.argument.size[2];
            data_n = input_grad.argument.size[3];

            spatial_location_stride = data_n;
            element_stride = data_n * data_w * data_h;
            batch_stride = 1;
        }
        else if(format == memory::format::fxyb_f32 ||
                format == memory::format::fxyb_f64)
        {
            data_c = input_grad.argument.size[0];
            data_w = input_grad.argument.size[1];
            data_h = input_grad.argument.size[2];
            data_n = input_grad.argument.size[3];

            spatial_location_stride = data_n;
            element_stride = data_n * data_w * data_h;
            batch_stride = 1;
        }
        else if(format == memory::format::byxf_f32 ||
                format == memory::format::byxf_f64)
        {
            data_n = input_grad.argument.size[0];
            data_h = input_grad.argument.size[1];
            data_w = input_grad.argument.size[2];
            data_c = input_grad.argument.size[3];

            spatial_location_stride = data_c;
            element_stride = 1;
            batch_stride = data_c * data_w * data_h;
        }
        else if(format == memory::format::bxyf_f32 ||
                format == memory::format::bxyf_f64)
        {
            data_n = input_grad.argument.size[0];
            data_w = input_grad.argument.size[1];
            data_h = input_grad.argument.size[2];
            data_c = input_grad.argument.size[3];

            spatial_location_stride = data_c;
            element_stride = 1;
            batch_stride = data_c * data_w * data_h;
        }
        else if(format == memory::format::bfyx_f32 ||
                format == memory::format::bfyx_f64)
        {
            data_n = input_grad.argument.size[0];
            data_c = input_grad.argument.size[1];
            data_h = input_grad.argument.size[2];
            data_w = input_grad.argument.size[3];

            spatial_location_stride = 1;
            element_stride = data_w * data_h;
            batch_stride = data_c * data_w * data_h;
        }
        else if(format == memory::format::bfxy_f32 ||
                format == memory::format::bfxy_f64)
        {
            data_n = input_grad.argument.size[0];
            data_c = input_grad.argument.size[1];
            data_w = input_grad.argument.size[2];
            data_h = input_grad.argument.size[3];

            spatial_location_stride = 1;
            element_stride = data_w * data_h;
            batch_stride = data_c * data_w * data_h;
        }
        else
        {
            throw std::runtime_error("Unknown BatchNorm format.");
        }

        const auto spatial_size = (spatial) ? data_w * data_h : 1;
        const auto num_averages = (spatial) ? data_c : data_c * data_w * data_h;
        const T inv_num_average_over = static_cast<T>(1.0 / (data_n * spatial_size));

        memset(scale_grad_buffer, 0, scale_grad.count() * data_size);
        memset(bias_grad_buffer, 0, bias_grad.count() * data_size);
        memset(input_grad_buffer, 0, input_grad.count() * data_size);

        auto compute_io_data_offset = [&element_stride, &batch_stride, &spatial_location_stride](int element, int batch, int spatial_location)
        {
            return spatial_location * spatial_location_stride + element * element_stride + batch * batch_stride;
        };

        // Compute scale and bias gradients. (can be MT over element)
        for(int element = 0; element < num_averages; ++element)
            for(int batch = 0; batch < data_n; ++batch)
                for(int spatial_location = 0; spatial_location < spatial_size; ++spatial_location)
                {
                    auto io_offset = compute_io_data_offset(element, batch, spatial_location);

                    scale_grad_buffer[element] += 
                          output_grad_buffer[io_offset] 
                        * (forward_input_buffer[io_offset] - current_mean_buffer[element])
                        * current_inv_std_dev_buffer[element];

                    bias_grad_buffer[element] += output_grad_buffer[io_offset];
                }

        // Compute input gradients. (can be MT over element)
        for(int element = 0; element < num_averages; ++element)
            for(int batch = 0; batch < data_n; ++batch)
                for(int spatial_location = 0; spatial_location < spatial_size; ++spatial_location)
                {
                    auto io_offset = compute_io_data_offset(element, batch, spatial_location);
                    auto x_norm = (forward_input_buffer[io_offset] - current_mean_buffer[element]) * current_inv_std_dev_buffer[element];

                    input_grad_buffer[io_offset] +=
                          forward_scale_buffer[element]
                        * current_inv_std_dev_buffer[element]
                        * (output_grad_buffer[io_offset] - (x_norm * scale_grad_buffer[element] + bias_grad_buffer[element]) * inv_num_average_over);                
                }
    }
};


template <class T>
struct batch_normalization_inference_reference : is_an_implementation {

    struct request_data
    {
        const batch_normalization_inference* primitive;
    };

    const batch_normalization_inference &outer;

    std::vector<request_data> requests;

    batch_normalization_inference_reference(batch_normalization_inference &arg)
        : is_an_implementation(neural::type_id<batch_normalization_inference_reference>()) 
        , outer(arg) {}

    ~batch_normalization_inference_reference() {}

    std::vector<task> work() 
    {
        requests.push_back({&outer});
        return {task{implementation, &requests[0]}};
    }

    static is_an_implementation *create(batch_normalization_inference &arg) { return new batch_normalization_inference_reference(arg); };

    static void implementation(const void *ptr) 
    {
        const auto request = reinterpret_cast<const request_data *>(ptr);
        const auto this_bn = static_cast<const batch_normalization_inference *>(request->primitive);

        if(this_bn->input().size() != 5 ||
           this_bn->output().size() != 1)
            throw std::runtime_error("Incorrect number of BatchNorm input/outputs.");

        const auto spatial = this_bn->argument.spatial;

        auto& input = this_bn->input_memory(0);
        auto& scale = this_bn->input_memory(1);
        auto& bias = this_bn->input_memory(2);
        auto& mean = this_bn->input_memory(3);
        auto& inv_std_dev = this_bn->input_memory(4);

        auto& output = this_bn->output_memory(0);

        auto input_buffer = static_cast<T*>(input.pointer);
        auto scale_buffer = static_cast<T*>(scale.pointer);
        auto bias_buffer = static_cast<T*>(bias.pointer);
        auto mean_buffer = static_cast<T*>(mean.pointer);
        auto inv_std_dev_buffer = static_cast<T*>(inv_std_dev.pointer);

        auto output_buffer = static_cast<T*>(output.pointer);

        if(output.argument.format != input.argument.format)
            throw std::runtime_error("Output format doesn't match.");

        const auto format = input.argument.format;

        auto data_w = 0;
        auto data_h = 0;
        auto data_c = 0;
        auto data_n = 0;

        auto spatial_location_stride = 0;
        auto element_stride = 0;
        auto batch_stride = 0;

        if(format == memory::format::yxfb_f32 ||
           format == memory::format::yxfb_f64)
        {
            data_h = input.argument.size[0];
            data_w = input.argument.size[1];
            data_c = input.argument.size[2];
            data_n = input.argument.size[3];

            spatial_location_stride = data_n * data_c;
            element_stride = data_n;
            batch_stride = 1;
        }
        else if(format == memory::format::xyfb_f32 ||
                format == memory::format::xyfb_f64)
        {
            data_w = input.argument.size[0];
            data_h = input.argument.size[1];
            data_c = input.argument.size[2];
            data_n = input.argument.size[3];

            spatial_location_stride = data_n * data_c;
            element_stride = data_n;
            batch_stride = 1;
        }
        else if(format == memory::format::fyxb_f32 ||
                format == memory::format::fyxb_f64)
        {
            data_c = input.argument.size[0];
            data_h = input.argument.size[1];
            data_w = input.argument.size[2];
            data_n = input.argument.size[3];

            spatial_location_stride = data_n;
            element_stride = data_n * data_w * data_h;
            batch_stride = 1;
        }
        else if(format == memory::format::fxyb_f32 ||
                format == memory::format::fxyb_f64)
        {
            data_c = input.argument.size[0];
            data_w = input.argument.size[1];
            data_h = input.argument.size[2];
            data_n = input.argument.size[3];

            spatial_location_stride = data_n;
            element_stride = data_n * data_w * data_h;
            batch_stride = 1;
        }
        else if(format == memory::format::byxf_f32 ||
                format == memory::format::byxf_f64)
        {
            data_n = input.argument.size[0];
            data_h = input.argument.size[1];
            data_w = input.argument.size[2];
            data_c = input.argument.size[3];

            spatial_location_stride = data_c;
            element_stride = 1;
            batch_stride = data_c * data_w * data_h;
        }
        else if(format == memory::format::bxyf_f32 ||
                format == memory::format::bxyf_f64)
        {
            data_n = input.argument.size[0];
            data_w = input.argument.size[1];
            data_h = input.argument.size[2];
            data_c = input.argument.size[3];

            spatial_location_stride = data_c;
            element_stride = 1;
            batch_stride = data_c * data_w * data_h;
        }
        else if(format == memory::format::bfyx_f32 ||
                format == memory::format::bfyx_f64)
        {
            data_n = input.argument.size[0];
            data_c = input.argument.size[1];
            data_h = input.argument.size[2];
            data_w = input.argument.size[3];

            spatial_location_stride = 1;
            element_stride = data_w * data_h;
            batch_stride = data_c * data_w * data_h;
        }
        else if(format == memory::format::bfxy_f32 ||
                format == memory::format::bfxy_f64)
        {
            data_n = input.argument.size[0];
            data_c = input.argument.size[1];
            data_w = input.argument.size[2];
            data_h = input.argument.size[3];

            spatial_location_stride = 1;
            element_stride = data_w * data_h;
            batch_stride = data_c * data_w * data_h;
        }
        else
        {
            throw std::runtime_error("Unknown BatchNorm format.");
        }

        const auto spatial_size = (spatial) ? data_w * data_h : 1;
        const auto num_averages = (spatial) ? data_c : data_c * data_w * data_h;

        auto compute_io_data_offset = [&element_stride, &batch_stride, &spatial_location_stride](int element, int batch, int spatial_location)
        {
            return spatial_location * spatial_location_stride + element * element_stride + batch * batch_stride;
        };

        // Normalize data. (can be MT over element)
        for(int element = 0; element < num_averages; ++element)
            for(int batch = 0; batch < data_n; ++batch)
                for(int spatial_location = 0; spatial_location < spatial_size; ++spatial_location)
                    output_buffer[compute_io_data_offset(element, batch, spatial_location)] = 
                    (input_buffer[compute_io_data_offset(element, batch, spatial_location)] - mean_buffer[element])
                    * inv_std_dev_buffer[element]
                    * scale_buffer[element]
                    + bias_buffer[element];
    }
};

//                                    engine                output                        input
using implementation_key = std::tuple<neural::engine::type, neural::memory::format::type, neural::memory::format::type>;

// maps of available implementations
static std::map<implementation_key, std::function<is_an_implementation *(batch_normalization_training_forward &)>> training_forward_implementation_map = 
{
    {std::make_tuple(engine::reference, memory::format::yxfb_f32, memory::format::yxfb_f32), batch_normalization_training_forward_reference<float>::create},
    {std::make_tuple(engine::reference, memory::format::fyxb_f32, memory::format::fyxb_f32), batch_normalization_training_forward_reference<float>::create},
    {std::make_tuple(engine::reference, memory::format::xyfb_f32, memory::format::xyfb_f32), batch_normalization_training_forward_reference<float>::create},
    {std::make_tuple(engine::reference, memory::format::fxyb_f32, memory::format::fxyb_f32), batch_normalization_training_forward_reference<float>::create},
    {std::make_tuple(engine::reference, memory::format::byxf_f32, memory::format::byxf_f32), batch_normalization_training_forward_reference<float>::create},
    {std::make_tuple(engine::reference, memory::format::bfyx_f32, memory::format::bfyx_f32), batch_normalization_training_forward_reference<float>::create},
    {std::make_tuple(engine::reference, memory::format::bxyf_f32, memory::format::bxyf_f32), batch_normalization_training_forward_reference<float>::create},
    {std::make_tuple(engine::reference, memory::format::bfxy_f32, memory::format::bfxy_f32), batch_normalization_training_forward_reference<float>::create},
    {std::make_tuple(engine::reference, memory::format::yxfb_f64, memory::format::yxfb_f64), batch_normalization_training_forward_reference<double>::create},
    {std::make_tuple(engine::reference, memory::format::fyxb_f64, memory::format::fyxb_f64), batch_normalization_training_forward_reference<double>::create},
    {std::make_tuple(engine::reference, memory::format::xyfb_f64, memory::format::xyfb_f64), batch_normalization_training_forward_reference<double>::create},
    {std::make_tuple(engine::reference, memory::format::fxyb_f64, memory::format::fxyb_f64), batch_normalization_training_forward_reference<double>::create},
    {std::make_tuple(engine::reference, memory::format::byxf_f64, memory::format::byxf_f64), batch_normalization_training_forward_reference<double>::create},
    {std::make_tuple(engine::reference, memory::format::bfyx_f64, memory::format::bfyx_f64), batch_normalization_training_forward_reference<double>::create},
    {std::make_tuple(engine::reference, memory::format::bxyf_f64, memory::format::bxyf_f64), batch_normalization_training_forward_reference<double>::create},
    {std::make_tuple(engine::reference, memory::format::bfxy_f64, memory::format::bfxy_f64), batch_normalization_training_forward_reference<double>::create},
};
static std::map<implementation_key, std::function<is_an_implementation *(batch_normalization_training_backward &)>> training_backward_implementation_map = 
{
    {std::make_tuple(engine::reference, memory::format::yxfb_f32, memory::format::yxfb_f32), batch_normalization_training_backward_reference<float>::create},
    {std::make_tuple(engine::reference, memory::format::fyxb_f32, memory::format::fyxb_f32), batch_normalization_training_backward_reference<float>::create},
    {std::make_tuple(engine::reference, memory::format::xyfb_f32, memory::format::xyfb_f32), batch_normalization_training_backward_reference<float>::create},
    {std::make_tuple(engine::reference, memory::format::fxyb_f32, memory::format::fxyb_f32), batch_normalization_training_backward_reference<float>::create},
    {std::make_tuple(engine::reference, memory::format::byxf_f32, memory::format::byxf_f32), batch_normalization_training_backward_reference<float>::create},
    {std::make_tuple(engine::reference, memory::format::bfyx_f32, memory::format::bfyx_f32), batch_normalization_training_backward_reference<float>::create},
    {std::make_tuple(engine::reference, memory::format::bxyf_f32, memory::format::bxyf_f32), batch_normalization_training_backward_reference<float>::create},
    {std::make_tuple(engine::reference, memory::format::bfxy_f32, memory::format::bfxy_f32), batch_normalization_training_backward_reference<float>::create},
    {std::make_tuple(engine::reference, memory::format::yxfb_f64, memory::format::yxfb_f64), batch_normalization_training_backward_reference<double>::create},
    {std::make_tuple(engine::reference, memory::format::fyxb_f64, memory::format::fyxb_f64), batch_normalization_training_backward_reference<double>::create},
    {std::make_tuple(engine::reference, memory::format::xyfb_f64, memory::format::xyfb_f64), batch_normalization_training_backward_reference<double>::create},
    {std::make_tuple(engine::reference, memory::format::fxyb_f64, memory::format::fxyb_f64), batch_normalization_training_backward_reference<double>::create},
    {std::make_tuple(engine::reference, memory::format::byxf_f64, memory::format::byxf_f64), batch_normalization_training_backward_reference<double>::create},
    {std::make_tuple(engine::reference, memory::format::bfyx_f64, memory::format::bfyx_f64), batch_normalization_training_backward_reference<double>::create},
    {std::make_tuple(engine::reference, memory::format::bxyf_f64, memory::format::bxyf_f64), batch_normalization_training_backward_reference<double>::create},
    {std::make_tuple(engine::reference, memory::format::bfxy_f64, memory::format::bfxy_f64), batch_normalization_training_backward_reference<double>::create},
};
static std::map<implementation_key, std::function<is_an_implementation *(batch_normalization_inference &)>> inference_implementation_map = 
{
    {std::make_tuple(engine::reference, memory::format::yxfb_f32, memory::format::yxfb_f32), batch_normalization_inference_reference<float>::create},
    {std::make_tuple(engine::reference, memory::format::fyxb_f32, memory::format::fyxb_f32), batch_normalization_inference_reference<float>::create},
    {std::make_tuple(engine::reference, memory::format::xyfb_f32, memory::format::xyfb_f32), batch_normalization_inference_reference<float>::create},
    {std::make_tuple(engine::reference, memory::format::fxyb_f32, memory::format::fxyb_f32), batch_normalization_inference_reference<float>::create},
    {std::make_tuple(engine::reference, memory::format::byxf_f32, memory::format::byxf_f32), batch_normalization_inference_reference<float>::create},
    {std::make_tuple(engine::reference, memory::format::bfyx_f32, memory::format::bfyx_f32), batch_normalization_inference_reference<float>::create},
    {std::make_tuple(engine::reference, memory::format::bxyf_f32, memory::format::bxyf_f32), batch_normalization_inference_reference<float>::create},
    {std::make_tuple(engine::reference, memory::format::bfxy_f32, memory::format::bfxy_f32), batch_normalization_inference_reference<float>::create},
    {std::make_tuple(engine::reference, memory::format::yxfb_f64, memory::format::yxfb_f64), batch_normalization_inference_reference<double>::create},
    {std::make_tuple(engine::reference, memory::format::fyxb_f64, memory::format::fyxb_f64), batch_normalization_inference_reference<double>::create},
    {std::make_tuple(engine::reference, memory::format::xyfb_f64, memory::format::xyfb_f64), batch_normalization_inference_reference<double>::create},
    {std::make_tuple(engine::reference, memory::format::fxyb_f64, memory::format::fxyb_f64), batch_normalization_inference_reference<double>::create},
    {std::make_tuple(engine::reference, memory::format::byxf_f64, memory::format::byxf_f64), batch_normalization_inference_reference<double>::create},
    {std::make_tuple(engine::reference, memory::format::bfyx_f64, memory::format::bfyx_f64), batch_normalization_inference_reference<double>::create},
    {std::make_tuple(engine::reference, memory::format::bxyf_f64, memory::format::bxyf_f64), batch_normalization_inference_reference<double>::create},
    {std::make_tuple(engine::reference, memory::format::bfxy_f64, memory::format::bfxy_f64), batch_normalization_inference_reference<double>::create},
};

} // namespace {


batch_normalization_training_forward::arguments::arguments( neural::engine::type engine, std::vector<primitive> output, std::vector<primitive_at> input, bool spatial, double exp_avg_factor, double epsilon)
    : engine(engine)
    , output(output)
    , input(input)
    , spatial(spatial)
    , exp_avg_factor(exp_avg_factor)
    , epsilon(epsilon) {}

// creates primitive with batch_normalization implementation that supports provided arguments
primitive batch_normalization_training_forward::create(batch_normalization_training_forward::arguments arg) 
{
    // wrap relu into RAII wrapper
    std::unique_ptr<batch_normalization_training_forward> result(new batch_normalization_training_forward(arg));

    // lookup in database; throw if not found
    auto key = std::make_tuple(arg.engine, result->input_memory(0).argument.format, result->output_memory(0).argument.format);
    auto it = training_forward_implementation_map.find(key);
    if(it==std::end(training_forward_implementation_map)) throw std::runtime_error("not yet implemented");

    // create implementation & attach it to result
    auto implementation = it->second(*result);
    result->_work = implementation->work();

    // release RAII wrapper, return naked pointer
    return result.release();
}

batch_normalization_training_backward::arguments::arguments( neural::engine::type engine, std::vector<primitive> output, std::vector<primitive_at> input, bool spatial)
    : engine(engine)
    , output(output)
    , input(input)
    , spatial(spatial) {}

// creates primitive with batch_normalization implementation that supports provided arguments
primitive batch_normalization_training_backward::create(batch_normalization_training_backward::arguments arg) 
{
    // wrap relu into RAII wrapper
    std::unique_ptr<batch_normalization_training_backward> result(new batch_normalization_training_backward(arg));

    // lookup in database; throw if not found
    auto key = std::make_tuple(arg.engine, result->input_memory(0).argument.format, result->output_memory(0).argument.format);
    auto it = training_backward_implementation_map.find(key);
    if(it==std::end(training_backward_implementation_map)) throw std::runtime_error("not yet implemented");

    // create implementation & attach it to result
    auto implementation = it->second(*result);
    result->_work = implementation->work();

    // release RAII wrapper, return naked pointer
    return result.release();
}

batch_normalization_inference::arguments::arguments( neural::engine::type engine, std::vector<primitive> output, std::vector<primitive_at> input, bool spatial)
    : engine(engine)
    , output(output)
    , input(input)
    , spatial(spatial) {}

// creates primitive with batch_normalization implementation that supports provided arguments
primitive batch_normalization_inference::create(batch_normalization_inference::arguments arg) 
{
    // wrap relu into RAII wrapper
    std::unique_ptr<batch_normalization_inference> result(new batch_normalization_inference(arg));

    // lookup in database; throw if not found
    auto key = std::make_tuple(arg.engine, result->input_memory(0).argument.format, result->output_memory(0).argument.format);
    auto it = inference_implementation_map.find(key);
    if(it==std::end(inference_implementation_map)) throw std::runtime_error("not yet implemented");

    // create implementation & attach it to result
    auto implementation = it->second(*result);
    result->_work = implementation->work();

    // release RAII wrapper, return naked pointer
    return result.release();
}

}