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
#include "fully_connected.h"
#include "fully_connected_cpu_avx2.h"
#include "api/neural.h"
#include "multidimensional_counter.h"
#include "memory_utils.h"


namespace neural {


    fully_connected_forward_cpu_avx2::fully_connected_forward_cpu_avx2(fully_connected &arg)
        : is_an_implementation(neural::type_id<fully_connected_forward_cpu_avx2>())
        , outer(arg)
    {

    };

    fully_connected_forward_cpu_avx2::~fully_connected_forward_cpu_avx2()
    {

    }


	// NN_UNREACHABLE_CODE signal to supporting compiler that specific location in code cannot be reached
#if defined _MSC_VER
#   define NN_UNREACHABLE_CODE __assume(0)
#endif

#if defined __GNUC__
#   if (__GNUC__ * 100 + __GNUC_MINOR__) >= 405
#       define NN_UNREACHABLE_CODE __builtin_unreachable()
#   else
#       define NN_UNREACHABLE_CODE
#   endif
#endif

#if defined __clang__
#   if __has_builtin(__builtin_unreachable)
#       define NN_UNREACHABLE_CODE __builtin_unreachable()
#   else
#       define NN_UNREACHABLE_CODE
#   endif
#endif
	// ----------------------------------------------------------------------------------
	const auto C_simd_width = sizeof(__m256) / sizeof(float);

	static const auto C_max_acc_batch1 = 13u;
	static const auto C_max_acc_batch8 = 13u;
	static const auto C_max_acc_batch48 = 2u;

	static const auto C_batch8_size = C_simd_width;
	static const auto C_batch48_size = 6 * C_simd_width;

	static const auto C_data_stride_batch1 = C_simd_width * C_max_acc_batch1;
	static const auto C_data_stride_batch8 = C_batch8_size * C_max_acc_batch8;
	static const auto C_data_stride_batch48 = C_batch48_size * C_max_acc_batch48;
	// ------------------------------------------------------------------------------------

	typedef enum {
		NN_ACTIVATION_FUNCTION_NONE = 0,    /* f(x) = x */
		NN_ACTIVATION_FUNCTION_ABS,         /* f(x) = abs(x) */
		NN_ACTIVATION_FUNCTION_STEP,        /* f(x) = x<0 ? 0 : 1 */
		NN_ACTIVATION_FUNCTION_RELU,        /* f(x) = max(0, x) */
		NN_ACTIVATION_FUNCTION_SOFTPLUS,    /* f(x) = log(1+exp(x)) */
		NN_ACTIVATION_FUNCTION_LOGISTIC,    /* f(x) = 1/(1+exp(-x)) */
		NN_ACTIVATION_FUNCTION_TANH,        /* f(x) = a*tanh(x*b) | a&b are in nn_argument_activation_t.fp32_tanh*/
		NN_ACTIVATION_FUNCTION_LAST = NN_ACTIVATION_FUNCTION_TANH
	} NN_ACTIVATION_FUNCTION;


    template<uint32_t T_SIZE, NN_ACTIVATION_FUNCTION T_FUNCTION, bool T_NEED_BIAS_COPY>
    void fully_connected_compute_block_batch8(
        float* input_buffer,
        float* output_ptr,
        float* bias_ptr,
        float* weights_buffer,
        uint32_t input_width)
    {
        // We are not using table of registers and unroll pragmas
        // due to compiler which have issues with register allocation
        // and needs special, obvious treatment. Template immediate
        // arguments matching will remove all conditions in this code.
        __m256  acc0, acc1, acc2, acc3, acc4,
            acc5, acc6, acc7, acc8, acc9,
            acc10, acc11, acc12, acc13, acc14;

        if (T_NEED_BIAS_COPY)
        {
            if (T_SIZE >=  1)  acc0 = _mm256_setzero_ps();
            if (T_SIZE >=  2)  acc1 = _mm256_setzero_ps();
            if (T_SIZE >=  3)  acc2 = _mm256_setzero_ps();
            if (T_SIZE >=  4)  acc3 = _mm256_setzero_ps();
            if (T_SIZE >=  5)  acc4 = _mm256_setzero_ps();
            if (T_SIZE >=  6)  acc5 = _mm256_setzero_ps();
            if (T_SIZE >=  7)  acc6 = _mm256_setzero_ps();
            if (T_SIZE >=  8)  acc7 = _mm256_setzero_ps();
            if (T_SIZE >=  9)  acc8 = _mm256_setzero_ps();
            if (T_SIZE >= 10)  acc9 = _mm256_setzero_ps();
            if (T_SIZE >= 11) acc10 = _mm256_setzero_ps();
            if (T_SIZE >= 12) acc11 = _mm256_setzero_ps();
            if (T_SIZE >= 13) acc12 = _mm256_setzero_ps();
            if (T_SIZE >= 14) acc13 = _mm256_setzero_ps();
            if (T_SIZE >= 15) acc14 = _mm256_setzero_ps();
        }
        else
        {
            if (T_SIZE >=  1)  acc0 = _mm256_load_ps(output_ptr +  0 * C_batch8_size);
            if (T_SIZE >=  2)  acc1 = _mm256_load_ps(output_ptr +  1 * C_batch8_size);
            if (T_SIZE >=  3)  acc2 = _mm256_load_ps(output_ptr +  2 * C_batch8_size);
            if (T_SIZE >=  4)  acc3 = _mm256_load_ps(output_ptr +  3 * C_batch8_size);
            if (T_SIZE >=  5)  acc4 = _mm256_load_ps(output_ptr +  4 * C_batch8_size);
            if (T_SIZE >=  6)  acc5 = _mm256_load_ps(output_ptr +  5 * C_batch8_size);
            if (T_SIZE >=  7)  acc6 = _mm256_load_ps(output_ptr +  6 * C_batch8_size);
            if (T_SIZE >=  8)  acc7 = _mm256_load_ps(output_ptr +  7 * C_batch8_size);
            if (T_SIZE >=  9)  acc8 = _mm256_load_ps(output_ptr +  8 * C_batch8_size);
            if (T_SIZE >= 10)  acc9 = _mm256_load_ps(output_ptr +  9 * C_batch8_size);
            if (T_SIZE >= 11) acc10 = _mm256_load_ps(output_ptr + 10 * C_batch8_size);
            if (T_SIZE >= 12) acc11 = _mm256_load_ps(output_ptr + 11 * C_batch8_size);
            if (T_SIZE >= 13) acc12 = _mm256_load_ps(output_ptr + 12 * C_batch8_size);
            if (T_SIZE >= 14) acc13 = _mm256_load_ps(output_ptr + 13 * C_batch8_size);
            if (T_SIZE >= 15) acc14 = _mm256_load_ps(output_ptr + 14 * C_batch8_size);
        }

        auto input_ptr = &input_buffer[0];

        const auto input_ptr_end = &input_buffer[input_width*C_batch8_size];

        while (input_ptr < input_ptr_end)
        {
            // Do MADs.
            __m256 input = _mm256_load_ps(input_ptr);
            if (T_SIZE >=  1)  acc0 = _mm256_fmadd_ps(input, _mm256_broadcast_ss(weights_buffer +  0),  acc0);
            if (T_SIZE >=  2)  acc1 = _mm256_fmadd_ps(input, _mm256_broadcast_ss(weights_buffer +  1),  acc1);
            if (T_SIZE >=  3)  acc2 = _mm256_fmadd_ps(input, _mm256_broadcast_ss(weights_buffer +  2),  acc2);
            if (T_SIZE >=  4)  acc3 = _mm256_fmadd_ps(input, _mm256_broadcast_ss(weights_buffer +  3),  acc3);
            if (T_SIZE >=  5)  acc4 = _mm256_fmadd_ps(input, _mm256_broadcast_ss(weights_buffer +  4),  acc4);
            if (T_SIZE >=  6)  acc5 = _mm256_fmadd_ps(input, _mm256_broadcast_ss(weights_buffer +  5),  acc5);
            if (T_SIZE >=  7)  acc6 = _mm256_fmadd_ps(input, _mm256_broadcast_ss(weights_buffer +  6),  acc6);
            if (T_SIZE >=  8)  acc7 = _mm256_fmadd_ps(input, _mm256_broadcast_ss(weights_buffer +  7),  acc7);
            if (T_SIZE >=  9)  acc8 = _mm256_fmadd_ps(input, _mm256_broadcast_ss(weights_buffer +  8),  acc8);
            if (T_SIZE >= 10)  acc9 = _mm256_fmadd_ps(input, _mm256_broadcast_ss(weights_buffer +  9),  acc9);
            if (T_SIZE >= 11) acc10 = _mm256_fmadd_ps(input, _mm256_broadcast_ss(weights_buffer + 10), acc10);
            if (T_SIZE >= 12) acc11 = _mm256_fmadd_ps(input, _mm256_broadcast_ss(weights_buffer + 11), acc11);
            if (T_SIZE >= 13) acc12 = _mm256_fmadd_ps(input, _mm256_broadcast_ss(weights_buffer + 12), acc12);
            if (T_SIZE >= 14) acc13 = _mm256_fmadd_ps(input, _mm256_broadcast_ss(weights_buffer + 13), acc13);
            if (T_SIZE >= 15) acc14 = _mm256_fmadd_ps(input, _mm256_broadcast_ss(weights_buffer + 14), acc14);

            // Increment pointers.
            input_ptr += C_batch8_size;
            weights_buffer += C_max_acc_batch8;
            //weights_buffer += T_SIZE;
        }

        if (T_NEED_BIAS_COPY)
        {
            // Add biases.
            if (T_SIZE >=  1)  acc0 = _mm256_add_ps(_mm256_broadcast_ss(bias_ptr +  0),  acc0);
            if (T_SIZE >=  2)  acc1 = _mm256_add_ps(_mm256_broadcast_ss(bias_ptr +  1),  acc1);
            if (T_SIZE >=  3)  acc2 = _mm256_add_ps(_mm256_broadcast_ss(bias_ptr +  2),  acc2);
            if (T_SIZE >=  4)  acc3 = _mm256_add_ps(_mm256_broadcast_ss(bias_ptr +  3),  acc3);
            if (T_SIZE >=  5)  acc4 = _mm256_add_ps(_mm256_broadcast_ss(bias_ptr +  4),  acc4);
            if (T_SIZE >=  6)  acc5 = _mm256_add_ps(_mm256_broadcast_ss(bias_ptr +  5),  acc5);
            if (T_SIZE >=  7)  acc6 = _mm256_add_ps(_mm256_broadcast_ss(bias_ptr +  6),  acc6);
            if (T_SIZE >=  8)  acc7 = _mm256_add_ps(_mm256_broadcast_ss(bias_ptr +  7),  acc7);
            if (T_SIZE >=  9)  acc8 = _mm256_add_ps(_mm256_broadcast_ss(bias_ptr +  8),  acc8);
            if (T_SIZE >= 10)  acc9 = _mm256_add_ps(_mm256_broadcast_ss(bias_ptr +  9),  acc9);
            if (T_SIZE >= 11) acc10 = _mm256_add_ps(_mm256_broadcast_ss(bias_ptr + 10), acc10);
            if (T_SIZE >= 12) acc11 = _mm256_add_ps(_mm256_broadcast_ss(bias_ptr + 11), acc11);
            if (T_SIZE >= 13) acc12 = _mm256_add_ps(_mm256_broadcast_ss(bias_ptr + 12), acc12);
            if (T_SIZE >= 14) acc13 = _mm256_add_ps(_mm256_broadcast_ss(bias_ptr + 13), acc13);
            if (T_SIZE >= 15) acc14 = _mm256_add_ps(_mm256_broadcast_ss(bias_ptr + 14), acc14);
        }

        if (T_FUNCTION == NN_ACTIVATION_FUNCTION_RELU)
        {
            // Perform ReLU.
            if (T_SIZE >=  1)  acc0 = _mm256_max_ps(_mm256_setzero_ps(),  acc0);
            if (T_SIZE >=  2)  acc1 = _mm256_max_ps(_mm256_setzero_ps(),  acc1);
            if (T_SIZE >=  3)  acc2 = _mm256_max_ps(_mm256_setzero_ps(),  acc2);
            if (T_SIZE >=  4)  acc3 = _mm256_max_ps(_mm256_setzero_ps(),  acc3);
            if (T_SIZE >=  5)  acc4 = _mm256_max_ps(_mm256_setzero_ps(),  acc4);
            if (T_SIZE >=  6)  acc5 = _mm256_max_ps(_mm256_setzero_ps(),  acc5);
            if (T_SIZE >=  7)  acc6 = _mm256_max_ps(_mm256_setzero_ps(),  acc6);
            if (T_SIZE >=  8)  acc7 = _mm256_max_ps(_mm256_setzero_ps(),  acc7);
            if (T_SIZE >=  9)  acc8 = _mm256_max_ps(_mm256_setzero_ps(),  acc8);
            if (T_SIZE >= 10)  acc9 = _mm256_max_ps(_mm256_setzero_ps(),  acc9);
            if (T_SIZE >= 11) acc10 = _mm256_max_ps(_mm256_setzero_ps(), acc10);
            if (T_SIZE >= 12) acc11 = _mm256_max_ps(_mm256_setzero_ps(), acc11);
            if (T_SIZE >= 13) acc12 = _mm256_max_ps(_mm256_setzero_ps(), acc12);
            if (T_SIZE >= 14) acc13 = _mm256_max_ps(_mm256_setzero_ps(), acc13);
            if (T_SIZE >= 15) acc14 = _mm256_max_ps(_mm256_setzero_ps(), acc14);
        }

        // Store results.
        if (T_SIZE >=  1) _mm256_store_ps(output_ptr +  0 * C_batch8_size,  acc0);
        if (T_SIZE >=  2) _mm256_store_ps(output_ptr +  1 * C_batch8_size,  acc1);
        if (T_SIZE >=  3) _mm256_store_ps(output_ptr +  2 * C_batch8_size,  acc2);
        if (T_SIZE >=  4) _mm256_store_ps(output_ptr +  3 * C_batch8_size,  acc3);
        if (T_SIZE >=  5) _mm256_store_ps(output_ptr +  4 * C_batch8_size,  acc4);
        if (T_SIZE >=  6) _mm256_store_ps(output_ptr +  5 * C_batch8_size,  acc5);
        if (T_SIZE >=  7) _mm256_store_ps(output_ptr +  6 * C_batch8_size,  acc6);
        if (T_SIZE >=  8) _mm256_store_ps(output_ptr +  7 * C_batch8_size,  acc7);
        if (T_SIZE >=  9) _mm256_store_ps(output_ptr +  8 * C_batch8_size,  acc8);
        if (T_SIZE >= 10) _mm256_store_ps(output_ptr +  9 * C_batch8_size,  acc9);
        if (T_SIZE >= 11) _mm256_store_ps(output_ptr + 10 * C_batch8_size, acc10);
        if (T_SIZE >= 12) _mm256_store_ps(output_ptr + 11 * C_batch8_size, acc11);
        if (T_SIZE >= 13) _mm256_store_ps(output_ptr + 12 * C_batch8_size, acc12);
        if (T_SIZE >= 14) _mm256_store_ps(output_ptr + 13 * C_batch8_size, acc13);
        if (T_SIZE >= 15) _mm256_store_ps(output_ptr + 14 * C_batch8_size, acc14);
    }





    template <NN_ACTIVATION_FUNCTION T_FUNCTION, bool T_NEED_BIAS_COPY>
    void run_fully_connected_work_item_internal_batch8(const void *ptr) 
    {
        //const auto input_width = input->parent->lengths.t[NN_DATA_COORD_x];
        //const auto output_width = output->view_end.t[NN_DATA_COORD_x] - output->view_begin.t[NN_DATA_COORD_x] + 1;

        //const auto num_full_blocks = output_width / C_max_acc_batch8;
        //const auto partial_block_size = output_width % C_max_acc_batch8;

        //auto input_buffer = static_cast<float*>(input->parent->data_buffer);
        //auto output_buffer = static_cast<float*>(output->parent->data_buffer);
        //auto weights_buffer = static_cast<float*>(weights->parent->data_buffer);

        //// Output views.
        //const auto output_view_start = output->view_begin.t[NN_DATA_COORD_x];
        //const auto output_view_batch_offset = output_view_start * C_batch8_size;

        //// Weight views (for output-related weights).
        //auto weight_view_start =
        //    output_view_start / C_max_acc_batch8 * input_width * C_max_acc_batch8;

        //auto weights_ptr = &weights_buffer[weight_view_start];
        //auto output_ptr = &output_buffer[output_view_batch_offset];

        auto this_fc = static_cast<const fully_connected *>(ptr);
        auto input_buffer  = static_cast<float*>(this_fc->input_memory(0).pointer);
        auto output_buffer = static_cast<float*>(this_fc->output_memory(0).pointer);
        auto weight_buffer = static_cast<float*>(this_fc->input_memory(1).pointer);
        auto bias_buffer = static_cast<float*>(this_fc->argument.input[2].primitive.as<const memory&>().pointer);

        auto& input_arg = this_fc->input_memory(0).argument;
        auto& input_buffer_size = input_arg.size;

        auto& output_arg = this_fc->output_memory(0).argument;
        auto& output_buffer_size = output_arg.size;

        auto& weight_arg = this_fc->input_memory(1).argument;

        assert(1 == input_buffer_size.feature.size());
        assert(1 == input_buffer_size.batch.size());
        assert(1 == input_buffer_size.feature[0]);
        assert(0 == output_buffer_size.spatial[0] % C_max_acc_batch8);

        // what da f...???
        const auto input_width = input_buffer_size.spatial[0];
        const auto output_width = output_buffer_size.spatial[0];
        const auto output_length = output_width; // !!!!!!!!!!!!!!!
       
        auto weights_ptr = weight_buffer;
        auto output_ptr = output_buffer;
        auto bias_ptr = bias_buffer;

        const auto num_full_blocks = output_width / C_max_acc_batch8;
        const auto partial_block_size = output_width % C_max_acc_batch8;
     /*   
        float* bias_ptr = nullptr;
        if (T_NEED_BIAS_COPY)
        {
            auto biases_buffer = static_cast<float*>(bias->parent->data_buffer);
            bias_ptr = &biases_buffer[output_view_start];
        }
*/
        for (auto block = 0u; block < num_full_blocks; ++block)
        {
            // Run computation.
            fully_connected_compute_block_batch8<C_max_acc_batch8, T_FUNCTION, T_NEED_BIAS_COPY>(input_buffer, output_ptr, bias_ptr, weights_ptr, input_width);

            // Increment pointers.
            output_ptr += C_data_stride_batch8;
            weights_ptr += input_width*C_max_acc_batch8;

            if (T_NEED_BIAS_COPY)
            {
                bias_ptr += C_max_acc_batch8;
            }
        }

        switch (partial_block_size)
        {
        case  0: break;
        case  1: fully_connected_compute_block_batch8< 1, T_FUNCTION, T_NEED_BIAS_COPY>(input_buffer, output_ptr, bias_ptr, weights_ptr, input_width); break;
        case  2: fully_connected_compute_block_batch8< 2, T_FUNCTION, T_NEED_BIAS_COPY>(input_buffer, output_ptr, bias_ptr, weights_ptr, input_width); break;
        case  3: fully_connected_compute_block_batch8< 3, T_FUNCTION, T_NEED_BIAS_COPY>(input_buffer, output_ptr, bias_ptr, weights_ptr, input_width); break;
        case  4: fully_connected_compute_block_batch8< 4, T_FUNCTION, T_NEED_BIAS_COPY>(input_buffer, output_ptr, bias_ptr, weights_ptr, input_width); break;
        case  5: fully_connected_compute_block_batch8< 5, T_FUNCTION, T_NEED_BIAS_COPY>(input_buffer, output_ptr, bias_ptr, weights_ptr, input_width); break;
        case  6: fully_connected_compute_block_batch8< 6, T_FUNCTION, T_NEED_BIAS_COPY>(input_buffer, output_ptr, bias_ptr, weights_ptr, input_width); break;
        case  7: fully_connected_compute_block_batch8< 7, T_FUNCTION, T_NEED_BIAS_COPY>(input_buffer, output_ptr, bias_ptr, weights_ptr, input_width); break;
        case  8: fully_connected_compute_block_batch8< 8, T_FUNCTION, T_NEED_BIAS_COPY>(input_buffer, output_ptr, bias_ptr, weights_ptr, input_width); break;
        case  9: fully_connected_compute_block_batch8< 9, T_FUNCTION, T_NEED_BIAS_COPY>(input_buffer, output_ptr, bias_ptr, weights_ptr, input_width); break;
        case 10: fully_connected_compute_block_batch8<10, T_FUNCTION, T_NEED_BIAS_COPY>(input_buffer, output_ptr, bias_ptr, weights_ptr, input_width); break;
        case 11: fully_connected_compute_block_batch8<11, T_FUNCTION, T_NEED_BIAS_COPY>(input_buffer, output_ptr, bias_ptr, weights_ptr, input_width); break;
        case 12: fully_connected_compute_block_batch8<12, T_FUNCTION, T_NEED_BIAS_COPY>(input_buffer, output_ptr, bias_ptr, weights_ptr, input_width); break;
        default:
            NN_UNREACHABLE_CODE;
        }
    }


// ---------------------------------------------------------------------------------------------------------------------------------------



	template<uint32_t T_NUM_ITERATIONS, NN_ACTIVATION_FUNCTION T_FUNCTION, bool T_NEED_BIAS_COPY>
	void fully_connected_compute_subsimd_latency(
		float* input_buffer,
		float* &output_buffer,
		float* &bias_buffer,
		float* &weights_buffer,
		uint32_t input_width,
		uint32_t output_length)
	{
		for (auto iteration = 0u; iteration < T_NUM_ITERATIONS; ++iteration)
		{
			auto output_ptr = output_buffer;
			auto bias_ptr = bias_buffer;
			auto weights_ptr = weights_buffer;

			float acc0 = 0.0f;
			if (!T_NEED_BIAS_COPY)
			{
				acc0 = *output_ptr;
			}  

			auto input_ptr = &input_buffer[0];
			const auto input_ptr_end = &input_buffer[input_width ];

			while (input_ptr < input_ptr_end)
			{
				// Do MADs.
				acc0 += (*input_ptr) * (*weights_ptr);
				++input_ptr;

				weights_ptr += output_length;
			}

			if (T_NEED_BIAS_COPY)
			{
				// Add biases.
				acc0 += *bias_ptr;
			}

			//if (T_FUNCTION == NN_ACTIVATION_FUNCTION_RELU)
			//{
			//	// Perform ReLU.
			//	acc0 = std::max(0.0f, acc0);
			//}

			// Store results.
			*output_ptr = acc0;

			++output_buffer;
			++bias_buffer;
			++weights_buffer; // !!!!!!!!!!!!!????????????????????????????????
			//weights_buffer = weights_ptr;
		}
	}

	template<uint32_t T_SIZE, NN_ACTIVATION_FUNCTION T_FUNCTION, bool T_NEED_BIAS_COPY>
	void fully_connected_compute_block_latency(
		float* input_buffer,
		float* &output_buffer,
		float* &bias_buffer,
		float* &weights_buffer,
		uint32_t input_width,
		uint32_t output_length)
	{
		auto output_ptr = output_buffer;
		auto bias_ptr = bias_buffer;
		auto weights_ptr = weights_buffer;

		// We are not using table of registers and unroll pragmas
		// due to compiler which have issues with register allocation
		// and needs special, obvious treatment. Template immediate
		// arguments matching will remove all conditions in this code.
		__m256  acc0, acc1, acc2, acc3, acc4,
			acc5, acc6, acc7, acc8, acc9,
			acc10, acc11, acc12, acc13, acc14;

		if (T_NEED_BIAS_COPY)
		{
			if (T_SIZE >= 1)  acc0 = _mm256_setzero_ps();
			if (T_SIZE >= 2)  acc1 = _mm256_setzero_ps();
			if (T_SIZE >= 3)  acc2 = _mm256_setzero_ps();
			if (T_SIZE >= 4)  acc3 = _mm256_setzero_ps();
			if (T_SIZE >= 5)  acc4 = _mm256_setzero_ps();
			if (T_SIZE >= 6)  acc5 = _mm256_setzero_ps();
			if (T_SIZE >= 7)  acc6 = _mm256_setzero_ps();
			if (T_SIZE >= 8)  acc7 = _mm256_setzero_ps();
			if (T_SIZE >= 9)  acc8 = _mm256_setzero_ps();
			if (T_SIZE >= 10)  acc9 = _mm256_setzero_ps();
			if (T_SIZE >= 11) acc10 = _mm256_setzero_ps();
			if (T_SIZE >= 12) acc11 = _mm256_setzero_ps();
			if (T_SIZE >= 13) acc12 = _mm256_setzero_ps();
			if (T_SIZE >= 14) acc13 = _mm256_setzero_ps();
			if (T_SIZE >= 15) acc14 = _mm256_setzero_ps();
		}
		else
		{
			if (T_SIZE >= 1)  acc0 = _mm256_loadu_ps(output_ptr + 0 * C_simd_width);
			if (T_SIZE >= 2)  acc1 = _mm256_loadu_ps(output_ptr + 1 * C_simd_width);
			if (T_SIZE >= 3)  acc2 = _mm256_loadu_ps(output_ptr + 2 * C_simd_width);
			if (T_SIZE >= 4)  acc3 = _mm256_loadu_ps(output_ptr + 3 * C_simd_width);
			if (T_SIZE >= 5)  acc4 = _mm256_loadu_ps(output_ptr + 4 * C_simd_width);
			if (T_SIZE >= 6)  acc5 = _mm256_loadu_ps(output_ptr + 5 * C_simd_width);
			if (T_SIZE >= 7)  acc6 = _mm256_loadu_ps(output_ptr + 6 * C_simd_width);
			if (T_SIZE >= 8)  acc7 = _mm256_loadu_ps(output_ptr + 7 * C_simd_width);
			if (T_SIZE >= 9)  acc8 = _mm256_loadu_ps(output_ptr + 8 * C_simd_width);
			if (T_SIZE >= 10)  acc9 = _mm256_loadu_ps(output_ptr + 9 * C_simd_width);
			if (T_SIZE >= 11) acc10 = _mm256_loadu_ps(output_ptr + 10 * C_simd_width);
			if (T_SIZE >= 12) acc11 = _mm256_loadu_ps(output_ptr + 11 * C_simd_width);
			if (T_SIZE >= 13) acc12 = _mm256_loadu_ps(output_ptr + 12 * C_simd_width);
			if (T_SIZE >= 14) acc13 = _mm256_loadu_ps(output_ptr + 13 * C_simd_width);
			if (T_SIZE >= 15) acc14 = _mm256_loadu_ps(output_ptr + 14 * C_simd_width);
		}


		auto input_ptr = &input_buffer[0];

		const auto input_ptr_end = &input_buffer[input_width];

		while (input_ptr < input_ptr_end)
		{
			// Do MADs.
			__m256 input = _mm256_broadcast_ss(input_ptr);
			if (T_SIZE >= 1)  acc0 = _mm256_fmadd_ps(input, _mm256_loadu_ps(weights_ptr + 0 * C_simd_width), acc0);
			if (T_SIZE >= 2)  acc1 = _mm256_fmadd_ps(input, _mm256_loadu_ps(weights_ptr + 1 * C_simd_width), acc1);
			if (T_SIZE >= 3)  acc2 = _mm256_fmadd_ps(input, _mm256_loadu_ps(weights_ptr + 2 * C_simd_width), acc2);
			if (T_SIZE >= 4)  acc3 = _mm256_fmadd_ps(input, _mm256_loadu_ps(weights_ptr + 3 * C_simd_width), acc3);
			if (T_SIZE >= 5)  acc4 = _mm256_fmadd_ps(input, _mm256_loadu_ps(weights_ptr + 4 * C_simd_width), acc4);
			if (T_SIZE >= 6)  acc5 = _mm256_fmadd_ps(input, _mm256_loadu_ps(weights_ptr + 5 * C_simd_width), acc5);
			if (T_SIZE >= 7)  acc6 = _mm256_fmadd_ps(input, _mm256_loadu_ps(weights_ptr + 6 * C_simd_width), acc6);
			if (T_SIZE >= 8)  acc7 = _mm256_fmadd_ps(input, _mm256_loadu_ps(weights_ptr + 7 * C_simd_width), acc7);
			if (T_SIZE >= 9)  acc8 = _mm256_fmadd_ps(input, _mm256_loadu_ps(weights_ptr + 8 * C_simd_width), acc8);
			if (T_SIZE >= 10)  acc9 = _mm256_fmadd_ps(input, _mm256_loadu_ps(weights_ptr + 9 * C_simd_width), acc9);
			if (T_SIZE >= 11) acc10 = _mm256_fmadd_ps(input, _mm256_loadu_ps(weights_ptr + 10 * C_simd_width), acc10);
			if (T_SIZE >= 12) acc11 = _mm256_fmadd_ps(input, _mm256_loadu_ps(weights_ptr + 11 * C_simd_width), acc11);
			if (T_SIZE >= 13) acc12 = _mm256_fmadd_ps(input, _mm256_loadu_ps(weights_ptr + 12 * C_simd_width), acc12);
			if (T_SIZE >= 14) acc13 = _mm256_fmadd_ps(input, _mm256_loadu_ps(weights_ptr + 13 * C_simd_width), acc13);
			if (T_SIZE >= 15) acc14 = _mm256_fmadd_ps(input, _mm256_loadu_ps(weights_ptr + 14 * C_simd_width), acc14);

			// Increment pointers.
			++input_ptr;
			weights_ptr += output_length;
		}

		if (T_NEED_BIAS_COPY)
		{
			// Add biases.
			if (T_SIZE >= 1)  acc0 = _mm256_add_ps(_mm256_loadu_ps(bias_ptr + 0 * C_simd_width), acc0);
			if (T_SIZE >= 2)  acc1 = _mm256_add_ps(_mm256_loadu_ps(bias_ptr + 1 * C_simd_width), acc1);
			if (T_SIZE >= 3)  acc2 = _mm256_add_ps(_mm256_loadu_ps(bias_ptr + 2 * C_simd_width), acc2);
			if (T_SIZE >= 4)  acc3 = _mm256_add_ps(_mm256_loadu_ps(bias_ptr + 3 * C_simd_width), acc3);
			if (T_SIZE >= 5)  acc4 = _mm256_add_ps(_mm256_loadu_ps(bias_ptr + 4 * C_simd_width), acc4);
			if (T_SIZE >= 6)  acc5 = _mm256_add_ps(_mm256_loadu_ps(bias_ptr + 5 * C_simd_width), acc5);
			if (T_SIZE >= 7)  acc6 = _mm256_add_ps(_mm256_loadu_ps(bias_ptr + 6 * C_simd_width), acc6);
			if (T_SIZE >= 8)  acc7 = _mm256_add_ps(_mm256_loadu_ps(bias_ptr + 7 * C_simd_width), acc7);
			if (T_SIZE >= 9)  acc8 = _mm256_add_ps(_mm256_loadu_ps(bias_ptr + 8 * C_simd_width), acc8);
			if (T_SIZE >= 10)  acc9 = _mm256_add_ps(_mm256_loadu_ps(bias_ptr + 9 * C_simd_width), acc9);
			if (T_SIZE >= 11) acc10 = _mm256_add_ps(_mm256_loadu_ps(bias_ptr + 10 * C_simd_width), acc10);
			if (T_SIZE >= 12) acc11 = _mm256_add_ps(_mm256_loadu_ps(bias_ptr + 11 * C_simd_width), acc11);
			if (T_SIZE >= 13) acc12 = _mm256_add_ps(_mm256_loadu_ps(bias_ptr + 12 * C_simd_width), acc12);
			if (T_SIZE >= 14) acc13 = _mm256_add_ps(_mm256_loadu_ps(bias_ptr + 13 * C_simd_width), acc13);
			if (T_SIZE >= 15) acc14 = _mm256_add_ps(_mm256_loadu_ps(bias_ptr + 14 * C_simd_width), acc14);
		}

		if (T_FUNCTION == NN_ACTIVATION_FUNCTION_RELU)
		{
			// Perform ReLU.
			if (T_SIZE >= 1)  acc0 = _mm256_max_ps(_mm256_setzero_ps(), acc0);
			if (T_SIZE >= 2)  acc1 = _mm256_max_ps(_mm256_setzero_ps(), acc1);
			if (T_SIZE >= 3)  acc2 = _mm256_max_ps(_mm256_setzero_ps(), acc2);
			if (T_SIZE >= 4)  acc3 = _mm256_max_ps(_mm256_setzero_ps(), acc3);
			if (T_SIZE >= 5)  acc4 = _mm256_max_ps(_mm256_setzero_ps(), acc4);
			if (T_SIZE >= 6)  acc5 = _mm256_max_ps(_mm256_setzero_ps(), acc5);
			if (T_SIZE >= 7)  acc6 = _mm256_max_ps(_mm256_setzero_ps(), acc6);
			if (T_SIZE >= 8)  acc7 = _mm256_max_ps(_mm256_setzero_ps(), acc7);
			if (T_SIZE >= 9)  acc8 = _mm256_max_ps(_mm256_setzero_ps(), acc8);
			if (T_SIZE >= 10)  acc9 = _mm256_max_ps(_mm256_setzero_ps(), acc9);
			if (T_SIZE >= 11) acc10 = _mm256_max_ps(_mm256_setzero_ps(), acc10);
			if (T_SIZE >= 12) acc11 = _mm256_max_ps(_mm256_setzero_ps(), acc11);
			if (T_SIZE >= 13) acc12 = _mm256_max_ps(_mm256_setzero_ps(), acc12);
			if (T_SIZE >= 14) acc13 = _mm256_max_ps(_mm256_setzero_ps(), acc13);
			if (T_SIZE >= 15) acc14 = _mm256_max_ps(_mm256_setzero_ps(), acc14);
		}

		// Store results.
		if (T_SIZE >= 1) _mm256_storeu_ps(output_ptr + 0 * C_simd_width, acc0);
		if (T_SIZE >= 2) _mm256_storeu_ps(output_ptr + 1 * C_simd_width, acc1);
		if (T_SIZE >= 3) _mm256_storeu_ps(output_ptr + 2 * C_simd_width, acc2);
		if (T_SIZE >= 4) _mm256_storeu_ps(output_ptr + 3 * C_simd_width, acc3);
		if (T_SIZE >= 5) _mm256_storeu_ps(output_ptr + 4 * C_simd_width, acc4);
		if (T_SIZE >= 6) _mm256_storeu_ps(output_ptr + 5 * C_simd_width, acc5);
		if (T_SIZE >= 7) _mm256_storeu_ps(output_ptr + 6 * C_simd_width, acc6);
		if (T_SIZE >= 8) _mm256_storeu_ps(output_ptr + 7 * C_simd_width, acc7);
		if (T_SIZE >= 9) _mm256_storeu_ps(output_ptr + 8 * C_simd_width, acc8);
		if (T_SIZE >= 10) _mm256_storeu_ps(output_ptr + 9 * C_simd_width, acc9);
		if (T_SIZE >= 11) _mm256_storeu_ps(output_ptr + 10 * C_simd_width, acc10);
		if (T_SIZE >= 12) _mm256_storeu_ps(output_ptr + 11 * C_simd_width, acc11);
		if (T_SIZE >= 13) _mm256_storeu_ps(output_ptr + 12 * C_simd_width, acc12);
		if (T_SIZE >= 14) _mm256_storeu_ps(output_ptr + 13 * C_simd_width, acc13);
		if (T_SIZE >= 15) _mm256_storeu_ps(output_ptr + 14 * C_simd_width, acc14);

		output_buffer += C_simd_width*T_SIZE;
		weights_buffer += C_simd_width*T_SIZE;

		if (T_NEED_BIAS_COPY)
		{
			bias_buffer += C_simd_width*T_SIZE;
		}
	}


	template <NN_ACTIVATION_FUNCTION T_FUNCTION, bool T_NEED_BIAS_COPY>
	static void run_fully_connected_work_item_internal_latency(const void *ptr)
	{
		auto this_fc = static_cast<const fully_connected *>(ptr);
		auto input_buffer  = static_cast<float*>(this_fc->input_memory(0).pointer);
		auto output_buffer = static_cast<float*>(this_fc->output_memory(0).pointer);
		auto weight_buffer = static_cast<float*>(this_fc->input_memory(1).pointer);
		auto bias_buffer = static_cast<float*>(this_fc->argument.input[2].primitive.as<const memory&>().pointer);
		
		auto& input_arg = this_fc->input_memory(0).argument;
		auto& input_buffer_size = input_arg.size;
		
		auto& output_arg = this_fc->output_memory(0).argument;
		auto& output_buffer_size = output_arg.size;

		auto& weight_arg = this_fc->input_memory(1).argument;

		assert(1 == input_buffer_size.feature.size());
		assert(1 == input_buffer_size.batch.size());
		assert(1 == input_buffer_size.feature[0]);

		// what da f...???
		const auto input_width = input_buffer_size.spatial[0];
		const auto output_width = output_buffer_size.spatial[0];
        
    
		const auto output_length = output_width; // !!!!!!!!!!!!!!!

		// ----------------------------------------------------------------------

		const auto num_full_blocks = output_width / C_data_stride_batch1;
		const auto partial_block_size = (output_width / C_simd_width) % C_max_acc_batch1;
		const auto subsimd_block_size = output_width % C_simd_width;

		auto weights_ptr = weight_buffer;
		auto output_ptr = output_buffer;
		auto bias_ptr = bias_buffer;

			for (auto block = 0u; block < num_full_blocks; ++block)
			{
				// Run computation.
				fully_connected_compute_block_latency<C_max_acc_batch1, NN_ACTIVATION_FUNCTION_NONE, true>(input_buffer, output_ptr, bias_ptr, weights_ptr, input_width, output_length);
			}

			switch (partial_block_size)
			{
			case  0: break;
			case  1: fully_connected_compute_block_latency< 1, T_FUNCTION, T_NEED_BIAS_COPY>(input_buffer, output_ptr, bias_ptr, weights_ptr, input_width, output_length); break;
			case  2: fully_connected_compute_block_latency< 2, T_FUNCTION, T_NEED_BIAS_COPY>(input_buffer, output_ptr, bias_ptr, weights_ptr, input_width, output_length); break;
			case  3: fully_connected_compute_block_latency< 3, T_FUNCTION, T_NEED_BIAS_COPY>(input_buffer, output_ptr, bias_ptr, weights_ptr, input_width, output_length); break;
			case  4: fully_connected_compute_block_latency< 4, T_FUNCTION, T_NEED_BIAS_COPY>(input_buffer, output_ptr, bias_ptr, weights_ptr, input_width, output_length); break;
			case  5: fully_connected_compute_block_latency< 5, T_FUNCTION, T_NEED_BIAS_COPY>(input_buffer, output_ptr, bias_ptr, weights_ptr, input_width, output_length); break;
			case  6: fully_connected_compute_block_latency< 6, T_FUNCTION, T_NEED_BIAS_COPY>(input_buffer, output_ptr, bias_ptr, weights_ptr, input_width, output_length); break;
			case  7: fully_connected_compute_block_latency< 7, T_FUNCTION, T_NEED_BIAS_COPY>(input_buffer, output_ptr, bias_ptr, weights_ptr, input_width, output_length); break;
			case  8: fully_connected_compute_block_latency< 8, T_FUNCTION, T_NEED_BIAS_COPY>(input_buffer, output_ptr, bias_ptr, weights_ptr, input_width, output_length); break;
			case  9: fully_connected_compute_block_latency< 9, T_FUNCTION, T_NEED_BIAS_COPY>(input_buffer, output_ptr, bias_ptr, weights_ptr, input_width, output_length); break;
			case 10: fully_connected_compute_block_latency<10, T_FUNCTION, T_NEED_BIAS_COPY>(input_buffer, output_ptr, bias_ptr, weights_ptr, input_width, output_length); break;
			case 11: fully_connected_compute_block_latency<11, T_FUNCTION, T_NEED_BIAS_COPY>(input_buffer, output_ptr, bias_ptr, weights_ptr, input_width, output_length); break;
			case 12: fully_connected_compute_block_latency<12, T_FUNCTION, T_NEED_BIAS_COPY>(input_buffer, output_ptr, bias_ptr, weights_ptr, input_width, output_length); break;
			case 13: fully_connected_compute_block_latency<13, T_FUNCTION, T_NEED_BIAS_COPY>(input_buffer, output_ptr, bias_ptr, weights_ptr, input_width, output_length); break;
			case 14: fully_connected_compute_block_latency<14, T_FUNCTION, T_NEED_BIAS_COPY>(input_buffer, output_ptr, bias_ptr, weights_ptr, input_width, output_length); break;
			default:
				NN_UNREACHABLE_CODE;
			}

			switch (subsimd_block_size)
			{
			case 0: break;
			case 1: fully_connected_compute_subsimd_latency<1, T_FUNCTION, T_NEED_BIAS_COPY>(input_buffer, output_ptr, bias_ptr, weights_ptr, input_width, output_length); break;
			case 2: fully_connected_compute_subsimd_latency<2, T_FUNCTION, T_NEED_BIAS_COPY>(input_buffer, output_ptr, bias_ptr, weights_ptr, input_width, output_length); break;
			case 3: fully_connected_compute_subsimd_latency<3, T_FUNCTION, T_NEED_BIAS_COPY>(input_buffer, output_ptr, bias_ptr, weights_ptr, input_width, output_length); break;
			case 4: fully_connected_compute_subsimd_latency<4, T_FUNCTION, T_NEED_BIAS_COPY>(input_buffer, output_ptr, bias_ptr, weights_ptr, input_width, output_length); break;
			case 5: fully_connected_compute_subsimd_latency<5, T_FUNCTION, T_NEED_BIAS_COPY>(input_buffer, output_ptr, bias_ptr, weights_ptr, input_width, output_length); break;
			case 6: fully_connected_compute_subsimd_latency<6, T_FUNCTION, T_NEED_BIAS_COPY>(input_buffer, output_ptr, bias_ptr, weights_ptr, input_width, output_length); break;
			case 7: fully_connected_compute_subsimd_latency<7, T_FUNCTION, T_NEED_BIAS_COPY>(input_buffer, output_ptr, bias_ptr, weights_ptr, input_width, output_length); break;
			default:
				NN_UNREACHABLE_CODE;
			}
		
	}




	static void implementation(const void *ptr)
	{
		auto this_fc = static_cast<const fully_connected *>(ptr);
		auto& input_arg = this_fc->input_memory(0).argument;
		auto& input_buffer_size = input_arg.size;
		
		auto batch_size = input_buffer_size.batch[0];
		if (!(batch_size == 1 || batch_size == 8 || batch_size == 48))  throw std::runtime_error("Batch size not supported");


		switch (batch_size)
		{
		case 1:
			run_fully_connected_work_item_internal_latency<NN_ACTIVATION_FUNCTION_NONE, true>(ptr);
			break;
		case 8:
			run_fully_connected_work_item_internal_batch8<NN_ACTIVATION_FUNCTION_NONE, true>(ptr);
			break;
	/*	case 48:
			run_fully_connected_work_item_internal_batch48<NN_ACTIVATION_FUNCTION_NONE, true>(ptr);
			break;*/
		default:
			break;
		}
	}



    static void implementation_cpu(const void *ptr) {
      	auto this_fc = static_cast<const fully_connected *>(ptr);
        auto input = static_cast<float*>(this_fc->input_memory(0).pointer);
        auto output = static_cast<float*>(this_fc->output_memory(0).pointer);
        auto weight = static_cast<float*>(this_fc->input_memory(1).pointer);
        auto& weight_buffer_size = this_fc->input_memory(1).argument.size;
        auto bias = static_cast<float*>(this_fc->argument.input[2].primitive.as<const memory&>().pointer);


        auto& input_arg = this_fc->input_memory(0).argument;
        auto& input_buffer_size = input_arg.size;

        auto& output_arg = this_fc->output_memory(0).argument;
        auto& output_buffer_size = output_arg.size;

        auto& weight_arg = this_fc->input_memory(1).argument;

        assert(1 == input_buffer_size.feature.size());
        assert(1 == input_buffer_size.batch.size());
        assert(1 == input_buffer_size.feature[0]);

        namespace nd = ndimensional;
        fill(this_fc->output_memory(0), 0.0f);

        const int DATA_INDEX = 2;
        const int BATCH_INDEX = 0;

        nd::value<uint32_t> range_output(output_buffer_size);
        range_output[BATCH_INDEX] = 1; //in every iteration whole batch is computed at once, so it has to be removed from the range
        nd::value<uint32_t> range_input(input_buffer_size);
        nd::value<uint32_t> range_weight(weight_buffer_size);

        auto calc_in_idx = nd::choose_calculate_idx(input_arg.format);
        auto calc_out_idx = nd::choose_calculate_idx(output_arg.format);
        auto calc_w_idx = nd::choose_calculate_idx(weight_arg.format);

        std::vector<uint32_t> arg_weight_idx(3);
        for (auto pos_out : range_output) {
            auto out_idx = calc_out_idx(output_arg.size.raw, pos_out);

            for (auto pos_in : range_input) {
                auto in_idx = calc_in_idx(input_arg.size.raw, pos_in);

                arg_weight_idx[DATA_INDEX] = pos_out[DATA_INDEX];
                arg_weight_idx[BATCH_INDEX] = pos_in[DATA_INDEX];
                auto w_idx = calc_w_idx(weight_arg.size.raw, arg_weight_idx);
                output[out_idx + pos_in[BATCH_INDEX]] += input[in_idx] * weight[w_idx];
            }
            for (auto b = 0u; b < range_input[BATCH_INDEX]; b++)
                output[out_idx + b] += bias[pos_out[DATA_INDEX]];
        }
    }

    task_group fully_connected_forward_cpu_avx2::work()
    {
        return{ { task{ implementation, &outer } }, schedule::single };
    }


    
namespace {
    struct attach {
        attach() {
            fully_con_implementation_map::instance().insert({ std::make_tuple(engine::cpu, memory::format::xb_f32, memory::format::xb_f32), fully_connected_forward_cpu_avx2::create });
            fully_con_implementation_map::instance().insert({ std::make_tuple(engine::cpu, memory::format::x_f32,  memory::format::x_f32),  fully_connected_forward_cpu_avx2::create });
        }
        ~attach() {}
    };

#ifdef __GNUC__
    __attribute__((visibility("default"))) //todo meybe dll_sym?
#elif _MSC_VER
#   pragma section(".nn_init$m", read, write)
#endif
    attach attach_impl;

}

}


