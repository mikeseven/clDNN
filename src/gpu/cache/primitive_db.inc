// the file contents should look as follows: {"primitive id", "primitive code"},
#define COPYRIGHT_DEFINE R"(
/*
//
//                 INTEL CORPORATION PROPRIETARY INFORMATION
//    This software is supplied under the terms of a license agreement or 
//    nondisclosure agreement with Intel Corporation and may not be copied
//    or disclosed except in accordance with the terms of that agreement. 
//        Copyright(c) 2015 Intel Corporation. All Rights Reserved. 
*/
)"

// Convert from Q1.7.8 fixed point decimal into regular decimal
#define SHORT_2_FLOAT_DEFINE R"(inline float shortToFloat(short val)
{
#ifdef TYPE_Q78
    short sign = (val & 0x8000) >> 15;

    if (sign == 1)
    {
        val ^= 0xffff;
        val++;
    }
    short mantisa = (val & 0x7F00) >> 8;
    short fraction = (val & 0x00FF);

    short checker = 0x0001;
    float counter = 0;
    for (int i = 8; i > 0; i--)
    {
        if ((fraction & checker) != 0)
        {
            counter += 1.0f / (1 << i);
        }
        checker <<= 1;
    }

    counter += mantisa;

    if (sign == 1)
    {
        counter = -counter;
    }

    return counter;
#elif defined TYPE_S16
    return (float)val;
#endif
}
)"

// Convert from Q1.7.8 fixed point decimal into regular decimal
#define FLOAT_2_SHORT_GENERIC_DEFINE R"(short floatToShortQ78(double x)
{
    double x_scaled = x * 256;
    short maxVal = (1 << (16 - 1)) - 1;

    if (x_scaled > maxVal)
        return maxVal;
    if (x_scaled < -maxVal)
        return -maxVal;

    return (short)(x_scaled + 0.5);
}
)"

#define FLOAT_2_SHORT_DEFINE R"(short floatToShort(double x)
{
#ifdef TYPE_Q78
    return floatToShortQ78(x);
#elif defined TYPE_S16
    return (short)(int)round(x);
#endif
}
)"

#define FLOAT_2_SHORT_NO_ROUND_DEFINE R"(short floatToShortNoRound(double x)
{
#ifdef TYPE_Q78
    return floatToShortQ78(x);
#elif defined TYPE_S16
    return (short)(int)x;
#endif
}
)"

#define SHORT_Q_MUL_DEFINE R"(inline short q_mul(short a, short b)
{
#define Q (8)
#define K (1 << (Q - 1))
    short result;
    long temp;

    temp = (signed int)a * (signed int)b; // result type is operand's type
    // Rounding; mid values are rounded up
    temp += K;
    // Correct by dividing by base and saturate result
    result = (short)(temp >> Q);
    return result;
#undef Q
#undef K
}
)"

#define SHORT_Q_DIV_DEFINE R"(inline short q_div(short a, short b)
{
    #define Q (8)
    short result;
    long temp;
    temp = (int)a << Q;
    // Rounding: mid values are rounded up (down for negative values).
    if ((temp >= 0 && b >= 0) || (temp < 0 && b < 0))
        temp += b / 2;
    else
        temp -= b / 2;
    result = (short)(temp / b);

    return result;
   #undef Q
}
)"

#define SHORT_GENERIC_DIV_DEFINE R"(inline short divide(short a, short b)
{
#ifdef TYPE_Q78
    return q_div(a, b);
#elif defined TYPE_S16
    return a / b;
#endif
}
)"

#define SHORT_GENERIC_MUL_DEFINE R"(inline short multiple(short a, short b)
{
#ifdef TYPE_Q78
    return q_mul(a, b);
#elif defined TYPE_S16
    return a * b;
#endif
}
)"

#define ACTIVATION_FUNCTION_DEFINE R"(inline short activation_function(short in, int m, int n)
{
    double val = shortToFloat(in);
#if defined ACTIVATION_FUNCTION_LOGISTIC
    double res = 1.0 / (1.0 + exp(-val));
    return isinf(res) ? 0 : floatToShortNoRound(res);

#elif defined ACTIVATION_FUNCTION_HYPERBOLIC_TAN
    return floatToShortNoRound(tanh(val));

#elif defined ACTIVATION_FUNCTION_RELU
    return (in > 0 ? in : 0);

#elif defined ACTIVATION_FUNCTION_BRELU
    return min((float)m, max((float)0, (float)in));

#elif defined ACTIVATION_FUNCTION_SOFTRELU
    double res = log(1 + exp(val));
    return isinf(res) ? 0 : floatToShortNoRound(res);

#elif defined ACTIVATION_FUNCTION_ABS
    return floatToShortNoRound(fabs(val));

#elif defined ACTIVATION_FUNCTION_SQUARE
    return multiple(in, in);

#elif defined ACTIVATION_FUNCTION_SQRT
    return floatToShortNoRound(sqrt(val));

#elif defined ACTIVATION_FUNCTION_LINEAR
    return multiple(m, in) + n;
#else
    return in;
#endif
}
)"

#define DEFINE_ALL              \
COPYRIGHT_DEFINE                \
SHORT_2_FLOAT_DEFINE            \
FLOAT_2_SHORT_GENERIC_DEFINE    \
FLOAT_2_SHORT_DEFINE            \
FLOAT_2_SHORT_NO_ROUND_DEFINE   \
SHORT_Q_MUL_DEFINE              \
SHORT_Q_DIV_DEFINE              \
SHORT_GENERIC_DIV_DEFINE        \
SHORT_GENERIC_MUL_DEFINE        \
ACTIVATION_FUNCTION_DEFINE 

{
"convolution",
DEFINE_ALL
R"(
__kernel void convolution(__global signed short* input, int src_step, __global signed short int* output, int dst_step, 
                          __global signed short* weights, __global signed short* biases, 
                          int input_width, int input_height, int output_height)
{
#ifndef BIASES
#define BIASES biases
#endif 
#ifndef WEIGHTS
#define WEIGHTS weights
#endif 
    const unsigned int x = get_global_id(0);
    const unsigned int y = get_global_id(1);
    const unsigned int z = get_global_id(2);

    const unsigned int out_map_size = output_height * dst_step;
    const unsigned int filter_size = KERNEL_DEPTH * KERNEL_HEIGHT * KERNEL_WIDTH;

    signed short dotProd  =  BIASES[z];

    const int x_start = ((int)x * STRIDE - PADDING);
    const int y_start = ((int)y * STRIDE - PADDING);

    unsigned int filter_offset = z * filter_size;
    unsigned int input_offset = y_start * src_step + x_start;

    #pragma unroll
    for (unsigned int k = 0; k < KERNEL_DEPTH; ++k) 
    {
        #pragma unroll
        for (unsigned int j = 0; j < KERNEL_HEIGHT ; ++j) 
        {
            #pragma unroll
            for (unsigned int i = 0; i < KERNEL_WIDTH ; ++i) 
            {
                int input_offset_x = x_start + i;
                int input_offset_y = y_start + j;
                bool zero = false;
                zero = input_offset_x < 0 ? true : zero;
                zero = input_offset_y < 0 ? true : zero;
                zero = input_offset_x >= input_width ? true : zero;
                zero = input_offset_y >= input_height ? true : zero;

                dotProd += zero ? 0 : multiple(input[input_offset], WEIGHTS[filter_offset]);

                ++input_offset;
                ++filter_offset;
            }
            input_offset +=  src_step - KERNEL_WIDTH;
        }
        input_offset += (input_height - KERNEL_HEIGHT) * src_step;
    }
    output[z*out_map_size + y*dst_step + x] = activation_function(dotProd, LINEAR_M, LINEAR_N);

}
)"},
{
"normalization",
DEFINE_ALL
R"(
inline void internal_normalize(int index, float sum, unsigned int num_elements, __global const signed short* input, __global signed short* output)
{
    float f_in = shortToFloat(input[index]);
    float normalization_factor = pow(INPUT_K + ALPHA*sum / num_elements, -BETA);

    float normres =  f_in*normalization_factor;
    short s_out = floatToShort(normres);
    output[index] = activation_function(s_out, LINEAR_M ,LINEAR_N);
}
 
__kernel void normalization(__global const signed short* input, int src_step, __global signed short* output, 
                            int input_width, int input_height)
{
    const unsigned int x            = get_global_id(0);
    const unsigned int y            = get_global_id(1);
    const unsigned int z            = get_global_id(2);
    const unsigned int height_step  = input_height * src_step;
    const unsigned int elm_index    = z * height_step + y*src_step + x;

    float sum = 0.0f;

#ifdef ACROSS_MAPS

#ifdef MANUTALLY_ITERATE_ACROSS_MAPS
    unsigned int j_offset = elm_index
    int end_of_first_batch = min((int)ROUND_NORM_SIZE/2 + 1, NUM_OF_INPUTS);
    int start_of_last_batch = max((int)NUM_OF_INPUTS - ROUND_NORM_SIZE/2 - 1, 0);
    int end_of_adding_only = min(end_of_first_batch, start_of_last_batch);
    int end_batch_offset = (ROUND_NORM_SIZE/2)*height_step;
    for(int j = 0 ; j < end_of_first_batch ; j++)
    {
        float val = shortToFloat(input[j_offset]);
        sum += val*val;
        j_offset += height_step;
    }

    j_offset = elm_index;

    for(int j = 0 ; j < end_of_adding_only ; j++)
    {
        internal_normalize(j_offset, sum, ROUND_NORM_SIZE, input, output);

        j_offset += height_step;
        float last_val = shortToFloat(input[j_offset + end_batch_offset]);
        sum += last_val*last_val; 
    }

    for(int j = end_of_adding_only ; j < start_of_last_batch ; j++)
    {
        internal_normalize(j_offset, sum, ROUND_NORM_SIZE, input, output);

        float first_val = shortToFloat(input[j_offset - end_batch_offset]);
        j_offset += height_step;
        float last_val = shortToFloat(input[j_offset + end_batch_offset]);
        sum += last_val*last_val - first_val*first_val; 
    }

    for(int j = start_of_last_batch ; j < NUM_OF_INPUTS ; j++)
    {
        internal_normalize(j_offset, sum, ROUND_NORM_SIZE, input, output);

        float first_val = shortToFloat(input[j_offset - end_batch_offset]);
        j_offset += height_step;
        sum -= first_val*first_val; 
    }
#else
    unsigned int j_offset = elm_index - (ROUND_NORM_SIZE/2)*height_step;

    for(int j = 0 ; j < ROUND_NORM_SIZE ; j++)
    {
        const int z_idx = (j + z - ROUND_NORM_SIZE/2);
        bool zero = (z_idx < 0 || z_idx >= NUM_OF_INPUTS);
        float val = zero ? 0.0f : shortToFloat(input[j_offset]);
        sum += val*val;
        j_offset += height_step;
    }

    internal_normalize(elm_index, sum, ROUND_NORM_SIZE, input, output);
#endif
#else

    const int x_start = ((int)x - ROUND_NORM_SIZE/2);
    const int y_start = ((int)y - ROUND_NORM_SIZE/2);
    unsigned int input_offset  = z * height_step + y_start*src_step + x_start;

    for (unsigned int j = 0; j < ROUND_NORM_SIZE ; ++j) 
    {
        for (unsigned int i = 0; i < ROUND_NORM_SIZE ; ++i) 
        {
            int input_offset_x = x_start + i;
            int input_offset_y = y_start + j;
            bool zero = false;
            zero = input_offset_x < 0 ? true : zero;
            zero = input_offset_y < 0 ? true : zero;
            zero = input_offset_x >= input_width ? true : zero;
            zero = input_offset_y >= input_height ? true : zero;

            float val = zero ? 0.0f : shortToFloat(input[input_offset]);
            sum += val*val;
            ++input_offset;
        }
        input_offset += src_step - ROUND_NORM_SIZE;
    }

    internal_normalize(elm_index, sum, ROUND_NORM_SIZE*ROUND_NORM_SIZE, input, output);
#endif
}
)"},
{
"pooling",
DEFINE_ALL
R"(
__kernel void pooling(__global const short *src, int src_step, __global short *out, int dst_step, 
                      int input_width, int input_height, int output_height)
{
    const unsigned int x = get_global_id(0);
    const unsigned int y = get_global_id(1);
    const unsigned int z = get_global_id(2);

    const unsigned int start_x = x*POOL_STRIDE;
    const unsigned int start_y = y*POOL_STRIDE;

    __global const short* src_ptr = src + z*src_step*input_height;

    #ifdef MAX_POOLING
        short tmpRes = src_ptr[start_y*src_step + start_x];
    #else
        short tmpRes = 0;
    #endif

    short res = tmpRes;

    for(unsigned int j = 0; j < POOL_SIZE; ++j)
    {
        for(unsigned int i = 0; i < POOL_SIZE; ++i)
        {
            int src_x = start_x + i - POOL_PADDING;
            int src_y = start_y + j - POOL_PADDING;

            bool zero = false;
            zero = src_x < 0 ? true : zero;
            zero = src_y < 0 ? true : zero;
            zero = src_x >= input_width ? true : zero;
            zero = src_y >= input_height ? true : zero;

            tmpRes = zero ? 0 : src_ptr[src_y * src_step + src_x];

            #ifdef MAX_POOLING
                res = max(res, tmpRes);
            #else
                res += tmpRes;
            #endif
        }
    }

    const unsigned int out_index = x + (y + z*output_height) * dst_step;

    #ifdef MAX_POOLING
        out[out_index] = res;
    #else
        out[out_index] = divide(res, floatToShort(POOL_SIZE * POOL_SIZE));
    #endif

}
)"},
{
"fully_connected",
DEFINE_ALL
R"(
__kernel void cnn_fully_connected(__global signed short int* input, int input_step, __global signed short int* output, int output_step,
                                  __global signed short int* weights, __global signed short int* biases,
                                  int input_width, int input_height, int output_width, int output_height)
{
#ifndef INPUT_WIDTH
#define INPUT_WIDTH input_width
#endif 
#ifndef INPUT_HEIGHT
#define INPUT_HEIGHT input_height
#endif 
#ifndef OUTPUT_WIDTH
#define OUTPUT_WIDTH output_width
#endif 
#ifndef OUTPUT_HEIGHT
#define OUTPUT_HEIGHT output_height
#endif 

    const unsigned int x = get_global_id(0);
    const unsigned int y = get_global_id(1);
    const unsigned int z = get_global_id(2);

    const unsigned out_offset = z*OUTPUT_WIDTH*OUTPUT_HEIGHT + y*OUTPUT_WIDTH + x;
    int dotProd = biases[out_offset];

    __global short int* processed_neuron_weights = weights + out_offset * INPUT_WIDTH * INPUT_HEIGHT * INPUT_DEPTH;
    __global short int* processed_input_batch    = input;

    const int line_padding = input_step - INPUT_WIDTH;
    unsigned int weight_idx = 0;
    unsigned int input_idx = 0;

    for (unsigned int k = 0; k < INPUT_DEPTH; ++k)
    {
        for (unsigned int j = 0; j < INPUT_HEIGHT; ++j)
        {
            for(unsigned int i = 0; i < INPUT_WIDTH; ++i )
            {
                dotProd += multiple(processed_input_batch[input_idx], processed_neuron_weights[weight_idx]);
                weight_idx++;
                input_idx++;
            }
            input_idx += line_padding;
        }
    }

    unsigned int output_idx = z*output_step*OUTPUT_HEIGHT + y*output_step + x;
    output[output_idx] = activation_function(dotProd, LINEAR_M, LINEAR_N);
}
)"}