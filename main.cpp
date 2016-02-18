#include "neural.h"
#include "thread_pool.h"


auto main(int, char *[]) -> int {
    extern void spatial_bn_trivial_example_forward_training_float();
    extern void spatial_bn_trivial_example_forward_training_double();
    extern void spatial_bn_trivial_example_backward_training_float();
    extern void spatial_bn_trivial_example_backward_training_double();
    extern void spatial_bn_trivial_example_inference_float();
    extern void spatial_bn_trivial_example_inference_double();
    
    spatial_bn_trivial_example_forward_training_float();
    spatial_bn_trivial_example_forward_training_double();
    spatial_bn_trivial_example_backward_training_float();
    spatial_bn_trivial_example_backward_training_double();
    spatial_bn_trivial_example_inference_float();
    spatial_bn_trivial_example_inference_double();
}