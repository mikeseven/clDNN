#include "neural.h"

// AlexNet with weights & biases from file
void example_005() {
    using namespace neural;

    char  *input_buffer = nullptr;
    char *output_buffer = nullptr;

    uint32_t batch_size = 24;

    auto input  = memory::create({engine::cpu, memory::format::xyzb, {227, 227, 3,  batch_size}});
    auto output = memory::create({engine::cpu, memory::format::xb, {1000, batch_size}});

    // [227x227x3xB] convolution->relu->pooling->lrn [1000xB]
    auto conv_relu1 = convolution_relu::create({engine::cpu, memory::format::xyzb, input, 2, file::create({engine::cpu, "weight1.nnb"}), file::create({engine::cpu, "bias1.nnb"}), padding::zero, 0.0f});
    auto pool1      = pooling::create({engine::cpu, pooling::mode::max, memory::format::xyzb, conv_relu1, 3, 2, padding::zero});
    auto lrn1       = normalization::response::create({engine::cpu, memory::format::xyzb, pool1, 5, padding::zero, 1.0f, 0.00002f, 0.75f });
    auto conv_relu2 = convolution_relu::create({engine::cpu, memory::format::xyzb, lrn1, file::create({engine::cpu, "weight2.nnb"}), file::create({engine::cpu, "bias2.nnb"}), padding::zero, 0.0f});
    auto pool2      = pooling::create({engine::cpu, pooling::mode::max, memory::format::xyzb, conv_relu2, 3, 2, padding::zero});
    auto lrn2       = normalization::response::create({engine::cpu, memory::format::xyzb, pool2, 5, padding::zero, 1.0f, 0.00002f, 0.75f });
    auto conv_relu3 = convolution_relu::create({engine::cpu, memory::format::xyzb, lrn2, file::create({engine::cpu, "weight3.nnb"}), file::create({engine::cpu, "bias3.nnb"}), padding::zero, 0.0f});
    auto conv_relu4 = convolution_relu::create({engine::cpu, memory::format::xyzb, conv_relu3, file::create({engine::cpu, "weight4.nnb"}), file::create({engine::cpu, "bias4.nnb"}), padding::zero, 0.0f});
    auto conv_relu5 = convolution_relu::create({engine::cpu, memory::format::xyzb, conv_relu4, file::create({engine::cpu, "weight5.nnb"}), file::create({engine::cpu, "bias5.nnb"}), padding::zero, 0.0f});
    auto pool5      = pooling::create({engine::cpu, pooling::mode::max, memory::format::xyzb, conv_relu5, 3, 2, padding::zero});
    auto fc_relu6   = fully_connected_relu::create({engine::cpu, memory::format::xyzb, pool5, file::create({engine::cpu, "weight6.nnb"}), file::create({engine::cpu, "bias6.nnb"}), 0.0f});
    auto fc_relu7   = fully_connected_relu::create({engine::cpu, memory::format::xyzb, fc_relu6, file::create({engine::cpu, "weight7.nnb"}), file::create({engine::cpu, "bias7.nnb"}), 0.0f});
    auto fc_relu8    = fully_connected_relu::create({engine::cpu, memory::format::xyzb, fc_relu7, file::create({engine::cpu, "weight8.nnb"}), file::create({engine::cpu, "bias8.nnb"}), 0.0f});
    auto soft_max   = normalization::softmax::create({engine::cpu, output, fc_relu8});

    execute({input(input_buffer), output(input_buffer), conv_relu1, pool1, lrn1, conv_relu2, pool2, lrn2, conv_relu3, conv_relu4, conv_relu5, pool5, fc_relu6, fc_relu7, fc_relu8, soft_max});
}
