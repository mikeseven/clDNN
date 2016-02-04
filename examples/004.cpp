#include "neural.h"

// AlexNet with weights & biases from file
void example_004() {
    char  *input_buffer = nullptr;
    char *output_buffer = nullptr;

    uint32_t batch_size = 24;

    using namespace neural;
    auto input  = memory::create({engine::cpu, memory::format::xyzb, {227, 227, 3,  batch_size}});
    auto output = memory::create({engine::cpu, memory::format::xb, {1000, batch_size}});

    // [227x227x3xB] convolution->relu->pooling->lrn [27x27x96xB]
    auto conv1  = convolution::create({engine::cpu, memory::format::xyzb, input, 2, file::create({engine::cpu, "weight1.nnb"}), file::create({engine::cpu, "bias1.nnb"}), padding::zero});
    auto relu1  = relu::create({engine::cpu, memory::format::xyzb, conv1});
    auto pool1  = pooling::create({engine::cpu, pooling::mode::max, memory::format::xyzb, relu1, 3, 2, padding::zero});
    auto lrn1   = normalization::response::create({engine::cpu, memory::format::xyzb, pool1, 5, padding::zero, 1.0f, 0.00002f, 0.75f });

    // [27x27x96xB] convolution->relu->pooling->lrn [13x13x256xB]
    auto conv2  = convolution::create({engine::cpu, memory::format::xyzb, lrn1, file::create({engine::cpu, "weight2.nnb"}), file::create({engine::cpu, "bias2.nnb"}), padding::zero});
    auto relu2  = relu::create({engine::cpu, memory::format::xyzb, conv2});
    auto pool2  = pooling::create({engine::cpu, pooling::mode::max, memory::format::xyzb, relu2, 3, 2, padding::zero});
    auto lrn2   = normalization::response::create({engine::cpu, memory::format::xyzb, pool2, 5, padding::zero, 1.0f, 0.00002f, 0.75f });

    // [13x13x256xB] convolution->relu [13x13x384xB]
    auto conv3  = convolution::create({engine::cpu, memory::format::xyzb, lrn2, file::create({engine::cpu, "weight3.nnb"}), file::create({engine::cpu, "bias3.nnb"}), padding::zero});
    auto relu3  = relu::create({engine::cpu, memory::format::xyzb, conv3});

    // [13x13x384xB] convolution->relu [13x13x256xB]
    auto conv4  = convolution::create({engine::cpu, memory::format::xyzb, relu3, file::create({engine::cpu, "weight4.nnb"}), file::create({engine::cpu, "bias4.nnb"}), padding::zero});
    auto relu4  = relu::create({engine::cpu, memory::format::xyzb, conv4});

    // [13x13x256xB] convolution->relu->pooling [6x6x256xB]
    auto conv5  = convolution::create({engine::cpu, memory::format::xyzb, relu4, file::create({engine::cpu, "weight5.nnb"}), file::create({engine::cpu, "bias5.nnb"}), padding::zero});
    auto relu5  = relu::create({engine::cpu, memory::format::xyzb, conv5});
    auto pool5  = pooling::create({engine::cpu, pooling::mode::max, memory::format::xyzb, relu5, 3, 2, padding::zero});

    // [6x6x256xB] fully_connected->relu [4096xB]
    auto fc6    = fully_connected::create({engine::cpu, memory::format::xyzb, pool5, file::create({engine::cpu, "weight6.nnb"}), file::create({engine::cpu, "bias6.nnb"})});
    auto relu6  = relu::create({engine::cpu, memory::format::xyzb, fc6});

    // [4096xB] convolution->relu [4096xB]
    auto fc7    = fully_connected::create({engine::cpu, memory::format::xyzb, relu6, file::create({engine::cpu, "weight7.nnb"}), file::create({engine::cpu, "bias7.nnb"})});
    auto relu7  = relu::create({engine::cpu, memory::format::xyzb, fc7});

    // [4096xB] convolution->relu [1000xB]
    auto fc8    = fully_connected::create({engine::cpu, memory::format::xyzb, relu7, file::create({engine::cpu, "weight8.nnb"}), file::create({engine::cpu, "bias8.nnb"})});
    auto relu8  = relu::create({engine::cpu, memory::format::xyzb, fc8});

    // [1000xB] softmax [1000xB]
    auto sftmax = normalization::softmax::create({engine::cpu, output, relu8});

    execute({input(input_buffer), output(input_buffer), conv1, relu1, pool1, lrn1});
}
