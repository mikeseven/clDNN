// Copyright (c) 2018 Intel Corporation
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


#include "topologies.h"

#include <api/CPP/activation.hpp>
#include <api/CPP/border.hpp>
#include <api/CPP/convolution.hpp>
#include <api/CPP/crop.hpp>
#include <api/CPP/deconvolution.hpp>
#include <api/CPP/eltwise.hpp>
#include <api/CPP/input_layout.hpp>
#include <api/CPP/mvn.hpp>
#include <api/CPP/reorder.hpp>
#include <api/CPP/scale.hpp>

#include <string>

#include "file.h"
#include "file_system_utils.h"


using namespace cldnn;
using namespace cldnn::utils::examples;


static primitive_id add_conv_layer(const std::string& weights_dir, const engine& engine, topology& topology_inst,
                                   const std::string& layer_name, const primitive_id& input,
                                   const tensor& padding = {0, 0, 0, 0}, const tensor& stride = {1, 1, 1, 1},
                                   const bool add_relu = false)
{
    const auto weights_data = file::create({engine, join_path(weights_dir, layer_name + "_weights.nnd")});
    const auto bias_data    = file::create({engine, join_path(weights_dir, layer_name + "_bias.nnd")});

    auto conv_layer = convolution(
        layer_name,
        input,
        {weights_data},
        {bias_data},
        stride,
        padding,
        {1, 1, 1, 1},
        add_relu);

    topology_inst.add(weights_data, bias_data, conv_layer);

    return conv_layer;
}


static primitive_id add_inst_norm_layer(const std::string& weights_dir, const engine& engine, topology& topology_inst,
                                        const std::string& layer_name, const primitive_id& input,
                                        const float epsilon = 1e-5f, const bool add_relu = false)
{
    const auto scale_weights_data = file::create({engine, join_path(weights_dir, layer_name + "_weights.nnd")});
    const auto scale_bias_data    = file::create({engine, join_path(weights_dir, layer_name + "_bias.nnd")});

    const auto mvn_layer   = mvn(layer_name + "_mvn", input, false, true, epsilon);
    const auto scale_layer = scale(layer_name + "_scale", mvn_layer, scale_weights_data, scale_bias_data);

    topology_inst.add(scale_weights_data, scale_bias_data, mvn_layer, scale_layer);

    if (add_relu)
    {
        const auto relu_layer = activation(layer_name + "_relu", scale_layer, activation_relu);

        topology_inst.add(relu_layer);

        return relu_layer;
    }
    return scale_layer;
}

static primitive_id add_deconv_layer(const std::string& weights_dir, const engine& engine, topology& topology_inst,
                                     const std::string& layer_name, const primitive_id& input,
                                     const tensor& padding = {0, 0, 0, 0}, const tensor& stride = {1, 1, 1, 1},
                                     const tensor& out_size = {}, const bool add_relu = false)
{
    const auto weights_data = file::create({engine, join_path(weights_dir, layer_name + "_weights.nnd")});
    const auto bias_data    = file::create({engine, join_path(weights_dir, layer_name + "_bias.nnd")});

    if (out_size.spatial[0] > 0 && out_size.spatial[1] > 0)
    {
        auto deconv_layer = deconvolution(
            layer_name,
            input,
            {weights_data},
            {bias_data},
            stride,
            padding,
            add_relu,
            0.0f,
            out_size);

        topology_inst.add(weights_data, bias_data, deconv_layer);
        return deconv_layer;
    }

    auto deconv_layer = deconvolution(
        layer_name,
        input,
        {weights_data},
        {bias_data},
        stride,
        padding,
        add_relu);

    topology_inst.add(weights_data, bias_data, deconv_layer);
    return deconv_layer;
}

topology build_fns_instance_norm(const std::string& weights_dir, const engine& engine, layout& input_layout,
                                 const bool mean_subtract)
{
    // Set default spatial size if it is not provided before.
    if (input_layout.size.spatial[0] <= 0 || input_layout.size.spatial[1] <= 0)
    {
        input_layout.size.spatial[0] = 720;
        input_layout.size.spatial[1] = 720;
    }

    // Round up to multiply of 4.
    input_layout.size.spatial[0] = (input_layout.size.spatial[0] + 3) / 4 * 4;
    input_layout.size.spatial[1] = (input_layout.size.spatial[1] + 3) / 4 * 4;

    tensor deconv1_out_size {1, 1, input_layout.size.spatial[0] / 2 + 2, input_layout.size.spatial[1] / 2 + 2};
    tensor deconv2_out_size {1, 1, input_layout.size.spatial[0] + 2,     input_layout.size.spatial[1] + 2};

    // ONNX: "inputImage" [Bx3xWxH]
    const auto input = cldnn::input_layout("input", input_layout);
    topology topology_inst{input};

    // ----------------------------------------------------------------------------------------------------------------
    // ----------------------------------------------------------------------------------------------------------------
    // ----------------------------------------------------------------------------------------------------------------

    // Scale used is 1.0, so we can use reorder (with optional, applied by default bias).
    // ONNX: "inputImage2" [Bx3xWxH]
    primitive_id scaled_input;
    if (mean_subtract)
    {
        // Subtract mean values if necessary.
        const auto input_mean       = file::create({engine, join_path(weights_dir, "fns_instance_norm_mean.nnd")});
        const auto scaled_1_0_input = reorder(
            "scale1",
            input,
            {input_layout.data_type, format::bfyx, input_layout.size},
            input_mean);
        topology_inst.add(input_mean, scaled_1_0_input);
        scaled_input = scaled_1_0_input;
    }
    else
    {
        const auto scaled_1_0_input = reorder("scale1", input, format::bfyx, input_layout.data_type);
        topology_inst.add(scaled_1_0_input);
        scaled_input = scaled_1_0_input;
    }

    // ONNX: "SpatialReflectionPadding_3" [Bx3x(W+80)x(H+80)]
    const auto padded_image = border("pad", scaled_input, {0, 0, 40, 40}, border_type::mirror_101);
    topology_inst.add(padded_image);

    // ONNX: "SpatialConvolution_4" [Bx16x(W+80)x(H+80)]
    const auto conv1 = add_conv_layer(weights_dir, engine, topology_inst, "conv1", padded_image,
                                      {0, 0, -4, -4});

    // ONNX: "InstanceNormalization_5" / "ReLU_6" [Bx16x(W+80)x(H+80)]
    const auto inst_norm1 = add_inst_norm_layer(weights_dir, engine, topology_inst, "inst_norm1", conv1,
                                                1e-05f, true);

    // ----------------------------------------------------------------------------------------------------------------
    // ----------------------------------------------------------------------------------------------------------------

    // ONNX: "SpatialConvolution_7" [Bx32x(W/2+40)x(H/2+40)]
    const auto conv2 = add_conv_layer(weights_dir, engine, topology_inst, "conv2", inst_norm1,
                                      {0, 0, -1, -1}, {1, 1, 2, 2});

    // ONNX: "InstanceNormalization_8" / "ReLU_9" [Bx32x(W/2+40)x(H/2+40)]
    const auto inst_norm2 = add_inst_norm_layer(weights_dir, engine, topology_inst, "inst_norm2", conv2,
                                                1e-05f, true);

    // ----------------------------------------------------------------------------------------------------------------

    // ONNX: "SpatialConvolution_10" [Bx64x(W/4+20)x(H/4+20)]
    const auto conv3 = add_conv_layer(weights_dir, engine, topology_inst, "conv3", inst_norm2,
                                      {0, 0, -1, -1}, {1, 1, 2, 2});

    // ONNX: "InstanceNormalization_11" / "ReLU_12" [Bx64x(W/4+20)x(H/4+20)]
    const auto inst_norm3 = add_inst_norm_layer(weights_dir, engine, topology_inst, "inst_norm3", conv3,
                                                1e-05f, true);

    // ----------------------------------------------------------------------------------------------------------------
    // ----------------------------------------------------------------------------------------------------------------

    // ONNX: "SpatialZeroPadding_21" [Bx64x(W/4+16)x(H/4+16)]
    const auto crop1 = crop("crop1", inst_norm3, {0, 0, 2, 2}, {0, 0, 2, 2}, crop_borders);
    topology_inst.add(crop1);

    // ONNX: "SpatialConvolution_16" [Bx64x(W/4+18)x(H/4+18)]
    const auto conv4 = add_conv_layer(weights_dir, engine, topology_inst, "conv4", inst_norm3);

    // ONNX: "InstanceNormalization_17" / "ReLU_18" [Bx64x(W/4+18)x(H/4+18)]
    const auto inst_norm4 = add_inst_norm_layer(weights_dir, engine, topology_inst, "inst_norm4", conv4,
                                                1e-05f, true);

    // ONNX: "SpatialConvolution_19" [Bx64x(W/4+16)x(H/4+16)]
    const auto conv5 = add_conv_layer(weights_dir, engine, topology_inst, "conv5", inst_norm4);

    // ONNX: "Sequential_15" [Bx64x(W/4+16)x(H/4+16)]
    const auto inst_norm5 = add_inst_norm_layer(weights_dir, engine, topology_inst, "inst_norm5", conv5,
                                                1e-05f, false);

    // ONNX: "Sequential_13" [Bx64x(W/4+16)x(H/4+16)]
    const auto add1 = eltwise("add1", inst_norm5, crop1, eltwise_mode::sum);
    topology_inst.add(add1);


    // ----------------------------------------------------------------------------------------------------------------

    // ONNX: "SpatialZeroPadding_31" [Bx64x(W/4+12)x(H/4+12)]
    const auto crop2 = crop("crop2", add1, {0, 0, 2, 2}, {0, 0, 2, 2}, crop_borders);
    topology_inst.add(crop2);

    // ONNX: "SpatialConvolution_26" [Bx64x(W/4+14)x(H/4+14)]
    const auto conv6 = add_conv_layer(weights_dir, engine, topology_inst, "conv6", add1);

    // ONNX: "InstanceNormalization_27" / "ReLU_28" [Bx64x(W/4+14)x(H/4+14)]
    const auto inst_norm6 = add_inst_norm_layer(weights_dir, engine, topology_inst, "inst_norm6", conv6,
                                                1e-05f, true);

    // ONNX: "SpatialConvolution_29" [Bx64x(W/4+12)x(H/4+12)]
    const auto conv7 = add_conv_layer(weights_dir, engine, topology_inst, "conv7", inst_norm6);

    // ONNX: "Sequential_25" [Bx64x(W/4+12)x(H/4+12)]
    const auto inst_norm7 = add_inst_norm_layer(weights_dir, engine, topology_inst, "inst_norm7", conv7,
                                                1e-05f, false);

    // ONNX: "Sequential_23" [Bx64x(W/4+12)x(H/4+12)]
    const auto add2 = eltwise("add2", inst_norm7, crop2, eltwise_mode::sum);
    topology_inst.add(add2);

    // ----------------------------------------------------------------------------------------------------------------

    // ONNX: "SpatialZeroPadding_41" [Bx64x(W/4+8)x(H/4+8)]
    const auto crop3 = crop("crop3", add2, {0, 0, 2, 2}, {0, 0, 2, 2}, crop_borders);
    topology_inst.add(crop3);

    // ONNX: "SpatialConvolution_36" [Bx64x(W/4+10)x(H/4+10)]
    const auto conv8 = add_conv_layer(weights_dir, engine, topology_inst, "conv8", add2);

    // ONNX: "InstanceNormalization_37" / "ReLU_38" [Bx64x(W/4+10)x(H/4+10)]
    const auto inst_norm8 = add_inst_norm_layer(weights_dir, engine, topology_inst, "inst_norm8", conv8,
                                                1e-05f, true);

    // ONNX: "SpatialConvolution_39" [Bx64x(W/4+8)x(H/4+8)]
    const auto conv9 = add_conv_layer(weights_dir, engine, topology_inst, "conv9", inst_norm8);

    // ONNX: "Sequential_35" [Bx64x(W/4+8)x(H/4+8)]
    const auto inst_norm9 = add_inst_norm_layer(weights_dir, engine, topology_inst, "inst_norm9", conv9,
                                                1e-05f, false);

    // ONNX: "Sequential_33" [Bx64x(W/4+8)x(H/4+8)]
    const auto add3 = eltwise("add3", inst_norm9, crop3, eltwise_mode::sum);
    topology_inst.add(add3);

    // ----------------------------------------------------------------------------------------------------------------

    // ONNX: "SpatialZeroPadding_51" [Bx64x(W/4+4)x(H/4+4)]
    const auto crop4 = crop("crop4", add3, {0, 0, 2, 2}, {0, 0, 2, 2}, crop_borders);
    topology_inst.add(crop4);

    // ONNX: "SpatialConvolution_46" [Bx64x(W/4+6)x(H/4+6)]
    const auto conv10 = add_conv_layer(weights_dir, engine, topology_inst, "conv10", add3);

    // ONNX: "InstanceNormalization_47" / "ReLU_48" [Bx64x(W/4+6)x(H/4+6)]
    const auto inst_norm10 = add_inst_norm_layer(weights_dir, engine, topology_inst, "inst_norm10", conv10,
                                                1e-05f, true);

    // ONNX: "SpatialConvolution_49" [Bx64x(W/4+4)x(H/4+4)]
    const auto conv11 = add_conv_layer(weights_dir, engine, topology_inst, "conv11", inst_norm10);

    // ONNX: "Sequential_45" [Bx64x(W/4+4)x(H/4+4)]
    const auto inst_norm11 = add_inst_norm_layer(weights_dir, engine, topology_inst, "inst_norm11", conv11,
                                                1e-05f, false);

    // ONNX: "Sequential_43" [Bx64x(W/4+4)x(H/4+4)]
    const auto add4 = eltwise("add4", inst_norm11, crop4, eltwise_mode::sum);
    topology_inst.add(add4);

    // ----------------------------------------------------------------------------------------------------------------

    // ONNX: "SpatialZeroPadding_61" [Bx64x(W/4)x(H/4)]
    const auto crop5 = crop("crop5", add4, {0, 0, 2, 2}, {0, 0, 2, 2}, crop_borders);
    topology_inst.add(crop5);

    // ONNX: "SpatialConvolution_56" [Bx64x(W/4+2)x(H/4+2)]
    const auto conv12 = add_conv_layer(weights_dir, engine, topology_inst, "conv12", add4);

    // ONNX: "InstanceNormalization_57" / "ReLU_58" [Bx64x(W/4+2)x(H/4+2)]
    const auto inst_norm12 = add_inst_norm_layer(weights_dir, engine, topology_inst, "inst_norm12", conv12,
                                                1e-05f, true);

    // ONNX: "SpatialConvolution_59" [Bx64x(W/4)x(H/4)]
    const auto conv13 = add_conv_layer(weights_dir, engine, topology_inst, "conv13", inst_norm12);

    // ONNX: "Sequential_55" [Bx64x(W/4)x(H/4)]
    const auto inst_norm13 = add_inst_norm_layer(weights_dir, engine, topology_inst, "inst_norm13", conv13,
                                                1e-05f, false);

    // ONNX: "Sequential_53" [Bx64x(W/4)x(H/4)]
    const auto add5 = eltwise("add5", inst_norm13, crop5, eltwise_mode::sum);
    topology_inst.add(add5);

    // ----------------------------------------------------------------------------------------------------------------
    // ----------------------------------------------------------------------------------------------------------------

    // ONNX: "SpatialFullConvolution_63_output" [Bx32x(W/2+2)x(H/2+2)]
    const auto deconv1 = add_deconv_layer(weights_dir, engine, topology_inst, "deconv1", add5,
                                          {0, 0, 0, 0}, {1, 1, 2, 2}, deconv1_out_size);

    // ONNX: "SpatialFullConvolution_63" [Bx32x(W/2)x(H/2)]
    const auto crop6 = crop("crop6", deconv1, {0, 0, 1, 1}, {0, 0, 1, 1}, crop_borders);
    topology_inst.add(crop6);

    // ONNX: "InstanceNormalization_65" / "ReLU_66" [Bx32x(W/2)x(H/2)]
    const auto inst_norm14 = add_inst_norm_layer(weights_dir, engine, topology_inst, "inst_norm14", crop6,
                                                1e-05f, true);

    // ----------------------------------------------------------------------------------------------------------------

    // ONNX: "SpatialFullConvolution_67_output" [Bx16x(W+2)x(H+2)]
    const auto deconv2 = add_deconv_layer(weights_dir, engine, topology_inst, "deconv2", inst_norm14,
                                          {0, 0, 0, 0}, {1, 1, 2, 2}, deconv2_out_size);

    // ONNX: "SpatialFullConvolution_67" [Bx16xWxH]
    const auto crop7 = crop("crop7", deconv2, {0, 0, 1, 1}, {0, 0, 1, 1}, crop_borders);
    topology_inst.add(crop7);

    // ONNX: "InstanceNormalization_69" / "ReLU_70" [Bx16xWxH]
    const auto inst_norm15 = add_inst_norm_layer(weights_dir, engine, topology_inst, "inst_norm15", crop7,
                                                 1e-05f, true);

    // ----------------------------------------------------------------------------------------------------------------
    // ----------------------------------------------------------------------------------------------------------------

    // ONNX: "SpatialConvolution_71" [Bx3xWxH]
    const auto conv14 = add_conv_layer(weights_dir, engine, topology_inst, "conv14", inst_norm15,
                                       {0, 0, -4, -4});

    // ONNX: "Tanh_72" [Bx3xWxH]
    const auto tanh1   = activation("tanh1", conv14, activation_hyperbolic_tan);
    // ONNX: "deprocess_image_2" [Bx3xWxH]
    const auto linear1 = activation("linear1", tanh1, activation_linear, {150.0, 0.0});

    const auto scale_weights_data = file::create({engine, join_path(weights_dir, "scale2_weights.nnd")});
    const auto scale_bias_data    = file::create({engine, join_path(weights_dir, "scale2_bias.nnd")});
    // ONNX: "deprocess_image_2_scaled" / "outputImage" [Bx3xWxH] (scale per feature + bias per feature)
    const auto scale2 = scale("scale2", linear1, scale_weights_data, scale_bias_data);
    topology_inst.add(tanh1, linear1, scale_weights_data, scale_bias_data, scale2);

    // ----------------------------------------------------------------------------------------------------------------
    // ----------------------------------------------------------------------------------------------------------------
    // ----------------------------------------------------------------------------------------------------------------

    const auto img_output = reorder("output", scale2, format::byxf, input_layout.data_type);
    topology_inst.add(img_output);
    return topology_inst;
}
