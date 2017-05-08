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

#include "detection_output_inst.h"
#include "primitive_type_base.h"
#include "network_impl.h"

namespace cldnn
{
primitive_type_id detection_output_type_id()
{
    static primitive_type_base<detection_output, detection_output_inst> instance;
    return &instance;
}

layout detection_output_inst::calc_output_layout(detection_output_node const& node)
{
    if (node.get_dependencies().size() != 3)
    {
        throw std::invalid_argument("Detection output layer must get 3 inputs.");
    }

    auto input_layout = node.location().get_output_layout();

    // Batch size and feature size are 1.
    // Number of bounding boxes to be kept is set to keep_top_k*batch size. 
    // If number of detections is lower than keep_top_k, will write dummy results at the end with image_id=-1. 
    // Each row is a 7 dimension vector, which stores:
    // [image_id, label, confidence, xmin, ymin, xmax, ymax]
    return{ input_layout.data_type, cldnn::tensor(cldnn::format::bfyx,{ 1, 1, node.get_primitive()->keep_top_k * input_layout.size.batch[0], DETECTION_OUTPUT_ROW_SIZE }) };
}

std::string detection_output_inst::to_string(detection_output_node const& node)
{
    std::stringstream           primitive_description;
    auto desc                   = node.get_primitive();
    auto input_location         = node.location();
    auto input_confidence       = node.confidence();
    auto input_prior_box        = node.prior_box();
    auto share_location         = desc->share_location ? "true" : "false"; 
    auto variance_encoded       = desc->variance_encoded_in_target ? "true" : "false";
    std::string                 str_code_type;

    switch (desc->code_type)
    {
    case prior_box_code_type::corner:
        str_code_type = "corner";
        break;
    case prior_box_code_type::center_size:
        str_code_type = "center size";
        break;
    case prior_box_code_type::corner_size:
        str_code_type = "corner size";
        break;
    default:
        str_code_type = "not supported code type";
        break;
    }
    
    primitive_description << "id: " << desc->id << ", type: detection_output" <<
        "\n\tinput_location: " << input_location.id() << ", sizes: " << input_location.get_output_layout().size <<
        "\n\tinput_confidence: " << input_confidence.id() << ", sizes: " << input_confidence.get_output_layout().size <<
        "\n\tinput_prior_box: " << input_prior_box.id() << ", sizes: " << input_prior_box.get_output_layout().size <<
        "\n\tnum_classes: " << desc->num_classes << 
        "\n\tkeep_top_k: " << desc->keep_top_k << 
        "\n\tshare_location: " << share_location << 
        "\n\tbackground_label_id: " << desc->background_label_id << 
        "\n\tnms_treshold: " << desc->nms_threshold <<
        "\n\ttop_k: " << desc->top_k << 
        "\n\teta: " << desc->eta << 
        "\n\tcode_type: " << str_code_type << 
        "\n\tvariance_encoded: " << variance_encoded << 
        "\n\tconfidence_threshold: " << desc->confidence_threshold <<
        "\n\tinput padding: " << desc->input_padding <<
        "\n\toutput padding: " << desc->output_padding <<
        "\n\toutput: count: " << node.get_output_layout().count() << ",  size: " << node.get_output_layout().size << '\n';

    return primitive_description.str();
}

detection_output_inst::typed_primitive_inst(network_impl& network, detection_output_node const& node)
    :parent(network, node)
{
    if ( (location_memory().get_layout().size.format != format::bfyx) ||
         (confidence_memory().get_layout().size.format != format::bfyx) ||
         (prior_box_memory().get_layout().size.format != format::bfyx) )
    {
        throw std::invalid_argument("Detection output layer supports only bfyx input format.");
    }

    if (argument.input_padding || argument.output_padding)
    {
        throw std::invalid_argument("Detection output layer doesn't support input and output padding.");
    }
}
}
