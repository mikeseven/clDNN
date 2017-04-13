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
#pragma once
#include <limits>
#include "detection_output.h"
#include "../primitive.hpp"

namespace cldnn
{
/// @addtogroup cpp_api C++ API
/// @{
/// @addtogroup cpp_topology Network Topology
/// @{
/// @addtogroup cpp_primitives Primitives
/// @{

/// @brief Select method for coding the prior-boxes in the @ref detection output layer.
enum class prior_box_code_type : int32_t
{
	corner      = cldnn_code_type_corner,
	center_size = cldnn_code_type_center_size,
	corner_size = cldnn_code_type_corner_size
};

/// @brief Generates a list of detections based on location and confidence predictions by doing non maximum suppression.
/// @details Each row is a 7 dimension vector, which stores: [image_id, label, confidence, xmin, ymin, xmax, ymax].
/// If number of detections per image is lower than keep_top_k, will write dummy results at the end with image_id=-1. 
struct detection_output : public primitive_base<detection_output, CLDNN_PRIMITIVE_DESC(detection_output)>
{
    CLDNN_DECLATE_PRIMITIVE(detection_output)

	/// @brief Constructs pooling primitive.
    /// @param id This primitive id.
    /// @param input_location Input location primitive id.
	/// @param input_confidence Input confidence primitive id.
	/// @param input_prior_box Input prior-box primitive id.
    /// @param num_classes Number of classes to be predicted.
    /// @param keep_top_k Number of total bounding boxes to be kept per image after NMS step.
    /// @param share_location If true bounding box are shared among different classes.
	/// @param background_label_id Background label id (-1 if there is no background class).
	/// @param nms_threshold Threshold for NMS step.
	/// @param top_k Maximum number of results to be kept in NMS.
	/// @param eta Used for adaptive NMS.
	/// @param code_type Type of coding method for bounding box.
	/// @param variance_encoded_in_target If true, variance is encoded in target; otherwise we need to adjust the predicted offset accordingly.
	/// @param confidence_threshold Only keep detections with confidences larger than this threshold.
    detection_output(
        const primitive_id& id,
        const primitive_id& input_location,
		const primitive_id& input_confidence,
		const primitive_id& input_prior_box,
		const uint32_t num_classes,
		const uint32_t keep_top_k,
		const bool share_location = true,
		const int background_label_id = 0,
		const float nms_threshold = 0.3,
		const int top_k = -1,
		const float eta = 1.f,
		const prior_box_code_type code_type = prior_box_code_type::corner,
		const bool variance_encoded_in_target = false,
		const float confidence_threshold = -std::numeric_limits<float>::max(),
        const padding& input_padding = padding(),
        const padding& output_padding = padding()
        )
        : primitive_base(id, { input_location, input_confidence, input_prior_box }, input_padding, output_padding)
		, num_classes(num_classes)
		, keep_top_k(keep_top_k)
		, share_location(share_location)
		, background_label_id(background_label_id)
		, nms_threshold(nms_threshold)
		, top_k(top_k)
		, eta(eta)
		, code_type(code_type)
		, variance_encoded_in_target(variance_encoded_in_target)
		, confidence_threshold(confidence_threshold)
    {}

	/// @brief Constructs a copy from C API @CLDNN_PRIMITIVE_DESC{detection_output}
	detection_output(const dto* dto)
        : primitive_base(dto)
		, num_classes(dto->num_classes)
		, keep_top_k(dto->keep_top_k)
		, share_location(dto->share_location != 0)
		, background_label_id(dto->background_label_id)
		, nms_threshold(dto->nms_threshold)
		, top_k(dto->top_k)
		, eta(dto->eta)
		, code_type(static_cast<prior_box_code_type>(dto->code_type))
		, variance_encoded_in_target(dto->variance_encoded_in_target != 0)
		, confidence_threshold(dto->confidence_threshold)
    {}

	/// @brief Number of classes to be predicted.
	const uint32_t num_classes;
	/// @brief Number of total bounding boxes to be kept per image after NMS step.
	const int keep_top_k;
	/// @brief If true, bounding box are shared among different classes.
	const bool share_location;
	/// @brief Background label id (-1 if there is no background class).
	const int background_label_id;
	/// @brief Threshold for NMS step.
	const float nms_threshold;
	/// @brief Maximum number of results to be kept in NMS.
	const int top_k;
	/// @brief Used for adaptive NMS.
	const float eta;
	/// @brief Type of coding method for bounding box.
	const prior_box_code_type code_type;
	/// @brief If true, variance is encoded in target; otherwise we need to adjust the predicted offset accordingly.
	const bool variance_encoded_in_target;
	/// @brief Only keep detections with confidences larger than this threshold.
	const float confidence_threshold;

protected:
	void update_dto(dto& dto) const override
	{
		dto.num_classes = num_classes;
		dto.share_location = share_location;
		dto.background_label_id = background_label_id;
		dto.nms_threshold = nms_threshold;
		dto.top_k = top_k;
		dto.eta = eta;
		dto.code_type = static_cast<int32_t>(code_type);
		dto.variance_encoded_in_target = variance_encoded_in_target;
		dto.keep_top_k = keep_top_k;
		dto.confidence_threshold = confidence_threshold;
	}
};
/// @}
/// @}
/// @}
}