/*
// Copyright (c) 2017 Intel Corporation
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

#include <vector>

#include "simpler_nms.h"
#include "../primitive.hpp"

namespace cldnn
{
    
struct simpler_nms : public primitive_base<simpler_nms, CLDNN_PRIMITIVE_DESC(simpler_nms)>
{
    CLDNN_DECLATE_PRIMITIVE(simpler_nms)
 
    simpler_nms(
        const primitive_id& id,        
        const primitive_id& cls_scores,
        const primitive_id& bbox_pred,
		const primitive_id& image_info,
        int max_proposals,
        float iou_threshold,
        int min_bbox_size,
        int feature_stride,
        int pre_nms_topn,
        int post_nms_topn,
		const std::vector<float>& scales_param,
        const padding& input_padding = padding(),
        const padding& output_padding = padding()
        )
        : primitive_base(id, {cls_scores, bbox_pred, image_info}, input_padding, output_padding, 
                         max_proposals,
                         iou_threshold,
                         min_bbox_size,
                         feature_stride,
                         pre_nms_topn,
                         post_nms_topn,
                         cldnn_float_arr{nullptr, 0}),
                 max_proposals(_dto.max_proposals),
                 iou_threshold(_dto.iou_threshold),
                 min_bbox_size(_dto.min_bbox_size),
                 feature_stride(_dto.feature_stride),
                 pre_nms_topn(_dto.pre_nms_topn),
                 post_nms_topn(_dto.post_nms_topn),
                 scales(scales_param)
    {
        init_dto();
    }

    simpler_nms(const dto* dto) :
        primitive_base(dto),
        max_proposals(_dto.max_proposals),
        iou_threshold(_dto.iou_threshold),
        min_bbox_size(_dto.min_bbox_size),
        feature_stride(_dto.feature_stride),
        pre_nms_topn(_dto.pre_nms_topn),
        post_nms_topn(_dto.post_nms_topn),
        scales(float_arr_to_vector(_dto.scales))
    {
        init_dto();
    }

    int max_proposals;
    float iou_threshold;
    int min_bbox_size;
    int feature_stride;
    int pre_nms_topn;
    int post_nms_topn;      
    const std::vector<float> scales;

    private:

    void init_dto()
    {
        _dto.scales = float_vector_to_arr(scales);
    }
};

}