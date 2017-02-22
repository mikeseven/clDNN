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

#include "neural_impl.h"
#include "engine_impl.h"
#include "network_impl.h"
#include "implementation_map.h"
#include "kernel.h"
#include "kd_selector.h"
#include "api/primitives/simpler_nms.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

using namespace cldnn;

namespace neural
{

static inline float clamp_v(const float v, const float v_min, const float v_max)
{
    return std::max(v_min, std::min(v, v_max));
}

struct simpler_nms_roi_t
{
    float x0, y0, x1, y1;

    float area() const { return std::max<float>(0, y1 - y0 + 1) * std::max<float>(0, x1 - x0 + 1); }
    simpler_nms_roi_t intersect (simpler_nms_roi_t other) const
    {
        return
        {
            std::max(x0, other.x0),
            std::max(y0, other.y0),
            std::min(x1, other.x1),
            std::min(y1, other.y1)
        };
    }
    simpler_nms_roi_t clamp (simpler_nms_roi_t other) const
    {
        return
        {
            clamp_v(x0, other.x0, other.x1),
            clamp_v(y0, other.y0, other.y1),
            clamp_v(x1, other.x0, other.x1),
            clamp_v(y1, other.y0, other.y1)
        };
    }
};

struct simpler_nms_delta_t { float shift_x, shift_y, log_w, log_h; };
struct simpler_nms_proposal_t { simpler_nms_roi_t roi; float confidence; size_t ord; };
        
simpler_nms_roi_t simpler_nms_gen_bbox( const anchor& box,
                                        const simpler_nms_delta_t& delta,
                                        int anchor_shift_x,
                                        int anchor_shift_y)
{
    float anchor_w = box.end_x - box.start_x + 1.0f;
    float anchor_h = box.end_y - box.start_y + 1;
    float center_x = box.start_x + anchor_w * .5f;
    float center_y = box.start_y + anchor_h *.5f;

    float pred_center_x = delta.shift_x * anchor_w + center_x + anchor_shift_x;
    float pred_center_y = delta.shift_y * anchor_h + center_y + anchor_shift_y;
    float half_pred_w = std::exp(delta.log_w) * anchor_w * .5f;
    float half_pred_h = std::exp(delta.log_h) * anchor_h * .5f;

    return { pred_center_x - half_pred_w,
             pred_center_y - half_pred_h,
             pred_center_x + half_pred_w,
             pred_center_y + half_pred_h };
}
        
std::vector< simpler_nms_roi_t > simpler_nms_perform_nms(
        const std::vector<simpler_nms_proposal_t>& proposals,
        float iou_threshold,
        size_t top_n)
{
//TODO(ruv): can I mark the 1st arg, proposals as const? ifndef DONT_PRECALC_AREA, i can
//TODO(ruv): is it better to do the precalc or not? since we need to fetch the floats from memory anyway for -
//           intersect calc, it's only a question of whether it's faster to do (f-f)*(f-f) or fetch another val
#define DONT_PRECALC_AREA

#ifndef DONT_PRECALC_AREA
    std::vector<float> areas;
    areas.reserve(proposals.size());
    std::transform(proposals.begin(), proposals.end(), areas.begin(), [](const simpler_nms_proposals_t>& v)
    {
        return v.roi.area();
    });
#endif

    std::vector<simpler_nms_roi_t> res;
    res.reserve(top_n);
#ifdef DONT_PRECALC_AREA
    for (const auto & prop : proposals)
    {
        const simpler_nms_roi_t& bbox = prop.roi;
        const float area = bbox.area();
#else
        size_t proposal_count = proposals.size();
        for (size_t proposalIndex = 0; proposalIndex < proposal_count; ++proposalIndex)
        {
            const simpler_nms_roi_t& bbox = proposals[proposalIndex].roi;
#endif
            // For any realistic WL, this condition is true for all top_n values anyway
            if (prop.confidence > 0)
            {
                bool overlaps = std::any_of(res.begin(), res.end(), [&](const simpler_nms_roi_t& res_bbox)
                {
                    float interArea = bbox.intersect(res_bbox).area();
#ifdef DONT_PRECALC_AREA
                    float unionArea = res_bbox.area() + area - interArea;
#else
                    float unionArea = res_bbox.area() + areas[proposalIndex] - interArea;
#endif
                    return interArea > iou_threshold * unionArea;
                });

                if (! overlaps)
                {
                    res.push_back(bbox);
                    if (res.size() == top_n) break;
                }
            }
        }

        return res;
    }


bool cmp_fn(const simpler_nms_proposal_t& a, const simpler_nms_proposal_t& b)
{
    return (a.confidence > b.confidence) || (a.confidence == b.confidence && a.ord > b.ord);
}


inline void sort_and_keep_n_items(std::vector<simpler_nms_proposal_t>& proposals, size_t n)
{
    if (proposals.size() > n)
    {
        std::partial_sort(proposals.begin(), proposals.begin() + n, proposals.end(), cmp_fn);
        proposals.resize(n);
    }
    else
    {
        std::sort(proposals.begin(), proposals.end(), cmp_fn);
    }        
}
        

template <>
struct kd_default_value_selector<neural::gpu::engine_info_internal::architectures>
{
    static constexpr neural::gpu::engine_info_internal::architectures value = neural::gpu::engine_info_internal::architectures::GEN_UNKNOWN;
};

template <>
struct kd_default_value_selector<neural::gpu::engine_info_internal::configurations>
{
    static constexpr neural::gpu::engine_info_internal::configurations value = neural::gpu::engine_info_internal::configurations::GT_UNKNOWN;
};

struct simpler_nms_gpu : is_an_implementation {
    const simpler_nms& _outer;
    gpu::engine_info_internal _engine_info;

    struct kernel_data 
    {
        size_t gws0, gws1, gws2; ///< Local work sizes (3D).
        size_t lws0, lws1, lws2; ///< Global work sizes (3D).
        std::string kernel_name;
        bool fp16_unit_used;
    } _kernel_data;
    gpu::kernel _kernel;

    static kd_selector_t<kernel_data, simpler_nms, neural::memory::format::type, kd_optional_selector_t, int, neural::gpu::engine_info_internal::architectures, neural::gpu::engine_info_internal::configurations> ks;

    simpler_nms_gpu(const simpler_nms& outer)
        : _outer(outer),
        _engine_info(outer.get_network().get_engine()->get_context()->get_engine_info()),
        _kernel_data(ks.get_kernel(outer, outer.input_memory(0).argument().format, outer.input_memory(0).argument().size.batch[0], _engine_info.architecture, _engine_info.configuration)),
        _kernel(_outer.get_network().get_engine()->get_context(), _kernel_data.kernel_name, get_jit_constants(_outer, _kernel_data))
    {}

    static kernel_data set_default(const simpler_nms& outer)
    {
        kernel_data kd;

		cldnn::data_types input_dt = outer.input_memory(cldnn::simpler_nms_arg::cls_scores_index).get_layout().data_type;
		assert(input_dt == data_types::f32);

		kd.fp16_unit_used = (input_dt == cldnn::data_types::f16);

        // Determine global work sizes.
        kd.gws0 = 1;
        kd.gws1 = 1;
        kd.gws2 = 1;

        // Find largest positive local work size that is divider for global work size.
        kd.lws0 = 1;
        kd.lws1 = 1;
        kd.lws2 = 1;

        kd.kernel_name = "warm_up_gpu";

        return kd;
    }

    static gpu::jit_constants get_jit_constants(const simpler_nms& outer, const kernel_data& data)
    {   
        gpu::jit_constants foo{
            gpu::make_jit_constant("INPUT", outer.input_memory(0).argument().size),
			gpu::make_jit_constant("UNIT_TYPE", data.fp16_unit_used ? "half" : "float")
        };

        return foo;     
    }

    cldnn::refcounted_obj_ptr<cldnn::event_impl> execute(const std::vector<cldnn::refcounted_obj_ptr<cldnn::event_impl>>& events) override
    {
		for (auto a : events) {
			a->wait();
		}

        const std::vector<anchor>& anchors = _outer.get_anchors();

        size_t anchors_num = anchors.size();
      
        const cldnn::memory& cls_scores = _outer.input_memory(cldnn::simpler_nms_arg::cls_scores_index);  
        const cldnn::memory& bbox_pred  = _outer.input_memory(cldnn::simpler_nms_arg::bbox_pred_index);  
		const cldnn::memory& image_info = _outer.input_memory(cldnn::simpler_nms_arg::image_info_index);

        // feat map sizes
        int fm_h = cls_scores.get_layout().size.sizes()[2];
        int fm_w = cls_scores.get_layout().size.sizes()[3];
        
        int fm_sz = fm_w * fm_h;

        // original input image to the graph (after possible scaling etc.) so that coordinates are valid for it
		const int* image_info_mem = image_info.pointer<int>().data();

		int img_w = image_info_mem[cldnn::simpler_nms_arg::image_info_width_index];
        int img_h = image_info_mem[cldnn::simpler_nms_arg::image_info_height_index];
		int img_z = image_info_mem[cldnn::simpler_nms_arg::image_info_depth_index];

        int scaled_min_bbox_size = _outer.argument.min_bbox_size * img_z;

        float* cls_scores_mem = cls_scores.pointer<float>().data();
        float* bbox_pred_mem  = bbox_pred.pointer<float>().data();

        std::vector<simpler_nms_proposal_t> sorted_proposals_confidence;
        for (int y = 0; y < fm_h; ++y)
        {
            int anchor_shift_y = y * _outer.argument.feature_stride;

            for (int x = 0; x < fm_w; ++x)
            {
                int anchor_shift_x = x * _outer.argument.feature_stride;
                int location_index = y * fm_w + x;

                // we assume proposals are grouped by window location
                for (unsigned int anchor_index = 0; anchor_index < anchors_num ; anchor_index++)
                {
                    float dx0 = bbox_pred_mem[location_index + fm_sz * (anchor_index * 4 + 0)];
                    float dy0 = bbox_pred_mem[location_index + fm_sz * (anchor_index * 4 + 1)];
                    float dx1 = bbox_pred_mem[location_index + fm_sz * (anchor_index * 4 + 2)];
                    float dy1 = bbox_pred_mem[location_index + fm_sz * (anchor_index * 4 + 3)];
                    simpler_nms_delta_t bbox_delta { dx0, dy0, dx1, dy1 };

                    float proposal_confidence = cls_scores_mem[location_index + fm_sz * (anchor_index + anchors_num * 1)];

                    simpler_nms_roi_t tmp_roi = simpler_nms_gen_bbox(anchors[anchor_index], bbox_delta, anchor_shift_x, anchor_shift_y);
                    simpler_nms_roi_t roi = tmp_roi.clamp({ 0, 0, float(img_w - 1), float(img_h - 1) });

                    int bbox_w = (int)roi.x1 - (int)roi.x0 + 1;
                    int bbox_h = (int)roi.y1 - (int)roi.y0 + 1;

                    if (bbox_w >= scaled_min_bbox_size && bbox_h >= scaled_min_bbox_size)
                    {
                        simpler_nms_proposal_t proposal { roi, proposal_confidence, sorted_proposals_confidence.size() };
                        sorted_proposals_confidence.push_back(proposal);
                    }
                }
            }
        }

        sort_and_keep_n_items(sorted_proposals_confidence, _outer.argument.pre_nms_topn);
        std::vector< simpler_nms_roi_t > res = simpler_nms_perform_nms(sorted_proposals_confidence, _outer.argument.iou_threshold, _outer.argument.post_nms_topn);

        const cldnn::memory& output = _outer.output_memory();
        float* top_data = output.pointer<float>().data();        

        size_t res_num_rois = res.size();
        for (size_t i = 0; i < res_num_rois; ++i)
        {
            top_data[5 * i + 0] = 0;    // roi_batch_ind, always zero on test time
            top_data[5 * i + 1] = res[i].x0;
            top_data[5 * i + 2] = res[i].y0;
            top_data[5 * i + 3] = res[i].x1;
            top_data[5 * i + 4] = res[i].y1;
        }
       
        cldnn::event_impl* ev = _outer.get_network().get_engine().get()->create_user_event();
        ev->set();
        return ev;
    }

    static is_an_implementation *create(simpler_nms &arg) 
    {        
        return new simpler_nms_gpu(arg);
    }
};


simpler_nms_gpu::kernel_data defauly_bfyx_f32(const simpler_nms& arg)
{
    simpler_nms_gpu::kernel_data kd = simpler_nms_gpu::set_default(arg);

    return kd;
}

kd_selector_t<simpler_nms_gpu::kernel_data, simpler_nms, neural::memory::format::type, kd_optional_selector_t, int, neural::gpu::engine_info_internal::architectures, neural::gpu::engine_info_internal::configurations> simpler_nms_gpu::ks = {
    { std::make_tuple(memory::format::bfyx_f32, 0, gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), defauly_bfyx_f32 },
};

namespace
{

    struct attach
    {
        attach()
        {
            implementation_map<simpler_nms>::add(std::make_tuple(cldnn::engine_types::ocl, memory::format::bfyx_f32), simpler_nms_gpu::create);
        }

        ~attach()
        {
        }
    };

#ifdef __GNUC__
    __attribute__((visibility("default")))
#elif _MSC_VER
#   pragma section(".nn_init$m", read, write)
#endif
    attach attach_impl;

}
}
