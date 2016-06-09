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

#include "api/neural.h"
#include "multidimensional_counter.h"
#include "reorder.h"
#include <algorithm>
#include <tuple>
#include <map>
#include <functional>
#include <cstring>

namespace neural {

namespace {
namespace nd=ndimensional;

struct Dim{
    memory::format::type format;

    Dim(memory::format::type _fmt):format(_fmt){};

    static std::vector<char> get_order(memory::format::type fmt){
        switch(fmt) {
        case memory::format::   x_f32: return {'x'};
        case memory::format::  xb_f32: return {'x','b'};
        case memory::format::yxfb_f32: return {'y','x','f','b'};
        case memory::format::byxf_f32: return {'b','y','x','f'};
        case memory::format::bfyx_f32: return {'b','f','y','x'};
        case memory::format::fyxb_f32: return {'f','y','x','b'};
        default: throw std::runtime_error("unknown memory::format");
        };
    };

    template <typename T>
    nd::value<T> translate_pos(nd::value<T> _srcPosition, Dim _dimSrc) {
        translate_pos(this->format,_srcPosition,_dimSrc.format);
    };

    template <typename T>
    static bool can_translate(memory::format::type _src_fmt_type, memory::format::type _dest_fmt_type) {
        auto src_fmt = memory::traits(_src_fmt_type);
        auto dest_fmt = memory::traits(_dest_fmt_type);
        return (src_fmt.dimension == dest_fmt.dimension) && (src_fmt.type->name == dest_fmt.type->name);
    };

    template <typename T>
    static nd::value<T> translate_pos(memory::format::type _fmtDest, const nd::value<T>& _srcPosition, memory::format::type _fmtSrc) {
        if (_fmtDest == _fmtSrc)
            return _srcPosition;
        if (!can_translate<T>(_fmtSrc,_fmtDest))
            throw std::runtime_error("cannot translate memory formats");

        nd::value<T>  tmp_pos(0);
        auto srcOrder = get_order(_fmtSrc);
        auto destOrder = get_order(_fmtDest);
        for(int i =0; i< _srcPosition.size();++i){
            auto od = destOrder[i];
            for(int j =0; j <  _srcPosition.size();++j){
                auto po = srcOrder[j];
                if(od == po)
                    tmp_pos.push_back(_srcPosition[j]);
            }
        }
        return tmp_pos;
    };
};

struct reorder_reference : is_an_implementation {
    const reorder &outer;
    reorder_reference(reorder &arg)
        : is_an_implementation(neural::type_id<reorder_reference>())
        , outer(arg)
    {};
    ~reorder_reference() {}

    static void implementation(const void *ptr) {
        auto this_reorder = static_cast<const reorder *>(ptr);

        auto& input_mem     = this_reorder->input_memory(0);
        auto& output_mem    = this_reorder->output_memory(0);

        float *input, *output;

        auto& input_memory_arg  = this_reorder->input_memory(0).argument;
        auto& input_format = input_memory_arg.format;

        auto& output_memory_arg = this_reorder->output_memory(0).argument;
        auto& output_format= output_memory_arg.format;

        if (input_format == output_format)
        {
            auto input_mem_ptr = input_mem.pointer;
            auto output_mem_ptr = output_mem.pointer;

            if(input_mem_ptr != output_mem_ptr)
                memcpy(output_mem_ptr, input_mem_ptr, this_reorder->output_memory(0).count() * memory::traits(output_format).type->size);

            return;
        }

        auto& input_size = input_memory_arg.size;
        auto& output_size= output_memory_arg.size;

        if(input_size.raw.size() != output_size.raw.size()) throw std::runtime_error("Reorder input/output number of dimension does not match.");

        namespace nd = ndimensional;
        nd::value<uint32_t> range (output_size);
        auto calc_in_ptr = nd::choose_calculate_ptr(input_mem);
        auto calc_out_ptr = nd::choose_calculate_ptr(output_mem);

        for(auto pos : range) {
            input  = static_cast<float*>(calc_in_ptr(input_mem, pos));
            output = static_cast<float*>(calc_out_ptr(output_mem, pos));

            *output = *input;
        }
    }

    task_group work() {
        return {{task{implementation, &outer}}, schedule::unordered};
    }

    static is_an_implementation *create(reorder &arg) { return new reorder_reference(arg); };
};


} // namespace {

reorder::arguments::arguments(neural::engine::type _engine, primitive _out, primitive_at _in)
    : engine(_engine)
    , output({_out})
    , input({_in}) {}
reorder::arguments::arguments(neural::engine::type _engine, neural::memory::format::type _out_layout, neural::vector<uint32_t> _out_sizes, primitive_at _in)
    : engine(_engine)
    , output( {memory::allocate({_engine, _out_layout, _out_sizes})} )
    , input({_in}) {}

// creates primitive with reorder implementation that supports provided arguments
primitive reorder::create(reorder::arguments arg) {
    // wrap reorder into RAII wrapper
    std::unique_ptr<reorder> result(new reorder(arg));

    // create implementation for non-lazy evaluation
    if(0 == (arg.engine & engine::lazy)) {
        // lookup in database; throw if not found
        auto key = std::make_tuple(arg.engine, result-> input_memory(0).argument.format, result->output_memory(0).argument.format);
        auto &map = reorder_fw_implementation_map::instance();
        auto it = map.find(key);

        // create implementation & attach it to result
        auto implementation = it!=map.end() ? it->second(*result) : reorder_reference::create(*result);
        result->_private.reset(implementation);
        result->_work = implementation->work();
    }

    // release RAII wrapper, return naked pointer
    return result.release();
}



}
