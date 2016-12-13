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
#include "implementation_map.h"
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
            }

            template <typename T>
            static bool can_translate(memory::format::type _src_fmt_type, memory::format::type _dest_fmt_type) {
                auto src_fmt = memory::traits(_src_fmt_type);
                auto dest_fmt = memory::traits(_dest_fmt_type);
                return (src_fmt.dimension == dest_fmt.dimension) && (src_fmt.type->name == dest_fmt.type->name);
            }

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
           }
        };

   }
    reorder::arguments::arguments(primitive_at _in, primitive _out) //todo Artur fix arguments order
        : output({_out})
        , input({_in})
        , subtract_per_feature()
        , padding(get_memory_primitive(_out).argument.size.batch.size(),
            get_memory_primitive(_out).argument.size.feature.size(),
            get_memory_primitive(_out).argument.size.spatial.size()){}

	reorder::arguments::arguments(primitive _out, primitive _in, primitive subtract_values)
        : output({ _out })
        , input({ _in, subtract_values })
        , subtract_per_feature()
        , padding(get_memory_primitive(_out).argument.size.batch.size(),
            get_memory_primitive(_out).argument.size.feature.size(),
            get_memory_primitive(_out).argument.size.spatial.size()) {}


    reorder::arguments::arguments(primitive _out, primitive _in, const std::vector<float>& value_to_subtract, bool dummy)
        : output({ _out })
        , input({ _in })
        , subtract_per_feature(value_to_subtract)
        , dummy(dummy)
        , padding(get_memory_primitive(_out).argument.size.batch.size(),
            get_memory_primitive(_out).argument.size.feature.size(),
            get_memory_primitive(_out).argument.size.spatial.size()) {}


    reorder::arguments::arguments(neural::memory::format::type _out_layout, neural::vector<uint32_t> _out_sizes, primitive_at _in)
        : output( {memory::allocate({_out_layout, _out_sizes})} )
        , input({_in})
        , subtract_per_feature()
        , padding(get_memory_primitive(output[0]).argument.size.batch.size(),
            get_memory_primitive(output[0]).argument.size.feature.size(),
            get_memory_primitive(output[0]).argument.size.spatial.size()) {}


    reorder::arguments::arguments(neural::memory::format::type _out_layout, neural::vector<uint32_t> _out_sizes, primitive_at _in, primitive_at _subtract)
        : output({ memory::allocate({ _out_layout, _out_sizes }) })
        , input({ _in, _subtract })
        , subtract_per_feature()
        , padding(get_memory_primitive(output[0]).argument.size.batch.size(),
            get_memory_primitive(output[0]).argument.size.feature.size(),
            get_memory_primitive(output[0]).argument.size.spatial.size()) {}


#pragma message ("TODO!!! Remove dummy parameter from reorder class - this is due to bad design, need to change it!")
    reorder::arguments::arguments(neural::memory::format::type _out_layout, neural::vector<uint32_t> _out_sizes, primitive_at _in, const std::vector<float>& value_to_subtract, bool dummy)
        : output({ memory::allocate({ _out_layout, _out_sizes }) })
        , input({ _in })
        , subtract_per_feature(value_to_subtract)
        , dummy(dummy)
        , padding(get_memory_primitive(output[0]).argument.size.batch.size(),
            get_memory_primitive(output[0]).argument.size.feature.size(),
            get_memory_primitive(output[0]).argument.size.spatial.size()) {}


    reorder::arguments::arguments(uint32_t padX, uint32_t padY, neural::memory::format::type _out_layout, neural::vector<uint32_t> _out_sizes, primitive_at _in)
        : output({ memory::allocate({ _out_layout, _out_sizes }) })
        , input({ _in })
        , subtract_per_feature()
        , padding(get_memory_primitive(output[0]).argument.size.batch.size(),
            get_memory_primitive(output[0]).argument.size.feature.size(),
            get_memory_primitive(output[0]).argument.size.spatial.size()) {
        if (padding.spatial.size() > 0)
            padding.spatial[0] = padX;
        if (padding.spatial.size() > 1)
            padding.spatial[1] = padY;
    }

    // creates primitive with reorder implementation that supports provided arguments
    primitive reorder::create(reorder::arguments arg) 
	{
        auto& input_mem = get_memory_primitive(arg.input[0].primitive());
        if (input_mem.argument.size.raw.size() != arg.output[0].as<const memory&>().argument.size.raw.size())
            //            throw std::runtime_error("Number of dimensions in reorder does not match. Meybe you want to use reshape primitive?"); //todo reshape
            throw std::runtime_error("Number of dimensions in reorder does not match.");
        if (!arg.subtract_per_feature.empty())
        {
            if (input_mem.argument.size.feature.size() > 1)
            {
                throw std::runtime_error("Subtracting values work only for formats that have feature dimension == 1");
            }
            if (input_mem.argument.size.feature[0] != arg.subtract_per_feature.size())
                throw std::runtime_error("Number of features/channels in input does not match the number of features/channels in values to subtract");
        }
        return is_a_primitive::create<reorder>(arg);
    }

}
