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
#include <algorithm>
#include <tuple>
#include <map>
#include <functional>

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
                case memory::format::fyxb_f32: return {'f','y','x','b'};
                case memory::format::xyfb_f32: return {'x','y','f','b'};
                case memory::format::fxyb_f32: return {'f','x','y','b'};
                case memory::format::byxf_f32: return {'b','y','x','f'};
                case memory::format::bfyx_f32: return {'b','f','y','x'};
                case memory::format::bxyf_f32: return {'b','x','y','f'};
                case memory::format::bfxy_f32: return {'b','f','x','y'};
                case memory::format::   x_f64: return {'x'};
                case memory::format::yxfb_f64: return {'y','x','f','b'};
                case memory::format::fyxb_f64: return {'f','y','x','b'};
                case memory::format::xyfb_f64: return {'x','y','f','b'};
                case memory::format::fxyb_f64: return {'f','x','y','b'};
                case memory::format::byxf_f64: return {'b','y','x','f'};
                case memory::format::bfyx_f64: return {'b','f','y','x'};
                case memory::format::bxyf_f64: return {'b','x','y','f'};
                case memory::format::bfxy_f64: return {'b','f','x','y'};
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
                //auto input = static_cast<float*>(this_reorder->input_memory(0).pointer);
                //auto output = static_cast<float*>(this_reorder->output_memory(0).pointer);
                auto input = static_cast<float*>(this_reorder->argument.input[0].primitive.as<const memory&>().pointer);
                auto output = static_cast<float*>(this_reorder->argument.output[0].as<const memory&>().pointer);

                //auto& input_memory_arg  = this_reorder->input_memory(0).argument;
                auto& input_memory_arg = this_reorder->argument.input[0].primitive.as<const memory&>().argument;
                auto& input_format = input_memory_arg.format;

                //auto output_memory_arg = this_reorder->output_memory(0).argument;
                auto& output_memory_arg = this_reorder->argument.output[0].as<const memory&>().argument;
                auto output_format= output_memory_arg.format;
                //auto range_format= output_memory_arg.format;

                if (input_format == output_format)
                    return;

                auto& input_size = input_memory_arg.size;
                auto& output_size= output_memory_arg.size;

                if(input_size.raw.size() != output_size.raw.size()) throw std::runtime_error("Reorder input/output number of dimension does not match.");

                namespace nd = ndimensional;
                nd::value<uint32_t> range (output_size);
                auto calc_in_idx = nd::choose_calculate_idx(input_format);
                auto calc_out_idx = nd::choose_calculate_idx(output_format);

                for(auto pos : range) {
                    auto in_idx  = calc_in_idx(input_size.raw, pos);
                    auto out_idx = calc_out_idx(output_size.raw, pos);

                    output[out_idx] = input[in_idx];
                }
            }

            task_package work() {
                return{ task{ implementation, &outer } };
            }

            static is_an_implementation *create(reorder &arg) { return new reorder_reference(arg); };
        };

        //                                    engine                input                       output
        using implementation_key = std::tuple<neural::engine::type, neural::memory::format::type, neural::memory::format::type>;

        // map of available implementations
        static std::map<implementation_key, std::function<is_an_implementation *(reorder &)>> implementation_map = {
///// f32
            { std::make_tuple(engine::reference, memory::format::yxfb_f32, memory::format::yxfb_f32), reorder_reference::create },
            { std::make_tuple(engine::reference, memory::format::yxfb_f32, memory::format::fyxb_f32), reorder_reference::create },
            { std::make_tuple(engine::reference, memory::format::yxfb_f32, memory::format::xyfb_f32), reorder_reference::create },
            { std::make_tuple(engine::reference, memory::format::yxfb_f32, memory::format::fxyb_f32), reorder_reference::create },
            { std::make_tuple(engine::reference, memory::format::yxfb_f32, memory::format::byxf_f32), reorder_reference::create },
            { std::make_tuple(engine::reference, memory::format::yxfb_f32, memory::format::bfyx_f32), reorder_reference::create },
            { std::make_tuple(engine::reference, memory::format::yxfb_f32, memory::format::bxyf_f32), reorder_reference::create },
            { std::make_tuple(engine::reference, memory::format::yxfb_f32, memory::format::bfxy_f32), reorder_reference::create },

            { std::make_tuple(engine::reference, memory::format::fyxb_f32, memory::format::yxfb_f32), reorder_reference::create },
            { std::make_tuple(engine::reference, memory::format::fyxb_f32, memory::format::fyxb_f32), reorder_reference::create },
            { std::make_tuple(engine::reference, memory::format::fyxb_f32, memory::format::xyfb_f32), reorder_reference::create },
            { std::make_tuple(engine::reference, memory::format::fyxb_f32, memory::format::fxyb_f32), reorder_reference::create },
            { std::make_tuple(engine::reference, memory::format::fyxb_f32, memory::format::byxf_f32), reorder_reference::create },
            { std::make_tuple(engine::reference, memory::format::fyxb_f32, memory::format::bfyx_f32), reorder_reference::create },
            { std::make_tuple(engine::reference, memory::format::fyxb_f32, memory::format::bxyf_f32), reorder_reference::create },
            { std::make_tuple(engine::reference, memory::format::fyxb_f32, memory::format::bfxy_f32), reorder_reference::create },

            { std::make_tuple(engine::reference, memory::format::xyfb_f32, memory::format::yxfb_f32), reorder_reference::create },
            { std::make_tuple(engine::reference, memory::format::xyfb_f32, memory::format::fyxb_f32), reorder_reference::create },
            { std::make_tuple(engine::reference, memory::format::xyfb_f32, memory::format::xyfb_f32), reorder_reference::create },
            { std::make_tuple(engine::reference, memory::format::xyfb_f32, memory::format::fxyb_f32), reorder_reference::create },
            { std::make_tuple(engine::reference, memory::format::xyfb_f32, memory::format::byxf_f32), reorder_reference::create },
            { std::make_tuple(engine::reference, memory::format::xyfb_f32, memory::format::bfyx_f32), reorder_reference::create },
            { std::make_tuple(engine::reference, memory::format::xyfb_f32, memory::format::bxyf_f32), reorder_reference::create },
            { std::make_tuple(engine::reference, memory::format::xyfb_f32, memory::format::bfxy_f32), reorder_reference::create },

            { std::make_tuple(engine::reference, memory::format::fxyb_f32, memory::format::yxfb_f32), reorder_reference::create },
            { std::make_tuple(engine::reference, memory::format::fxyb_f32, memory::format::fyxb_f32), reorder_reference::create },
            { std::make_tuple(engine::reference, memory::format::fxyb_f32, memory::format::xyfb_f32), reorder_reference::create },
            { std::make_tuple(engine::reference, memory::format::fxyb_f32, memory::format::fxyb_f32), reorder_reference::create },
            { std::make_tuple(engine::reference, memory::format::fxyb_f32, memory::format::byxf_f32), reorder_reference::create },
            { std::make_tuple(engine::reference, memory::format::fxyb_f32, memory::format::bfyx_f32), reorder_reference::create },
            { std::make_tuple(engine::reference, memory::format::fxyb_f32, memory::format::bxyf_f32), reorder_reference::create },
            { std::make_tuple(engine::reference, memory::format::fxyb_f32, memory::format::bfxy_f32), reorder_reference::create },

            { std::make_tuple(engine::reference, memory::format::byxf_f32, memory::format::yxfb_f32), reorder_reference::create },
            { std::make_tuple(engine::reference, memory::format::byxf_f32, memory::format::fyxb_f32), reorder_reference::create },
            { std::make_tuple(engine::reference, memory::format::byxf_f32, memory::format::xyfb_f32), reorder_reference::create },
            { std::make_tuple(engine::reference, memory::format::byxf_f32, memory::format::fxyb_f32), reorder_reference::create },
            { std::make_tuple(engine::reference, memory::format::byxf_f32, memory::format::byxf_f32), reorder_reference::create },
            { std::make_tuple(engine::reference, memory::format::byxf_f32, memory::format::bfyx_f32), reorder_reference::create },
            { std::make_tuple(engine::reference, memory::format::byxf_f32, memory::format::bxyf_f32), reorder_reference::create },
            { std::make_tuple(engine::reference, memory::format::byxf_f32, memory::format::bfxy_f32), reorder_reference::create },

            { std::make_tuple(engine::reference, memory::format::bfyx_f32, memory::format::yxfb_f32), reorder_reference::create },
            { std::make_tuple(engine::reference, memory::format::bfyx_f32, memory::format::fyxb_f32), reorder_reference::create },
            { std::make_tuple(engine::reference, memory::format::bfyx_f32, memory::format::xyfb_f32), reorder_reference::create },
            { std::make_tuple(engine::reference, memory::format::bfyx_f32, memory::format::fxyb_f32), reorder_reference::create },
            { std::make_tuple(engine::reference, memory::format::bfyx_f32, memory::format::byxf_f32), reorder_reference::create },
            { std::make_tuple(engine::reference, memory::format::bfyx_f32, memory::format::bfyx_f32), reorder_reference::create },
            { std::make_tuple(engine::reference, memory::format::bfyx_f32, memory::format::bxyf_f32), reorder_reference::create },
            { std::make_tuple(engine::reference, memory::format::bfyx_f32, memory::format::bfxy_f32), reorder_reference::create },

            { std::make_tuple(engine::reference, memory::format::fyxb_f32, memory::format::yxfb_f32), reorder_reference::create },
            { std::make_tuple(engine::reference, memory::format::fyxb_f32, memory::format::fyxb_f32), reorder_reference::create },
            { std::make_tuple(engine::reference, memory::format::fyxb_f32, memory::format::xyfb_f32), reorder_reference::create },
            { std::make_tuple(engine::reference, memory::format::fyxb_f32, memory::format::fxyb_f32), reorder_reference::create },
            { std::make_tuple(engine::reference, memory::format::fyxb_f32, memory::format::byxf_f32), reorder_reference::create },
            { std::make_tuple(engine::reference, memory::format::fyxb_f32, memory::format::bfyx_f32), reorder_reference::create },
            { std::make_tuple(engine::reference, memory::format::fyxb_f32, memory::format::bxyf_f32), reorder_reference::create },
            { std::make_tuple(engine::reference, memory::format::fyxb_f32, memory::format::bfxy_f32), reorder_reference::create },

            { std::make_tuple(engine::reference, memory::format::bfxy_f32, memory::format::yxfb_f32), reorder_reference::create },
            { std::make_tuple(engine::reference, memory::format::bfxy_f32, memory::format::fyxb_f32), reorder_reference::create },
            { std::make_tuple(engine::reference, memory::format::bfxy_f32, memory::format::xyfb_f32), reorder_reference::create },
            { std::make_tuple(engine::reference, memory::format::bfxy_f32, memory::format::fxyb_f32), reorder_reference::create },
            { std::make_tuple(engine::reference, memory::format::bfxy_f32, memory::format::byxf_f32), reorder_reference::create },
            { std::make_tuple(engine::reference, memory::format::bfxy_f32, memory::format::bfyx_f32), reorder_reference::create },
            { std::make_tuple(engine::reference, memory::format::bfxy_f32, memory::format::bxyf_f32), reorder_reference::create },
            { std::make_tuple(engine::reference, memory::format::bfxy_f32, memory::format::bfxy_f32), reorder_reference::create },
///// f64
            { std::make_tuple(engine::reference, memory::format::yxfb_f64, memory::format::yxfb_f64), reorder_reference::create },
            { std::make_tuple(engine::reference, memory::format::yxfb_f64, memory::format::fyxb_f64), reorder_reference::create },
            { std::make_tuple(engine::reference, memory::format::yxfb_f64, memory::format::xyfb_f64), reorder_reference::create },
            { std::make_tuple(engine::reference, memory::format::yxfb_f64, memory::format::fxyb_f64), reorder_reference::create },
            { std::make_tuple(engine::reference, memory::format::yxfb_f64, memory::format::byxf_f64), reorder_reference::create },
            { std::make_tuple(engine::reference, memory::format::yxfb_f64, memory::format::bfyx_f64), reorder_reference::create },
            { std::make_tuple(engine::reference, memory::format::yxfb_f64, memory::format::bxyf_f64), reorder_reference::create },
            { std::make_tuple(engine::reference, memory::format::yxfb_f64, memory::format::bfxy_f64), reorder_reference::create },

            { std::make_tuple(engine::reference, memory::format::fyxb_f64, memory::format::yxfb_f64), reorder_reference::create },
            { std::make_tuple(engine::reference, memory::format::fyxb_f64, memory::format::fyxb_f64), reorder_reference::create },
            { std::make_tuple(engine::reference, memory::format::fyxb_f64, memory::format::xyfb_f64), reorder_reference::create },
            { std::make_tuple(engine::reference, memory::format::fyxb_f64, memory::format::fxyb_f64), reorder_reference::create },
            { std::make_tuple(engine::reference, memory::format::fyxb_f64, memory::format::byxf_f64), reorder_reference::create },
            { std::make_tuple(engine::reference, memory::format::fyxb_f64, memory::format::bfyx_f64), reorder_reference::create },
            { std::make_tuple(engine::reference, memory::format::fyxb_f64, memory::format::bxyf_f64), reorder_reference::create },
            { std::make_tuple(engine::reference, memory::format::fyxb_f64, memory::format::bfxy_f64), reorder_reference::create },

            { std::make_tuple(engine::reference, memory::format::xyfb_f64, memory::format::yxfb_f64), reorder_reference::create },
            { std::make_tuple(engine::reference, memory::format::xyfb_f64, memory::format::fyxb_f64), reorder_reference::create },
            { std::make_tuple(engine::reference, memory::format::xyfb_f64, memory::format::xyfb_f64), reorder_reference::create },
            { std::make_tuple(engine::reference, memory::format::xyfb_f64, memory::format::fxyb_f64), reorder_reference::create },
            { std::make_tuple(engine::reference, memory::format::xyfb_f64, memory::format::byxf_f64), reorder_reference::create },
            { std::make_tuple(engine::reference, memory::format::xyfb_f64, memory::format::bfyx_f64), reorder_reference::create },
            { std::make_tuple(engine::reference, memory::format::xyfb_f64, memory::format::bxyf_f64), reorder_reference::create },
            { std::make_tuple(engine::reference, memory::format::xyfb_f64, memory::format::bfxy_f64), reorder_reference::create },

            { std::make_tuple(engine::reference, memory::format::fxyb_f64, memory::format::yxfb_f64), reorder_reference::create },
            { std::make_tuple(engine::reference, memory::format::fxyb_f64, memory::format::fyxb_f64), reorder_reference::create },
            { std::make_tuple(engine::reference, memory::format::fxyb_f64, memory::format::xyfb_f64), reorder_reference::create },
            { std::make_tuple(engine::reference, memory::format::fxyb_f64, memory::format::fxyb_f64), reorder_reference::create },
            { std::make_tuple(engine::reference, memory::format::fxyb_f64, memory::format::byxf_f64), reorder_reference::create },
            { std::make_tuple(engine::reference, memory::format::fxyb_f64, memory::format::bfyx_f64), reorder_reference::create },
            { std::make_tuple(engine::reference, memory::format::fxyb_f64, memory::format::bxyf_f64), reorder_reference::create },
            { std::make_tuple(engine::reference, memory::format::fxyb_f64, memory::format::bfxy_f64), reorder_reference::create },

            { std::make_tuple(engine::reference, memory::format::byxf_f64, memory::format::yxfb_f64), reorder_reference::create },
            { std::make_tuple(engine::reference, memory::format::byxf_f64, memory::format::fyxb_f64), reorder_reference::create },
            { std::make_tuple(engine::reference, memory::format::byxf_f64, memory::format::xyfb_f64), reorder_reference::create },
            { std::make_tuple(engine::reference, memory::format::byxf_f64, memory::format::fxyb_f64), reorder_reference::create },
            { std::make_tuple(engine::reference, memory::format::byxf_f64, memory::format::byxf_f64), reorder_reference::create },
            { std::make_tuple(engine::reference, memory::format::byxf_f64, memory::format::bfyx_f64), reorder_reference::create },
            { std::make_tuple(engine::reference, memory::format::byxf_f64, memory::format::bxyf_f64), reorder_reference::create },
            { std::make_tuple(engine::reference, memory::format::byxf_f64, memory::format::bfxy_f64), reorder_reference::create },

            { std::make_tuple(engine::reference, memory::format::bfyx_f64, memory::format::yxfb_f64), reorder_reference::create },
            { std::make_tuple(engine::reference, memory::format::bfyx_f64, memory::format::fyxb_f64), reorder_reference::create },
            { std::make_tuple(engine::reference, memory::format::bfyx_f64, memory::format::xyfb_f64), reorder_reference::create },
            { std::make_tuple(engine::reference, memory::format::bfyx_f64, memory::format::fxyb_f64), reorder_reference::create },
            { std::make_tuple(engine::reference, memory::format::bfyx_f64, memory::format::byxf_f64), reorder_reference::create },
            { std::make_tuple(engine::reference, memory::format::bfyx_f64, memory::format::bfyx_f64), reorder_reference::create },
            { std::make_tuple(engine::reference, memory::format::bfyx_f64, memory::format::bxyf_f64), reorder_reference::create },
            { std::make_tuple(engine::reference, memory::format::bfyx_f64, memory::format::bfxy_f64), reorder_reference::create },

            { std::make_tuple(engine::reference, memory::format::fyxb_f64, memory::format::yxfb_f64), reorder_reference::create },
            { std::make_tuple(engine::reference, memory::format::fyxb_f64, memory::format::fyxb_f64), reorder_reference::create },
            { std::make_tuple(engine::reference, memory::format::fyxb_f64, memory::format::xyfb_f64), reorder_reference::create },
            { std::make_tuple(engine::reference, memory::format::fyxb_f64, memory::format::fxyb_f64), reorder_reference::create },
            { std::make_tuple(engine::reference, memory::format::fyxb_f64, memory::format::byxf_f64), reorder_reference::create },
            { std::make_tuple(engine::reference, memory::format::fyxb_f64, memory::format::bfyx_f64), reorder_reference::create },
            { std::make_tuple(engine::reference, memory::format::fyxb_f64, memory::format::bxyf_f64), reorder_reference::create },
            { std::make_tuple(engine::reference, memory::format::fyxb_f64, memory::format::bfxy_f64), reorder_reference::create },

            { std::make_tuple(engine::reference, memory::format::bfxy_f64, memory::format::yxfb_f64), reorder_reference::create },
            { std::make_tuple(engine::reference, memory::format::bfxy_f64, memory::format::fyxb_f64), reorder_reference::create },
            { std::make_tuple(engine::reference, memory::format::bfxy_f64, memory::format::xyfb_f64), reorder_reference::create },
            { std::make_tuple(engine::reference, memory::format::bfxy_f64, memory::format::fxyb_f64), reorder_reference::create },
            { std::make_tuple(engine::reference, memory::format::bfxy_f64, memory::format::byxf_f64), reorder_reference::create },
            { std::make_tuple(engine::reference, memory::format::bfxy_f64, memory::format::bfyx_f64), reorder_reference::create },
            { std::make_tuple(engine::reference, memory::format::bfxy_f64, memory::format::bxyf_f64), reorder_reference::create },
            { std::make_tuple(engine::reference, memory::format::bfxy_f64, memory::format::bfxy_f64), reorder_reference::create },
        };

    }
    reorder::arguments::arguments(neural::engine::type _engine, primitive_at _in, primitive _out)
        : engine(_engine)
        , output({_out})
        , input({_in}) {}
    reorder::arguments::arguments(neural::engine::type _engine, neural::memory::format::type _out_layout, neural::vector<uint32_t> _out_sizes, primitive_at _in)
        : engine(_engine)
        , output( {memory::create({_engine, _out_layout, _out_sizes, true})} )
        , input({_in}) {}

    // creates primitive with reorder implementation that supports provided arguments
    primitive reorder::create(reorder::arguments arg) {
        // wrap reorder into RAII wrapper
        std::unique_ptr<reorder> result(new reorder(arg));

        // lookup in database; throw if not found
        auto key = std::make_tuple(arg.engine, result->input_memory(0).argument.format, result->output_memory(0).argument.format);
        auto it = implementation_map.find(key);
        if (it == std::end(implementation_map)) throw std::runtime_error("not yet implemented");

        // create implementation & attach it to result
        auto implementation = it->second(*result);
        result->_work = implementation->work();

        // release RAII wrapper, return naked pointer
        return result.release();
    }

}