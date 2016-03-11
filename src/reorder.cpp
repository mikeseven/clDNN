#include "api/neural.h"
#include "multidimensional_counter.h"
#include <algorithm>
#include <tuple>
#include <map>
#include <functional>

namespace neural {

	namespace {

        class Dims{
            std::vector<char> order;
            memory::format::type format;

            Dims(memory::format::type _fmt, std::vector<char> _order):format(_fmt), order(_order){};

            static std::vector<char> getOrder(memory::format::type fmt){
                switch(fmt) {
                case memory::format::  xb_f32: return {'x','b'};
                case memory::format::yxfb_f32: return {'y','x','f','b'};
                case memory::format::fyxb_f32: return {'f','y','x','b'};
                case memory::format::xyfb_f32: return {'x','y','f','b'};
                case memory::format::fxyb_f32: return {'f','x','y','b'};
                case memory::format::byxf_f32: return {'b','y','x','f'};
                case memory::format::bfyx_f32: return {'b','f','y','x'};
                case memory::format::bxyf_f32: return {'b','x','y','f'};
		        case memory::format::bfxy_f32: return {'b','f','x','y'};
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
        public:
            static Dims create(memory::format::type fmt){
                return {fmt,getOrder(fmt)};
            };

            template <typename T>
            std::vector<T>&  translatePos(Dims _other) {
                return translatePos(_other.order,_other.format);
            };
            template <typename T>
            std::vector<T>&  translatePos(std::vector<T>& _srcPosition, const memory::format::type& _srcFormat) {
	            std::vector<T> tmp_pos(0);
	            for(int i =0; i< _srcPosition.size();++i){
		            auto o = this->order[i];
		            for(int j =0; j <  _srcPosition.size();++j)
		            {
			            auto p = getOrder(_srcFormat)[j];
			            if(o == p)
				            tmp_pos.push_back(_srcPosition[j]);
		            }
	            }
                _srcPosition.clear();
                for(auto p:tmp_pos) {
                    _srcPosition.push_back(p);
                }
                return _srcPosition;
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
				auto input = static_cast<float*>(this_reorder->input_memory(0).pointer);
				auto output = static_cast<float*>(this_reorder->output_memory(0).pointer);

				auto input_memory_arg  = this_reorder->input_memory(0).argument;
				auto input_format = input_memory_arg.format;

				auto output_memory_arg = this_reorder->output_memory(0).argument;
				auto output_format= output_memory_arg.format;

                if (input_format == output_format)
                    return;

				auto input_size = input_memory_arg.size;
				auto output_size= output_memory_arg.size;

				if(input_size.size() != output_size.size())throw std::runtime_error("Reorder input/output number of dimension does not match.");

				namespace nd = ndimensional;
				nd::value<uint32_t> range (output_size);
                Dims dimRange = Dims::create(output_format);

                //nd::calculate_idx<uint32_t> *calc_in_idx = nullptr;
                //if(input_format == memory::format::yxfb_f32 && output_format == memory::format::bfxy_f32)
                //    calc_in_idx = new nd::calculate_idx<uint32_t>({input_size[3],input_size[2],input_size[1],input_size[0]});
                //else
                //    calc_in_idx = new nd::calculate_idx<uint32_t>(output_size);
                nd::calculate_idx<uint32_t> calc_in_idx (input_size);
                Dims dimInput = Dims::create(input_format);
				nd::calculate_idx<uint32_t> calc_out_idx (output_size);
                Dims dimOutput = Dims::create(output_format);
				for(auto pos : range) {
					//auto in_idx  = calc_in_idx(pos);
                    auto in_idx  = calc_in_idx(dimInput.translatePos(dimRange));
                    //auto in_idx  = calc_in_idx((Dims::create(input_format)).translatePos(input_size,input_format));
					auto out_idx = calc_out_idx(pos);

					output[out_idx] = input[in_idx];
				}
			}

			std::vector<task> work() {
				return{ task{ implementation, &outer } };
			}

			static is_an_implementation *create(reorder &arg) { return new reorder_reference(arg); };
		};

		//                                    engine          output                  input
		using implementation_key = std::tuple<neural::engine::type, neural::memory::format::type, neural::memory::format::type>;

		// map of available implementations
		static std::map<implementation_key, std::function<is_an_implementation *(reorder &)>> implementation_map = {
			{ std::make_tuple(engine::reference, memory::format::yxfb_f32, memory::format::bfxy_f32), reorder_reference::create }
		};

	}
	reorder::arguments::arguments(neural::engine::type _engine, primitive_at _in, primitive _out)
		: engine(_engine)
		, output({_out})
		, input({_in}) {}
	reorder::arguments::arguments(neural::engine::type _engine, neural::memory::format::type _out_layout, std::vector<uint32_t> _out_sizes, primitive_at _in)
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