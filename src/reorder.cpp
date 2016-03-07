#include "api/neural.h"
#include "multidimensional_counter.h"
#include <algorithm>
#include <tuple>
#include <map>
#include <functional>

namespace neural {

	namespace {
        template <typename T>
            class Axis<T> {
                memory::format::type fmt;
        };

        size_t reorder_reference::Axis operator()(memory::format::type fmt){

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

                nd::calculate_idx<uint32_t> *calc_in_idx = nullptr;
                if(input_format == memory::format::yxfb_f32 && output_format == memory::format::bfxy_f32)
                    calc_in_idx = new nd::calculate_idx<uint32_t>({input_size[3],input_size[2],input_size[1],input_size[0]});
                else
                    calc_in_idx = new nd::calculate_idx<uint32_t>(output_size);

				nd::calculate_idx<uint32_t> calc_out_idx (output_size);
				for(auto pos : range) {
					auto in_idx  = (*calc_in_idx)(pos);
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