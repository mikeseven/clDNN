#include "neural.h"
#include "multidimensional_counter.h"
#include <algorithm>
#include <tuple>
#include <map>
#include <functional>

namespace neural {

	namespace {

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
			{ std::make_tuple(engine::reference, memory::format::yxfb_f32, memory::format::xyfb_f32), reorder_reference::create }
		};

	} 
	reorder::arguments::arguments(neural::engine::type, neural::memory::format::type, primitive_at in)
		: engine(engine)
		, input({in}) {}
	
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