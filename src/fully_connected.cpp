#include "api/neural.h"
#include "multidimensional_counter.h"

namespace neural {

struct fully_connected_reference : is_an_implementation {
    const fully_connected &outer;
    fully_connected_reference(fully_connected &arg)
        : is_an_implementation(neural::type_id<fully_connected_reference>())
        , outer(arg)
    {};
    ~fully_connected_reference() {}

    static void implementation(const void *ptr) {
        auto this_ful_con = static_cast<const fully_connected *>(ptr);
        auto input = static_cast<float*>(this_ful_con->input_memory(0).pointer);
        auto output = static_cast<float*>(this_ful_con->output_memory(0).pointer);
        auto weight = static_cast<float*>(this_ful_con->argument.weight.as<const memory&>().pointer);
        auto weight_size = this_ful_con->argument.weight.as<const memory&>().argument.size;

        auto input_memory_arg = this_ful_con->input_memory(0).argument;
        auto input_buffer_size = input_memory_arg.size;
        auto input_size = this_ful_con->input_memory(0).argument.size;

        auto output_memory_arg = this_ful_con->output_memory(0).argument;
        auto output_buffer_size = output_memory_arg.size;
        auto output_size = this_ful_con->argument.output_size;

        if (input_buffer_size.size() != output_buffer_size.size())throw std::runtime_error("Fully connected input/output number of dimension does not match.");
        if (input_memory_arg.format != output_memory_arg.format)  throw std::runtime_error("Fully connected input/output data format does not match.");
        if (this_ful_con->argument.weight.as<const memory&>().argument.format != memory::format::xb_f32) throw std::runtime_error("Fully connected weight format is not xb_f32.");

        assert(this_ful_con->input_memory(0).argument.size.size()==1 || this_ful_con->input_memory(0).argument.size.size() == 2);

        // up-casts data format form 1D to 2D if necessery; DOES not copy memory, just redescribes 1D input buffer as 2D (x+batch) with batch=1
        auto mem_arg_in = this_ful_con->input_memory(0).argument;
        if(mem_arg_in.size.size()==1) {
            mem_arg_in.size.emplace_back(1);
            mem_arg_in.format = memory::format::xb_f32;
        }
        mem_arg_in.owns_memory = false;
        auto in_wrapper = memory::create(mem_arg_in);
        in_wrapper(input);

        auto mem_arg_out = this_ful_con->output_memory(0).argument;
        if (mem_arg_out.size.size() == 1) {
            mem_arg_out.size.emplace_back(1);
            mem_arg_out.format = memory::format::xb_f32;
        }
        mem_arg_out.owns_memory = false;
        auto out_wrapper = memory::create(mem_arg_out);
        out_wrapper(output);

        namespace nd = ndimensional;
        nd::value<uint32_t> range_output(output_size);
        nd::value<uint32_t> range_weight(weight_size);
        nd::value<uint32_t> range_input(input_size);
        nd::calculate_idx<uint32_t> calc_in_idx(input_buffer_size);
        nd::calculate_idx<uint32_t> calc_out_idx(output_buffer_size);
        nd::calculate_idx<uint32_t> calc_w_idx(weight_size);

        int data_index = 0; //todo type traits
        int batch_index = 1;

        for (auto pos_out : range_output){

            float acc = 0;
            auto out_idx = calc_out_idx(pos_out);

            for (auto pos_in : range_input){
                auto in_idx = calc_in_idx(pos_in);
                auto w_idx = calc_w_idx(std::vector<uint32_t>{ pos_out[data_index], pos_in[data_index] });
                acc += input[in_idx] * weight[w_idx];
            }
        output[out_idx] = acc;
        }
    }


    std::vector<task> work() {
        return{ task{ implementation, &outer } };
    }

    static is_an_implementation *create(fully_connected &arg) { return new fully_connected_reference(arg); };
};


//                                    engine                output                        input
using implementation_key = std::tuple<neural::engine::type, neural::memory::format::type, neural::memory::format::type>;

// map of available implementations
static std::map<implementation_key, std::function<is_an_implementation *(fully_connected &)>> implementation_map = {
    { std::make_tuple(engine::reference, memory::format::xb_f32, memory::format::xb_f32), fully_connected_reference::create }
};

/*fully_connected::arguments::arguments( neural::engine::type  eng,
                                       primitive             out,
                                       std::vector<uint32_t> out_off,
                                       std::vector<uint32_t> out_siz,
                                       primitive             in,
                                       std::vector<int32_t>  in_off,
                                       std::vector<uint32_t> in_str,
                                       primitive             weights)
: engine(eng)
, output({out})
, output_offset(out_off)
, output_size(out_siz)
, input({in})
, input_offset(in_off)
, input_stride(in_str)
, weight(weights){};*/

fully_connected::arguments::arguments( neural::engine::type  eng,
                                       primitive             out,
                                       primitive             in,
                                       primitive             weights)
: engine(eng)
, output({out})
, output_size(out.as<const memory&>().argument.size.begin(), out.as<const memory&>().argument.size.end())
, input({in})
, weight(weights){};

// creates primitive with fully_connected implementation that supports provided arguments
primitive fully_connected::create(fully_connected::arguments arg) {
    // wrap relu into RAII wrapper
    std::unique_ptr<fully_connected> result(new fully_connected(arg));

    // lookup in database; throw if not found
    auto key = std::make_tuple(arg.engine, result->input_memory(0).argument.format, result->output_memory(0).argument.format);
    auto it = implementation_map.find(key);
    if (it == std::end(implementation_map)) throw std::runtime_error("not yet implemented");

    // create implementation & attach it to result
    auto implementation = it->second(*result);
    result->_private.reset(implementation);
    result->_work = implementation->work();

    // release RAII wrapper, return naked pointer
    return result.release();
}

}