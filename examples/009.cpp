#include "api/neural.h"

// memory->memory reshape
void example_009() {
    using namespace neural;

    const uint32_t dim_y      = 6, dim_x      = 7, dim_f      = 3, dim_b      = 3;
	const uint32_t all_size = dim_y*dim_x*dim_f*dim_b;

    float in_buffer[all_size];
    float out_buffer[all_size];

	auto in_sizes = { dim_y, dim_x, dim_f, dim_b };
	auto out_sizes = { dim_b, dim_f, dim_x, dim_y };

	auto in_layout = memory::format::yxfb_f32;
    auto out_layout = memory::format::bfxy_f32;
	// input buffer should be initialized with valid data

	
    auto input  = memory::create({engine::cpu, in_layout, in_sizes});
    auto output = memory::create({engine::cpu, out_layout, out_sizes});
	
    auto act    = reorder::create(reorder::arguments{engine::reference,out_layout,out_sizes,input});

    execute({input(in_buffer), output(out_buffer), act});
}
