/*
// Copyright (c) 2018 Intel Corporation
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

#include "filler_inst.h"
#include "primitive_type_base.h"
#include "memory_impl.h"
#include <random>
#include "tests\test_utils\float16.h"
#include "error_handler.h"

namespace cldnn
{
    primitive_type_id filler_type_id()
    {
        static primitive_type_base<filler> instance;
        return &instance;
    }

    namespace {
        memory_impl::ptr attach_or_copy_data(network_impl& network, memory_impl& mem)
        {
            auto& engine = network.get_engine();
            if (mem.is_allocated_by(engine))
                return &mem;

            memory_impl::ptr result = engine.allocate_memory(mem.get_layout());
            mem_lock<char> src(mem);
            mem_lock<char> dst(result);
            std::copy(src.begin(), src.end(), dst.begin());
            return result;
        }
    }

    filler_node::typed_program_node(const std::shared_ptr<filler> dprim, program_impl& prog)
        : parent(dprim, prog), mem(api_cast(dprim->mem.get()))
    {
        recalc_output_layout(false);
        fill_memory();
    }

    void filler_node::attach_memory(memory_impl& new_mem, bool invalidate_users_if_changed)
    {
        mem = &new_mem;
        recalc_output_layout(invalidate_users_if_changed);
    }

    void filler_node::fill_memory()
    {
        auto d = get_primitive();
        switch (d->fill_type)
        {
        case filler::filler_type::xavier:
            fill_memory_xavier();
            break;
        default:
            break;
        }
    }

    void filler_node::fill_memory_xavier()
    {
        typedef std::chrono::high_resolution_clock clock;
        clock::time_point b = clock::now();
        auto memory = mem.get();
        auto layout = memory->get_layout();
        auto sizes = layout.size.sizes();
        int n = sizes[0];
        float scale = float(sqrt(3.0 / n));
        clock::duration time = clock::now() - b;
        unsigned seed = unsigned(time.count());
        std::default_random_engine generator(seed);
        if (layout.data_type == data_types::f16)
        {
            mem_lock<FLOAT16> lock(mem);
            std::uniform_real_distribution<float> distribution(-scale, scale);
            FLOAT16* new_mem = new FLOAT16[layout.count()];
            for (int i = 0; i < layout.count(); i++)
            {
                new_mem[i] = distribution(generator);
            }
            std::copy(new_mem, new_mem + layout.count(), lock.begin());
            delete[] new_mem;
        }
        else if (layout.data_type == data_types::f32)
        {
            mem_lock<float> lock(mem);
            std::uniform_real_distribution<float> distribution(-scale, scale);
            float* new_mem = new float[layout.count()];
            for (int i = 0; i < layout.count(); i++)
            {
                new_mem[i] = distribution(generator);
            }
            std::copy(new_mem, new_mem + layout.count(), lock.begin());
            delete[] new_mem;
        }
        else
        {
            CLDNN_ERROR_MESSAGE(id(), "only f32 and f16 data types can be filled");
        }
    }

    std::string filler_inst::to_string(filler_node const& node)
    {
        auto node_info = node.desc_to_json();

        std::stringstream primitive_description;

        node_info.dump(primitive_description);
        return primitive_description.str();
    }

    filler_inst::typed_primitive_inst(network_impl& network, filler_node const& node)
        : parent(network, node, *attach_or_copy_data(network, node.get_attached_memory()))
    {}

}
