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
#include "kernel.h"
#include "cache/primitive_db.h"

const std::string kernelName = "softmax_gpu";
const std::string kernelName2 = "softmax_gpu_batches";

// TODO: read this value from ocl device!!!
#define MAX_LWS 256

namespace neural {
namespace normalization {
struct softmax_gpu : is_an_implementation {
    softmax &outer;
    size_t _preferred_lws;
    size_t _preferred_gws;
    gpu::kernel _kernel;

    softmax_gpu(softmax &arg) : is_an_implementation(neural::type_id<softmax_gpu>())
        , outer(arg) 
        , _preferred_lws(0), _preferred_gws(0)
        , _kernel(select_kernel_name(), get_jit_constants())
    {}

    const std::string& select_kernel_name() const {
        return outer.argument.output_size.batch[0] == 1 ? kernelName : kernelName2;
    }

    gpu::jit_constants get_jit_constants() {
        auto batch_num = outer.argument.output_size.batch[0];

        auto dstSize = outer.output_memory(0).count();

        size_t items_num = 0;
        size_t leftovers = 0;

        if (batch_num == 1)
        {
            _preferred_lws = dstSize < 32 ? dstSize : 32;
            _preferred_gws = dstSize < 32 ? dstSize : dstSize - (dstSize % _preferred_lws);
            items_num = _preferred_gws / _preferred_lws;
            leftovers = dstSize < 32 ? 0 : dstSize % _preferred_lws;
        }
        else
        {
            _preferred_lws = batch_num;
            items_num = dstSize / _preferred_lws;
            while ( (items_num > 32 || _preferred_lws < items_num) && ((_preferred_lws << 1) <= MAX_LWS) )
            {
                _preferred_lws <<= 1;
                items_num >>= 1;
            }
            _preferred_gws = _preferred_lws;
            leftovers = dstSize % _preferred_lws;
        }

        assert(items_num > 0 && _preferred_lws > 0 && _preferred_gws > 0);

        return gpu::jit_constants{
            gpu::make_jit_constant("INPUT", outer.input_memory(0).argument.size),
            gpu::make_jit_constant("ITEMS_NUM", items_num),
            gpu::make_jit_constant("LWS", _preferred_lws),
            gpu::make_jit_constant("GWS", _preferred_gws),
            gpu::make_jit_constant("LEFTOVERS", leftovers)
        };
    }

    static void implementation(const void *ptr) {
        auto me = static_cast<const softmax_gpu*>(ptr);
        auto& outer = me->outer;

        assert(1 == outer.argument.output_size.feature.size());
        assert(1 == outer.argument.output_size.batch.size());

        auto& input_mem = outer.input_memory(0);
        auto& output_mem = outer.output_memory(0);

        me->_kernel.run<gpu::input_mem, gpu::output_mem>
            ({ me->_preferred_gws, me->_preferred_lws }, input_mem, output_mem);
    }

    static is_an_implementation *create(softmax &arg) { return new softmax_gpu(arg); };
    task_group work() override { return{ { task{ implementation, this } }, schedule::single }; };

};

namespace {
struct attach {
    attach() {
        auto key_fw = std::make_tuple(engine::gpu, memory::format::xb_f32, memory::format::xb_f32);
        auto val_fw = softmax_gpu::create;

        implementation_map<softmax>::add(key_fw, val_fw);
    }
    ~attach() {}
};
}

#ifdef __GNUC__
    __attribute__((visibility("default"))) //todo meybe dll_sym?
#elif _MSC_VER
#   pragma section(".nn_init$m", read, write)
#endif
attach attach_impl;

} // namespace normalization
} // namespace neural
