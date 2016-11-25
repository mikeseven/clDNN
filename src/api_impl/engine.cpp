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

///////////////////////////////////////////////////////////////////////////////////////////////////
#include "api/topology.hpp"
#include "api/cldnn.hpp"
#include "refcounted_obj.h"
#include "topology_impl.h"
#include "engine_impl.h"
#include "network_impl.h"
#include <algorithm>

namespace cldnn
{
class network_impl;
//namespace impl {
//    class primitive
//    {
//    public:
//
//        virtual ~primitive()
//        {
//        }
//        memory input_memory(size_t index) const { return _inputs.at(index)->output_memory(); }
//        memory output_memory() const { return _output; }
//        engine get_engine() const;
//
//        std::shared_ptr<const primitive> argument() const { return _arg; }
//
//    protected:
//        primitive(network_impl* net, std::shared_ptr<const primitive> arg, memory output);
//
//    private:
//        network_impl* _network;
//        std::shared_ptr<const primitive> _arg;
//        std::vector<std::shared_ptr<primitive>> _inputs;
//        memory _output;
//    };
//}
//
//
//namespace impl {
//    primitive::primitive(network_impl* net, std::shared_ptr<const primitive> arg, memory output)
//        : _network(net)
//        , _arg(arg)
//        , _output(output)
//    {
//        for (auto& i : arg->get_dto()->input)
//        {
//            _inputs.push_back(_network->get_primitive(i));
//        }
//    }
//
//    engine primitive::get_engine() const
//    {
//        return _network->get_engine();
//    }
//
//    class input : public primitive
//    {
//    public:
//        input(network_impl* net, std::shared_ptr<const primitive_desc> arg)
//            : primitive(net, arg, net->get_engine().allocate_memory(arg->get_dto()->as<primitive_types::input>()->layout))
//        {}
//    };
//
//    class convolution : public primitive
//    {
//    public:
//        layout calc_output_layout(network_impl* net, std::shared_ptr<const primitive_desc> arg)
//        {
//            auto conv_dto = arg->get_dto()->as<primitive_types::convolution>();
//            auto input_layout = net->get_primitive(conv_dto->input[0])->output_memory().get_layout();
//            auto weights_layout = net->get_primitive(conv_dto->weigths[0])->output_memory().get_layout();
//            auto input_offset = conv_dto->input_offset.transform(format::yx, 0);
//            auto strd = conv_dto->stride.transform(format::yx, 0);
//            auto split = conv_dto->weigths.size();
//
//            // compute how many outputs in rows and columns will be generate by filter. 
//            // outp <= (input_size - (2*input_offset) - kernel_size)/ stride 
//            auto kernel_xy = weights_layout.size.spatial;
//            assert(kernel_xy.size() == 2);
//            auto output_spatial_x = (input_layout.size.spatial[0] - (2 * input_offset.spatial[0]) - kernel_xy[0]) / strd.spatial[0] + 1;
//            auto output_spatial_y = (input_layout.size.spatial[1] - (2 * input_offset.spatial[1]) - kernel_xy[1]) / strd.spatial[1] + 1;
//            // get output feature map from weights. It should be the same as number of biases. Will be verifed in convolution::create()
//            auto number_of_features = weights_layout.size.feature[0] * static_cast<int32_t>(split);
//
//            tensor output_size(format::yxfb, {
//                input_layout.size.batch[0], number_of_features, output_spatial_x, output_spatial_y }
//            );
//
//            return{ input_layout.data_type, output_size.transform(input_layout.size.format, 1) };
//        }
//
//        convolution(network_impl* net, std::shared_ptr<const primitive_desc> arg)
//            : primitive(net, arg, net->get_engine().allocate_memory(calc_output_layout(net, arg)))
//        {}
//    };
//}
//

context engine::get_context()
{
    return _impl->get_context();
}

engine::engine(const engine& other):_impl(other._impl)
{
    _impl->add_ref();
}

engine& engine::operator=(const engine& other)
{
    if (_impl == other._impl) return *this;
    _impl->release();
    _impl = other._impl;
    _impl->add_ref();
    return *this;
}

engine::~engine()
{
    _impl->release();
}

buffer* engine::allocate_buffer(layout layout, status_t* status) noexcept
{
    try
    {
        if (status)
            *status = CLDNN_SUCCESS;
        return _impl->allocate_buffer(layout);
    }
    catch (...)
    {
        if (status)
            *status = CLDNN_ERROR;
        return nullptr;
    }
}

network_impl* engine::build_network_impl(topology topology, status_t* status) noexcept
{
    if (topology.get_context() != get_context())
    {
        if (status)
            *status = CLDNN_ERROR;
        return nullptr;
    }

    try
    {
        if (status)
            *status = CLDNN_SUCCESS;
        return new network_impl(*this, topology);
    }
    catch (...)
    {
        if (status)
            *status = CLDNN_ERROR;
        return nullptr;
    }
}
}
