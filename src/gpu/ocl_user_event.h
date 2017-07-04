#pragma once

#include "ocl_base_event.h"

#ifdef _WIN32
#pragma warning(disable: 4250)
#endif

namespace cldnn { namespace gpu {

struct user_event : public base_event, public cldnn::user_event
{
    user_event(cl::UserEvent const& ev, bool auto_set = false) : base_event(ev), cldnn::user_event(auto_set)
    {
        if (auto_set)
            user_event::set_impl();
    }

    void set_impl() override;
};

#ifdef _WIN32
#pragma warning(default: 4250)
#endif

} }