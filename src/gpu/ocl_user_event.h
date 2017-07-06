#pragma once

#include "ocl_base_event.h"

#ifdef _WIN32
#pragma warning(push)
#pragma warning(disable: 4250) //Visual Studio warns us about inheritance via dominance but it's done intentionally so turn it off
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
#pragma warning(pop)
#endif

} }