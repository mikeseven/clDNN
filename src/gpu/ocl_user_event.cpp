#include "ocl_user_event.h"

using namespace cldnn::gpu;

void user_event::set_impl()
{
    //we simulate "wrapper_cast" here to cast from cl::Event to cl::UserEvent which both wrap the same cl_event
    //casting is valid as long as cl::UserEvent does not add any members to cl::Event (which it shouldn't)
    static_assert(sizeof(cl::UserEvent) == sizeof(cl::Event) && alignof(cl::UserEvent) == alignof(cl::Event), "cl::UserEvent does not match cl::Event");
    static_cast<cl::UserEvent&&>(get()).setStatus(CL_COMPLETE);
}
