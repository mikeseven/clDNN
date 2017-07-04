#include "ocl_user_event.h"

using namespace cldnn::gpu;

void user_event::set_impl()
{
    static_cast<cl::UserEvent&&>(get()).setStatus(CL_COMPLETE);
}
