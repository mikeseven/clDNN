#include "command_queues_builder.h"
#include "error_handler.h"

namespace cldnn { namespace gpu{

    command_queues_builder::command_queues_builder(const cl::Context& context, const cl::Device& device, const cl_platform_id& platform_id)
        : _context(context)
        , _device(device)
        , _platform_id(platform_id)
        , _priority_mode(cldnn_priority_disabled)
        , _throttle_mode(cldnn_throttle_disabled)
    {}

    cl_command_queue_properties command_queues_builder::get_properties()
    {
        cl_command_queue_properties ret = ((_profiling ? CL_QUEUE_PROFILING_ENABLE : 0) | (_out_of_order ? CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE : 0));
        return ret;
    }

    void command_queues_builder::build()
    {
        auto propeties = get_properties();
        if (_priority_mode != cldnn_priority_disabled)
        {
            build_with_priority_mode(propeties);
        }
        else
        {
            build_without_mode(propeties);
        }
    }

    void command_queues_builder::set_priority_mode(cldnn_priority_mode_type priority, bool extension_support)
    {
        if (priority != cldnn_priority_disabled && !extension_support)
        {
            CLDNN_ERROR_MESSAGE(
                "Command queues builders - priority_mode",
                "The param priority_mode is set in engine_configuration,\
                but cl_khr_priority_hints or cl_khr_create_command_queue\
                is not supported by current OpenCL implementation.");
        }
        _priority_mode = priority;
    }

    void command_queues_builder::build_with_priority_mode(cl_command_queue_properties& properties)
    {
        
        // TODO: When cl_khr_create_command_queue will be availible the
        // function name will change to clCreateCommandQueueWithPropertiesKHR
        // in place of clCreateCommandQueueWithPropertiesINTEL.
        pfn_clCreateCommandQueueWithPropertiesINTEL clCreateCommandQueueWithPropertiesINTEL =
            (pfn_clCreateCommandQueueWithPropertiesINTEL)clGetExtensionFunctionAddressForPlatform(
                _platform_id, "clCreateCommandQueueWithPropertiesINTEL");

        unsigned cl_queue_priority_value = CL_QUEUE_PRIORITY_MED_KHR;

        switch (_priority_mode)
        {
        case cldnn_priority_high:
            cl_queue_priority_value = CL_QUEUE_PRIORITY_HIGH_KHR;
            break;
        case cldnn_priority_low:
            cl_queue_priority_value = CL_QUEUE_PRIORITY_LOW_KHR;
            break;
        default:
            break;
        }

        cl_int error_code = CL_SUCCESS;
        cl_queue_properties properties_low[] = {
            CL_QUEUE_PRIORITY_KHR, cl_queue_priority_value,
            CL_QUEUE_PROPERTIES, properties,
            0 };

        _queue = clCreateCommandQueueWithPropertiesINTEL(
            _context.get(),
            _device.get(),
            properties_low,
            &error_code);

        if (error_code != CL_SUCCESS) {
            CLDNN_ERROR_MESSAGE("Command queues builders", "clCreateCommandQueueWithPropertiesINTEL error " + std::to_string(error_code));
        }
    }

    void command_queues_builder::build_without_mode(cl_command_queue_properties& properties)
    {
        _queue = cl::CommandQueue(_context, _device, properties);
    }

    void command_queues_builder::set_throttle_mode(cldnn_throttle_mode_type throttle, bool extensions_support)
    {
        if (throttle != cldnn_throttle_disabled)
        {
            CLDNN_ERROR_MESSAGE(
                "Command queues builders - throttle_mode",
                "The param throttle_mode is set in engine_configuration,\
                but it is placeholder for future use. It has no effect for now\
                and should be set to cldnn_throttle_disabled");
            if (!extensions_support)
            {
                CLDNN_ERROR_MESSAGE(
                    "Command queues builders - throttle_mode",
                    "The param throttle_mode is set in engine_configuration,\
                    but cl_khr_throttle_hints is not supported by current OpenCL implementation.");
            }
        }
    }
}
}

