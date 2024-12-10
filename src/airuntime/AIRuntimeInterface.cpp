#ifdef _WIN32

#include "../../../include/public/AIRuntimeInterface.h"
#include "../../../include/private/airuntime/AIRuntime.h"
#include "../../../include/private/airuntime/logger.h"
#if USE_TRT
#include "../../../include/private/trt/trt_common/trt_builder.hpp"
#endif

#elif __linux__
#include "../../include/public/AIRuntimeInterface.h"
#include "../../include/private/airuntime/AIRuntime.h"
#include "../../include/private/airuntime/logger.h"
#if USE_TRT
#include "../../include/private/trt/trt_common/trt_builder.hpp"
#endif
#endif //__LINUX__WIN32__

#if USE_TRT
bool build_model(int type, int max_batch_size, const char* onnx_path, const char* model_save_path, const size_t max_work_space_size)
{
    TRT::Mode model_type;
    if (type == 0) {
        model_type = TRT::Mode::FP32;
    }
    else if (type == 1) {
        model_type = TRT::Mode::FP16;
    }
    else {
        LOG_INFOE("Model type must in [0-1], current intput is {}", type);
        return false;
    }
    bool status = TRT::compile(model_type, max_batch_size, onnx_path, model_save_path, max_work_space_size);
    return status;
}
#elif  USE_ORT

bool build_model(int type, int max_batch_size, const char* onnx_path, const char* model_save_path, const size_t max_work_space_size)
{
    return true;
}
#endif
AIRuntimeInterface* GetAIRuntime()
{
    auto obj = AIRuntime::get_instance();
    return obj;
}
