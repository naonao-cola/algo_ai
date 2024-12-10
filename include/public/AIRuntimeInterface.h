#ifndef __AIRUNTIMEINTERFACE_H__
#define __AIRUNTIMEINTERFACE_H__

#include "./AIRuntimeDataStruct.h"
#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#   define __export         __declspec(dllexport)
#elif defined(__GNUC__) && ((__GNUC__ >= 4) || (__GNUC__ == 3 && __GNUC_MINOR__ >= 3))
#   define __export         __attribute__((visibility("default")))
#else
#   define __export
#endif

//tensorrt
#define USE_TENSORRT

class IModelResultListener
{
public:
    virtual void OnModelResult(ModelResultPtr spResult) = 0;
};

class  AIRuntimeInterface {
public:
    virtual eAIErrorCode InitRuntime(const stAIConfigInfo& cfg) = 0;
    virtual eAIErrorCode DestoryRuntime() = 0;
    virtual eAIErrorCode CreateModle(stAIModelInfo& modelInfo) = 0;
    virtual eAIErrorCode CreateModle(json& modelInfo) = 0;
    virtual eAIErrorCode UpdateModle(stAIModelInfo& newModelInfo) = 0;

    /**
     * Set the model param.
     * 
     * \param newModelInfo  data struct:
     * {
     *      "modelID": xx,
     *      "confidence_threshold": xx,
     *      "nms_threshold": 0.5
     *      "max_objects": 100
     * }
     * \return eAIErrorCode
     */
    virtual eAIErrorCode UpdateModleParam(const json& newModelInfo) = 0;
    virtual eAIErrorCode DestroyModle(int modelID) = 0;
    virtual stAIModelInfo::mPtr GetModelInfo(int modelId)=0;
    virtual eAIErrorCode CommitInferTask(TaskInfoPtr spTaskInfo) = 0;
    virtual ModelResultPtr RunInferTask(TaskInfoPtr spTaskInfo) = 0;
    virtual eAIErrorCode RegisterResultListener(int modelID, IModelResultListener* resultListener) = 0;
    virtual eAIErrorCode UnregisterResultListener(IModelResultListener* resultListener) = 0;
    virtual stGPUInfo GetGPUInfo(int modelID) = 0;
};

__export AIRuntimeInterface* GetAIRuntime();

/**
 * @brief Compile onnx model to tensorRT engine model
 * model_type               model type [0: FP32, 1 : FP16]
 * max_batch_size           max batch size
 * onnx_path                onnx onnx file path
 * model_save_path          tensorRT engine file save path
 * max_work_space_size      tensorRT work space size. The default size is 1GB.
 */
__export bool   build_model(int model_type,int max_batch_size,const char* onnx_path,const char* model_save_path,const size_t max_work_space_size = 1ul<<30);

#ifdef __cplusplus
}
#endif

#endif // __AIRUNTIMEINTERFACE_H__
