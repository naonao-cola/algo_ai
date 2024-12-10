#ifndef __AIRUNTIME_H__
#define __AIRUNTIME_H__

#include <map>
#include "./inference.h"
#include "../../public/AIRuntimeUtils.h"
#include "../../public/AIRuntimeDataStruct.h"
#include "../../public/AIRuntimeInterface.h"
#include "./AIRuntimeDefines.h"
#include "./MPMCQueue.h"


struct InferResult {
    std::vector<std::shared_future<Algo::BoxArray>> promise_rst;
    TaskInfoPtr taskInfo;
};
using InferResultPtr = std::shared_ptr<InferResult>;

class AIRuntime:public AIRuntimeInterface
{
public:
    DECLARE_SINGLETON(AIRuntime)

    eAIErrorCode InitRuntime(const stAIConfigInfo& cfg);
    eAIErrorCode DestoryRuntime();
    eAIErrorCode CreateModle(stAIModelInfo& modelInfo);
    eAIErrorCode CreateModle(json& modelInfo);
    eAIErrorCode UpdateModle(stAIModelInfo& newModelInfo);
    eAIErrorCode UpdateModleParam(const json& newModelInfo);
    eAIErrorCode DestroyModle(int modelID);
    stAIModelInfo::mPtr GetModelInfo(int modelId);
    eAIErrorCode CommitInferTask(TaskInfoPtr spTaskInfo);
    ModelResultPtr RunInferTask(TaskInfoPtr spTaskInfo);
    eAIErrorCode RegisterResultListener(int modelID, IModelResultListener* resultListener);
    eAIErrorCode UnregisterResultListener(IModelResultListener* resultListener);
    stGPUInfo GetGPUInfo(int modelID) { stGPUInfo gpuInfo; return gpuInfo; };

private:
    HIDE_CREATE_METHODS(AIRuntime);
    void TaskWorker();
    void CallbackWorker();
    void ResultListenWorker();
    std::vector<stResultItem> ToRuntimeResult(Algo::BoxArray& boxs, TaskInfoPtr taskInfo);
#if USE_TRT
    void create_trt_model(stAIModelInfo modelInfo, std::shared_ptr<Algo::Infer>& infer);
#endif //USE_TRT
#ifdef _WIN32
#ifdef USE_ORT
    void create_ort_model(stAIModelInfo modelInfo, std::shared_ptr<Algo::Infer>& infer);
#endif //USE_ORT
#elif __linux__
#endif

public:
    bool is_init{ false };
private:
    rigtorp::MPMCQueue<TaskInfoPtr> m_queueTask;
    rigtorp::MPMCQueue<ModelResultPtr> m_queueResult;
    rigtorp::MPMCQueue<InferResultPtr> m_rstResult;

    
    std::mutex m_lockTask;
    std::mutex m_lockInfer;
    std::mutex m_lockResult;
    std::condition_variable m_condTask;
    std::condition_variable m_condInfer;
    std::condition_variable m_condResult;

    std::thread mInferThread;
    std::thread mCallbackThread;
    std::thread mRstListenerThread;

    std::atomic_bool m_bStopped;
    std::vector<IModelResultListener*> m_callbackList;
    std::map<int, std::shared_ptr<Algo::Infer>> m_modelMap;
    std::map<int, stAIModelInfo> m_model_param;
    std::mutex m_retlist_lock;
    std::list<std::shared_future<Algo::BoxArray>> m_result_list;
    std::mutex m_init_lock;
};

#endif // __AIRUNTIME_H__