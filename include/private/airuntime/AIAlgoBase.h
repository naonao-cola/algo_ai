#ifndef __AIALGOBASE_H__
#define __AIALGOBASE_H__

#include <thread>
#include <condition_variable>
#include <mutex>

#include "../../public/AIRuntimeInterface.h"
#include "./AIAlgoDefines.h"
#include "./AIRuntime.h"
#include "./MPMCQueue.h"

using InspParamPtr = std::shared_ptr<void>;
using AlgoResultPtr = std::shared_ptr<void>;

class IAlgoResultListener
{
public:
    virtual void OnAlgoResult(AlgoResultPtr spResult) = 0;
};

/**
 * @brief AI算法基类，封装算法处理流程及并行调度,具体AI算法类需继承自该类并实现:
 * - 前处理(OnPreProcess)
 * - 后处理(OnPostProcess)
 */
class AIAlgoBase : public IModelResultListener
{
public:
    AIAlgoBase();
    virtual ~AIAlgoBase();
    // 启动AI算法检测
    eAIErrorCode StartRunAlgo(InspParamPtr spInspParam);
    // 初始化AI算法预处理、后处理资源
    void Initialize(int preThrdCnt, int preThrdPriority, int postThrdCnt, int postThrdPriority);
    // 释放AI算法相关资源(线程、队列)
    void DeInitialize();
    // 注册结果回调
    void RegisterAlgoResultListener(IAlgoResultListener* listener);
    // 取消注册结果回调
    void UnregisterAlgoResultListener(IAlgoResultListener* listener);
    
protected:
    // AI算法前处理接口(子类需实现)
    virtual void OnPreProcess(TaskInfoPtr spTaskInfo) = 0;
    // AI算法后处理接口(子类需实现)
    virtual void OnPostProcess(ModelResultPtr spResult, AlgoResultPtr& algoResult, HANDLE& hInspEnd) = 0;
    
    // AIRuntime 返回模型推理结果
    virtual void OnModelResult(ModelResultPtr spResult);

private:
    // 前处理线程运行函数
    void PreProcessWorker();
    // 前后理线程运行函数
    void PostProcessWorker();

private:
    atomic_bool m_bStopped;
    std::vector<std::thread*> m_vecPrepThrds;
    std::vector<std::thread*> m_vecPostThrds;
    rigtorp::MPMCQueue<InspParamPtr> m_queuePrep;
    rigtorp::MPMCQueue<ModelResultPtr> m_queuePost;
    std::vector<IAlgoResultListener*> m_vecResultListener;
    std::mutex m_queLock;
    std::condition_variable cond_;
protected:
    int m_nModelId;
};


#endif // __AIALGOBASE_H__