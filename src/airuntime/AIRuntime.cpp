// #include <Windows.h>
#ifdef _WIN32
    #include <windows.h>
#elif __linux__
#endif
// #include <filesystem>
#ifdef _WIN32
    #include <filesystem>
    namespace fs = std::filesystem;
#else __linux__
//#if __cplusplus >= 202403L
    #include <experimental/filesystem>
    namespace fs = std::experimental::filesystem;
//#endif //__cplusplus >= 202403L
#endif
#ifdef _WIN32

#include "../../../include/private/airuntime/AIRuntime.h"
#include "../../../include/private/airuntime/logger.h"
#if USE_TRT
#include "../../../include/private/trt/trt_common/trt_infer.hpp"
#include "../../../include/private/trt/trt_models.h"
#endif // USE_TRT
#ifdef USE_ORT
#include "../../../include/private/ort/ort_common/ort_infer.hpp"
#include "../../../include/private/ort/ort_models.h"
#endif // USE_ORT
namespace fs = std::filesystem;
#elif __linux__
#include "../../include/private/airuntime/AIRuntime.h"
#include "../../include/private/airuntime/logger.h"
#include "../../include/private/trt/trt_common/trt_infer.hpp"
#include "../../include/private/trt/trt_models.h"

#include "../../include/private/ort/ort_common/ort_infer.hpp"
#include "../../include/private/ort/ort_models.h"

#endif
AIRuntime::AIRuntime()
    : m_queueTask(1000), m_queueResult(1000), m_rstResult(1000), m_bStopped(false) {}

AIRuntime::~AIRuntime()
{
    DestoryRuntime();
}

eAIErrorCode AIRuntime::InitRuntime(const stAIConfigInfo& cfg)
{
    m_init_lock.lock();
    if (!this->is_init) {
        this->is_init = true;
        m_bStopped    = false;//fix bug 如果不置为false，Destory之后再次InitRun会导致无法推理
        LOG_INFO("Starting initRuntime...");
        mInferThread       = std::thread(&AIRuntime::TaskWorker, this);
        mCallbackThread    = std::thread(&AIRuntime::CallbackWorker, this);
        mRstListenerThread = std::thread(&AIRuntime::ResultListenWorker, this);
        LOG_INFO("InitRuntime success !");
    }
    else {
        LOG_INFOW("AI runtime has been initialized!");
    }
    m_init_lock.unlock();
    return E_OK;
}

eAIErrorCode AIRuntime::DestoryRuntime()
// {
//    if (!m_bStopped) {
//         m_bStopped = true;
//         mInferThread.join();
//         mCallbackThread.join();
//         mRstListenerThread.join();
//    }
//     return E_OK;
// }
{
   if (!m_bStopped) {
        m_bStopped = true;
        this->is_init = false;

        m_condTask.notify_all();
        m_condResult.notify_all();
        m_condInfer.notify_all();
        mInferThread.join();
        mCallbackThread.join();
        mRstListenerThread.join();

        for (auto iter = this->m_model_param.begin(); iter != this->m_model_param.end();) {
                    this->m_model_param.erase(iter++);
        }//delete modelInfo
        for (auto iter = this->m_modelMap.begin(); iter != this->m_modelMap.end();) {
            this->m_modelMap.erase(iter++);
        }//delete model
   }
    return E_OK;
}
eAIErrorCode AIRuntime::CreateModle(json& modelInfo)
{
    auto err_code = E_OK;
    for (auto model : modelInfo) {
        stAIModelInfo model_info(model);
        auto          rst_code = CreateModle(model_info);
        err_code               = rst_code != E_OK ? rst_code : E_OK;
    }
    return err_code;
};

eAIErrorCode AIRuntime::UpdateModleParam(const json& newModelInfo)
{
    int modelID = get_param<int>(newModelInfo, "modelID", -1);
    LOG_INFO("Start UpdateModleParam. model id is : {}", modelID);
    if (this->m_modelMap.count(modelID) > 0) {
        this->m_modelMap[modelID]->set_param(newModelInfo);
        LOG_INFO("Finshed set model: {} paramter!", modelID);
    }
    else {
        LOG_INFOW("Not found model {}", modelID);
    }
    return E_OK;
}

stAIModelInfo::mPtr AIRuntime::GetModelInfo(int modelId)
{
    if (this->m_model_param.count(modelId)) {
        return std::make_shared<stAIModelInfo>(m_model_param[modelId]);
    }
    return nullptr;
}

eAIErrorCode AIRuntime::DestroyModle(int modelID)
{
    LOG_INFO("Start destroy model:{}", modelID);
    if (this->m_modelMap.count(modelID) > 0) {
        this->m_modelMap.erase(modelID);
    }
    if (this->m_model_param.count(modelID)) {
        this->m_model_param.erase(modelID);
    }
    LOG_INFO("Start destroy model: {} success!", modelID);
    return E_OK;
}

eAIErrorCode
AIRuntime::RegisterResultListener(int modelID, IModelResultListener* resultListener)
{
    LOG_INFO("[AIRuntime] RegisterResultListener()");
    m_callbackList.push_back(resultListener);
    return E_OK;
}
eAIErrorCode
AIRuntime::UnregisterResultListener(IModelResultListener* resultListener)
{
    LOG_INFO("[AIRuntime] RegisterResultListener()");
    if (!m_callbackList.empty())
        m_callbackList.pop_back();
    return E_OK;
}

eAIErrorCode AIRuntime::CommitInferTask(TaskInfoPtr spTaskInfo)
{
    if (!m_queueTask.try_push(spTaskInfo)) {
        LOG_INFOW("[AIRuntime] Commit inference task fail!");
        return E_QUEUUE_FULL;
    }
    else {
        LOG_INFO("[AIRuntime] Commit Infer task, TaskId: {}, task queue size: {}, "
                 "waiting queue size: {}, result queue size: {}",
                 spTaskInfo->taskId, m_queueTask.size(), m_rstResult.size(), m_queueResult.size());
        m_condTask.notify_one();
        return E_OK;
    }
}

std::vector<stResultItem> AIRuntime::ToRuntimeResult(Algo::BoxArray& boxs, TaskInfoPtr taskInfo)
{
    std::vector<stResultItem> rst_list;
    for (const Algo::Box& box : boxs) {
        stResultItem rst;
        switch (m_model_param[taskInfo->modelId].algoType) {
            case CLASSIFY:
                rst.code       = box.class_label;
                rst.confidence = box.confidence;
                break;
            case YOLOV5:
                rst.code       = box.class_label;
                rst.confidence = box.confidence;
                rst.points.push_back(stPoint(box.left, box.top));
                rst.points.push_back(stPoint(box.right, box.bottom));
                break;
            case OCR_REC:
                rst.confidence = box.confidence;
                if (box.ocr_ret_str != "") {
                    rst.confidence = box.ocr_ret_score;
                    rst.ocr_str    = box.ocr_ret_str;
                }
                if (!box.rec_conf_res.empty()) {
                    for (const auto& element : box.rec_conf_res) {
                        rst.rec_conf_res.push_back(element);

                    }
                    for (const auto& elementratio : box.rec_ratio_res) {
                        rst.res_ratio_res.push_back(elementratio);
                    }
                    for (const auto& element_single_pos : box.ocr_single_pos) {
                        rst.char_single_pos.push_back(element_single_pos);
                    }
                    for (const auto& char_index : box.ocr_char_index) {
                        rst.ocr_char_index.push_back(char_index);
                    }
                } 
                break;
            case OCR_DET:
                rst.code       = box.class_label;
                rst.confidence = box.confidence;
                if (box.ocr_det.size() > 0) {
                    rst.points.push_back(stPoint(box.ocr_det[0][0], box.ocr_det[0][1]));
                    rst.points.push_back(stPoint(box.ocr_det[1][0], box.ocr_det[1][1]));
                    rst.points.push_back(stPoint(box.ocr_det[2][0], box.ocr_det[2][1]));
                    rst.points.push_back(stPoint(box.ocr_det[3][0], box.ocr_det[3][1]));
                    if (!box.detMat.empty()) {
                        box.detMat.copyTo(rst.ocr_det);
                    }
                }
                break;
            case OCR_CLS:
                rst.code       = box.class_label;
                rst.confidence = box.confidence;
                break;
            case YOLO8:
                rst.code       = box.class_label;
                rst.confidence = box.confidence;
                rst.points.push_back(stPoint(box.left, box.top));
                rst.points.push_back(stPoint(box.right, box.bottom));
                break;
            case MSAE:
                rst.code       = box.class_label;
                rst.confidence = box.confidence;
                if (!box.msae_img.empty()) {
                    rst.msae_img = box.msae_img.clone();
                    // 12/26  hjf
                    rst.points.push_back(stPoint(box.left, box.top));
                    rst.points.push_back(stPoint(box.right, box.bottom));
                }
                break;
            case ANOMALIB:
                rst.code       = box.class_label;
                rst.confidence = box.confidence;
                rst.points.push_back(stPoint(box.left, box.top));
                rst.points.push_back(stPoint(box.right, box.bottom));
                if (!box.heatMap.empty()) {  // ANOMALIB
                    rst.heatMap  = box.heatMap.clone();
                    rst.msae_img = box.msae_img.clone();
                }
                break;
            case YOLO8_OBB:
                rst.confidence = box.confidence;
                rst.points.push_back(stPoint(box.center_x, box.center_y));
                rst.points.push_back(stPoint(box.width, box.height));
                rst.angle = box.angle;
                rst.code  = box.class_label;
                break;
            case YOLOV8_SEG:
                rst.confidence = box.confidence;
                if (box.contour.size() > 0) {
                    rst.mask.push_back(box.contour);
                }
                rst.points.push_back(stPoint(box.left, box.top));
                rst.points.push_back(stPoint(box.right, box.bottom));
                rst.code  = box.class_label;
                break;
            default:
                break;
        }
        rst_list.push_back(rst);
    }
    return rst_list;
}

ModelResultPtr AIRuntime::RunInferTask(TaskInfoPtr spTaskInfo)
{
    ModelResultPtr spResult;
    if (this->m_modelMap.count(spTaskInfo->modelId) == 0) {
        LOG_INFOE("Model: {} not exists!", spTaskInfo->modelId);
        return spResult;
    }
    if (spTaskInfo->imageData.size() == 0) {
        LOG_INFOE("RunInferTask fail! no image data.");
        return spResult;
    }

    spTaskInfo->tt.start();
    InferResultPtr inferRst = std::make_shared<InferResult>();
    auto rst = m_modelMap[spTaskInfo->modelId]->commits(spTaskInfo->imageData);
    spTaskInfo->tt.stop();

    spResult = std::make_shared<stModelResult>();
    spResult->taskInfo = spTaskInfo;
    for (auto pr : rst) {
        auto boxs = pr.get();
        spResult->itemList.emplace_back(ToRuntimeResult(boxs, spTaskInfo));
        spResult->taskInfo->preCostTime = boxs.pre_time;
        spResult->taskInfo->inferCostTime = boxs.infer_time;
        spResult->taskInfo->hostCostTime = boxs.host_time;
        spResult->taskInfo->totalCostTime = boxs.total_time;
    }
    return spResult;
}
void AIRuntime::TaskWorker()
{
    while (true) {
        // {} ��������������lock���ͷţ�����commit_task�ᱻ����
        {
            std::unique_lock<std::mutex> lock(m_lockTask);
            m_condTask.wait(lock, [&]() { return m_bStopped || !m_queueTask.empty(); });
        }

        TaskInfoPtr spTaskInfo;
        bool        found = m_queueTask.try_pop(spTaskInfo);

        if (m_bStopped)
            break;

        if (!found) {
            continue;
        }

        std::vector<std::shared_future<Algo::BoxArray>> rst_list;
        if (this->m_modelMap.count(spTaskInfo->modelId) == 0) {
            LOG_INFOE("Model: {} not exists!", spTaskInfo->modelId);
            continue;
        }
        spTaskInfo->tt.start();

        if (spTaskInfo->imageData.size() >= 1) {
            InferResultPtr inferRst = std::make_shared<InferResult>();
            auto           rst =
                m_modelMap[spTaskInfo->modelId]->commits(spTaskInfo->imageData);
            inferRst->promise_rst = rst;
            inferRst->taskInfo    = spTaskInfo;
            while (!m_rstResult.try_push(inferRst)) {
                LOG_INFOW(" enqueue m_rstResult fail! TaskId: {}, queue size: {}", spTaskInfo->taskId, m_rstResult.size());
                // Sleep(3);
            }
            m_condInfer.notify_one();
            LOG_INFOD("enqueue m_rstResult, \tTaskId: {}, \tqueue size: {}", spTaskInfo->taskId, m_rstResult.size());
        }
        if (m_bStopped)
            break;
    }
}
void AIRuntime::ResultListenWorker()
{
    while (true) {
        // {} ��������������lock���ͷţ�����queue.try_push�ᱻ����
        {
            std::unique_lock<std::mutex> lock(m_lockInfer);
            m_condInfer.wait(lock, [&]() { return m_bStopped || !m_rstResult.empty(); });
        }
        InferResultPtr inferRst;

        bool found = m_rstResult.try_pop(inferRst);
        // bool found = m_rstResult.wait_dequeue_timed(inferRst,
        // std::chrono::milliseconds(1000));
        if (m_bStopped)
            break;

        if (!found) {
            // std::cout << "[CallbackWorker] wait model Result timeout!" <<
            // std::endl;
            continue;
        }
        ModelResultPtr spResult = std::make_shared<stModelResult>();
        spResult->taskInfo      = inferRst->taskInfo;
        spResult->taskInfo->tt.stop();
        for (auto pr : inferRst->promise_rst) {
            auto boxs = pr.get();
            spResult->itemList.emplace_back(ToRuntimeResult(boxs, inferRst->taskInfo));
            spResult->taskInfo->preCostTime   = boxs.pre_time;
            spResult->taskInfo->inferCostTime = boxs.infer_time;
            spResult->taskInfo->hostCostTime  = boxs.host_time;
            spResult->taskInfo->totalCostTime = boxs.total_time;
        }

        if (inferRst->taskInfo->promiseResult != nullptr) {
            std::promise<ModelResultPtr>* promiseRst =static_cast<std::promise<ModelResultPtr>*>(inferRst->taskInfo->promiseResult);
            try {
                promiseRst->set_value(spResult);
            }
            catch (const std::exception& e) {
                std::cout << e.what() << std::endl;
                promiseRst->set_exception(std::current_exception());
            }
            continue;
        }
        while (!m_queueResult.try_push(spResult)) {
            LOG_INFOW("Enqueue task result fail. taskId: [{}], m_queueResult size:{}", inferRst->taskInfo->taskId, m_queueResult.size());
            // Sleep(10);
        }
        m_condResult.notify_one();
        LOG_INFOD("Enqueue m_queueResult success. \ttaskId: [{}], \tm_queueResult ""size:{}",inferRst->taskInfo->taskId, m_queueResult.size());
        if (m_bStopped)
            break;
    }
}

void AIRuntime::CallbackWorker()
{
    while (true) {
        {
            std::unique_lock<std::mutex> lock(m_lockResult);
            m_condResult.wait(lock, [&]() { return m_bStopped || !m_queueResult.empty(); });
        }

        ModelResultPtr modelResult;
        bool           found = m_queueResult.try_pop(modelResult);
        if (m_bStopped)
            break;

        if (!found) {
            continue;
        }
        LOG_INFO("[AIRuntime] recive model result, TaskId:{}, task queue size {}, "
                 "waiting queue {}, result queue size {}",
                 modelResult->taskInfo->taskId, m_queueTask.size(), m_rstResult.size(), m_queueResult.size());

        for (int i = 0; i < m_callbackList.size(); i++) {
            m_callbackList[i]->OnModelResult(modelResult);
        }
        if (m_bStopped)
            break;
    }
}

eAIErrorCode AIRuntime::CreateModle(stAIModelInfo& modelInfo)
{
    LOG_INFO("Start creating model: {}", modelInfo.ModelInfo());
    if (false == fs::exists(modelInfo.modelPath)) {
        LOG_INFOE("Create failed.Model file: {} not exists!", modelInfo.modelPath);
        return E_FILE_NOT_EXIST;
    }
    if (this->m_modelMap.count(modelInfo.modelId) > 0) {
        LOG_INFOW("The model: modelID = {} already exists!", modelInfo.modelId);
        return E_OK;
    }
    auto infer = std::shared_ptr<Algo::Infer>();
#if USE_TRT
    if (modelInfo.modelBackend == "tensorrt") {
        create_trt_model(modelInfo, infer);
    }
#endif //USE_TRT
#ifdef _WIN32
#if USE_ORT
    if (modelInfo.modelBackend == "onnxruntime") {
        create_ort_model(modelInfo, infer);
    }
#endif //USE_ORT
#elif __linux__
#endif

    if (infer == nullptr) {
        LOG_INFOE("Create failed. Can not create model from : {}!", modelInfo.modelPath);
        return E_CREATE_MODEL_FAILED;
    }

    auto ii = infer->infer_info();
    if (ii.contains("dims")) {
        auto dims = ii["dims"];
        for (int i = 0; i < dims.size(); i++) {
            int dim = dims[i];
            modelInfo.dims.push_back(dim);
        }
    }

    this->m_modelMap[modelInfo.modelId]    = infer;
    this->m_model_param[modelInfo.modelId] = modelInfo;

    LOG_INFO("Create model successful!");
    return E_OK;
};

eAIErrorCode AIRuntime::UpdateModle(stAIModelInfo& newModelInfo)
{
    LOG_INFO("Start updating model: {}", newModelInfo.ModelInfo());
    // auto infer = Yolo::create_infer(newModelInfo.modelPath, Yolo::Type::V5, 0,
    // 0.25, 0.45);
    auto infer = std::shared_ptr<Algo::Infer>();
#if USE_TRT
    if (newModelInfo.modelBackend == "tensorrt") {
        create_trt_model(newModelInfo, infer);
    }
#endif //USE_TRT
#ifdef _WIN32
#if USE_ORT
    if (newModelInfo.modelBackend == "onnxruntime") {
        create_ort_model(newModelInfo, infer);
    }
#endif
#elif __linux__
#endif
    if (infer != nullptr) {
        DestroyModle(newModelInfo.modelId);
    }
    return CreateModle(newModelInfo);
}
#if USE_TRT
void AIRuntime::create_trt_model(stAIModelInfo modelInfo, std::shared_ptr<Algo::Infer>& infer)
{
    switch (modelInfo.algoType) {
        case CLASSIFY:
            infer = Classification::create_infer(modelInfo.modelPath, modelInfo.inferParam.gpuId, modelInfo.inferParam.confidenceThreshold, modelInfo.inferParam.dim);
            break;
        case YOLOV5:
            infer = Yolo::create_infer(modelInfo.modelPath, Yolo::Type::V5, modelInfo.inferParam.gpuId, modelInfo.inferParam.confidenceThreshold, modelInfo.inferParam.nmsThreshold);
            break;
        case OCR_REC:
            infer = OCR::rec::create_infer(modelInfo.modelPath, modelInfo.inferParam.gpuId, modelInfo.inferParam.confidenceThreshold,modelInfo.modleLabelPath);
            break;
        case OCR_DET:
            infer = OCR::det::create_infer(modelInfo.modelPath, modelInfo.inferParam.gpuId, modelInfo.inferParam.kernelSize, modelInfo.inferParam.useDilat, modelInfo.inferParam.enableDetMat, modelInfo.inferParam.det_db_box_thresh, modelInfo.inferParam.det_db_unclip_ratio, modelInfo.inferParam.max_side_len);
            break;
        case OCR_CLS:
            infer = OCR::cls::create_infer(modelInfo.modelPath, modelInfo.inferParam.gpuId, modelInfo.inferParam.confidenceThreshold);
            break;
        case YOLO8:
            infer = yolo8::create_infer(modelInfo.modelPath, modelInfo.inferParam.gpuId, modelInfo.inferParam.confidenceThreshold, modelInfo.inferParam.nmsThreshold, modelInfo.inferParam.maxObjectNums);
            break;
        case MSAE:
            infer = msae::create_infer(modelInfo.modelPath, modelInfo.inferParam.gpuId, modelInfo.inferParam.confidenceThreshold, modelInfo.inferParam.nmsThreshold, modelInfo.inferParam.maxObjectNums);
            break;
        case ANOMALIB:
            infer = anomalib::create_infer(modelInfo.modelPath, modelInfo.inferParam.gpuId, modelInfo.inferParam.confidenceThreshold, modelInfo.inferParam.nmsThreshold, modelInfo.inferParam.maxObjectNums);
            break;
        case YOLO8_OBB:
            infer = yolo8_obb::create_infer(
                modelInfo.modelPath,          // engine file
                modelInfo.inferParam.gpuId,   // gpu id
                0.25f,                        // confidence threshold
                0.45f,                        // nms threshold
                yolo8_obb::NMSMethod::FastGPU,  // NMS method, fast GPU / CPU
                1024,                         // max objects
                false                         // preprocess use multi stream
            );
        case YOLOV8_SEG:
            infer = yolo8_seg::create_infer(modelInfo.modelPath, modelInfo.inferParam.gpuId, modelInfo.inferParam.confidenceThreshold, modelInfo.inferParam.nmsThreshold, modelInfo.inferParam.maxObjectNums, modelInfo.inferParam.segThreshold, modelInfo.inferParam.maxAreaCont, modelInfo.inferParam.dim);
            break;
        default:
            LOG_INFOE("Model Type error! code={}", modelInfo.algoType);
            break;
    }
    return;
}
#endif //USE_TRT
#ifdef _WIN32
#if USE_ORT
void AIRuntime::create_ort_model(stAIModelInfo modelInfo, std::shared_ptr<Algo::Infer>& infer)
{
    switch (modelInfo.algoType) {
        case CLASSIFY:
            infer = ort::classification::create_infer(modelInfo.modelPath, modelInfo.inferParam.confidenceThreshold, 0);
            break;
        case YOLOV5:
            infer = ort::yolo::create_infer(modelInfo.modelPath, ort::yolo::Type::V5, modelInfo.inferParam.confidenceThreshold, modelInfo.inferParam.nmsThreshold);
            break;
        case OCR_REC:
            infer = ort::ocr::rec::create_infer(modelInfo.modelPath, modelInfo.modleLabelPath, 0, modelInfo.inferParam.confidenceThreshold);
            break;
        case OCR_DET:
            infer = ort::ocr::det::create_infer(modelInfo.modelPath, 0, modelInfo.inferParam.kernelSize, modelInfo.inferParam.useDilat, modelInfo.inferParam.enableDetMat, modelInfo.inferParam.det_db_box_thresh, modelInfo.inferParam.det_db_unclip_ratio, modelInfo.inferParam.max_side_len);
            break;
        case YOLO8:
            infer = ort::yolo8::create_infer(modelInfo.modelPath, modelInfo.inferParam.confidenceThreshold, modelInfo.inferParam.nmsThreshold, modelInfo.inferParam.maxObjectNums);
            break;
        default:
            LOG_INFOE("Model Type error! code={}", modelInfo.algoType);
            break;
    }
}
#endif
#elif __linux__
#endif
