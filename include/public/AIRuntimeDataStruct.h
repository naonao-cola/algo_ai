#ifndef __AIRUNTIMEDATASTRUCT_H__
#define __AIRUNTIMEDATASTRUCT_H__
#include <memory>
#include <opencv2/opencv.hpp>
#include <future>
#include "./AIRuntimeUtils.h"


struct stAIConfigInfo {
    bool    usePinMemory{ true };
    size_t  workSpaceSize{ 2048 };
    size_t  GPUCachSize{ 1024 };
    size_t  CPUCachSize{ 1024 };
    int     preProcessThreadCnt{ 8 };
    int     preProcessThreadPriority{ 1 };
    int     inferThreadCnt{ 8 };
    int     inferThreadPriority{ 1 };
    stAIConfigInfo() = default;
    explicit stAIConfigInfo(json config) {
        usePinMemory = get_param<bool>(config, "usePinMemory", usePinMemory);
        workSpaceSize = get_param<size_t>(config, "workSpaceSize", workSpaceSize);
        GPUCachSize = get_param<size_t>(config, "GPUCachSize", GPUCachSize);
        CPUCachSize = get_param<size_t>(config, "CPUCachSize", CPUCachSize);
        preProcessThreadPriority = get_param<int>(config, "preProcessThreadPriority", preProcessThreadPriority);
        preProcessThreadCnt = get_param<int>(config, "preProcessThreadCnt", preProcessThreadCnt);
        inferThreadCnt = get_param<int>(config, "inferThreadCnt", inferThreadCnt);
        inferThreadPriority = get_param<int>(config, "inferThreadPriority", inferThreadPriority);
    }
};

enum eAIErrorCode
{
    E_OK = 0,
    E_OUT_OF_MEMORY,
    E_CREATE_MODEL_FAILED,
    E_FILE_NOT_EXIST,
    E_QUEUUE_FULL,
};


enum eAIAlgoType : int {
    CLASSIFY,
    YOLOV5,
    SEGMENT,
    OCR_REC,
    OCR_DET,
    OCR_CLS,
    YOLO8,
    MSAE,
    ANOMALIB,
    YOLO8_OBB,
    YOLOV8_SEG
};


struct stAIInferParam {
    int gpuId{ 0 };
    int maxBatchSize{ 1 };
    float confidenceThreshold{ 0.0 };
    float nmsThreshold{ 1.0 };
    int maxObjectNums{ 1024 };
    //OCR-Det
    bool enableDetMat{ false };
    bool useDilat{ false };
    int kernelSize{ 2 };
    float det_db_box_thresh{0.4};
    float det_db_unclip_ratio{1.5};
    int max_side_len{960};
    float single_char_shif_ratio{ 1.5f };

    // yolov8_seg
    float segThreshold{ 0.5 };
    bool  maxAreaCont{ true };
    int optBatchSize{1};
    int optChannalSize{-1};
    int optWidthSize{-1};
    int optHeightSize{-1};
    // 动态尺寸输入
    std::vector<std::vector<int>> dim;
    

    ////////////////
    stAIInferParam() = default;
    explicit stAIInferParam(const json& info) {
        gpuId = get_param<int>(info, "gpuId", gpuId);
        maxBatchSize = get_param<int>(info, "maxBatchSize", maxBatchSize);
        confidenceThreshold = get_param<float>(info, "confidenceThreshold", confidenceThreshold);
        nmsThreshold = get_param<float>(info, "nmsThreshold", nmsThreshold);
        maxObjectNums = get_param<int>(info, "maxObjectNums", maxObjectNums);
        enableDetMat = get_param<bool>(info, "enableDetMat", enableDetMat);
        useDilat = get_param<bool>(info, "useDilat", useDilat);
        kernelSize = get_param<bool>(info, "kernelSize", kernelSize);
        det_db_box_thresh = get_param<float>(info, "det_db_box_thresh", det_db_box_thresh);
        det_db_unclip_ratio = get_param<float>(info, "det_db_unclip_ratio", det_db_unclip_ratio);
        max_side_len = get_param<int>(info, "max_side_len", max_side_len);
        single_char_shif_ratio = get_param<float>(info, "single_char_shif_ratio", single_char_shif_ratio);
        segThreshold        = get_param<float>(info, "segThreshold", segThreshold);
        maxAreaCont         = get_param<bool>(info, "maxAreaCont", maxAreaCont);
        optBatchSize = get_param<int>(info, "optBatchSize", optBatchSize);
        optChannalSize = get_param<int>(info, "optChannalSize", optChannalSize);
        optWidthSize = get_param<int>(info, "optWidthSize", optWidthSize);
        optHeightSize = get_param<int>(info, "optHeightSize", optHeightSize);

    }
    void construct_dim()
    {
        dim = { { optBatchSize, optChannalSize, optWidthSize, optHeightSize } };
    }
};

struct stAIModelInfo {
    using  mPtr =std::shared_ptr<stAIModelInfo> ;
    int modelVersion{ 1 };
    int modelId{ 0 };
    eAIAlgoType algoType{ eAIAlgoType::CLASSIFY };
    std::string modelName{ "" };
    std::string modelPath{ "" };  // model file path
    stAIInferParam inferParam;
    std::string    modelBackend{"tensorrt"}; //onnxruntime,tensorrt
    std::string    modleLabelPath{ "" };     //ģ�͵�label�ļ�������ocr���

    // 输出参数，模型输入shapes
    std::vector<int> dims;
    stAIModelInfo() = default;
    explicit stAIModelInfo(const json& info) {
        modelVersion = get_param<int>(info, "modelVersion", modelVersion);
        modelId = get_param<int>(info, "modelId", modelId);
        modelName = get_param<std::string>(info, "modelName", modelName);
        modelPath = get_param<std::string>(info, "modelPath", modelPath);
        modelBackend = get_param<std::string>(info, "modelBackend", modelBackend);
        inferParam = stAIInferParam(info["InferParam"]);
    }

    std::string ModelInfo() const {
        char buff[400];
        std::string rst = " \n ============================================================================";
#ifdef _WIN32
        sprintf_s(buff, "%s\n model version:\t\t %d \n model id:\t\t %d \n algoType:\t\t %d \n modelPath:\t\t%s %s", 
                rst.c_str(), modelVersion, modelId, (int)algoType, modelPath.c_str(), rst.c_str());
#elif __linux__
        snprintf(buff, sizeof(buff), "%s\n model version:\t\t %d \n model id:\t\t %d \n algoType:\t\t %d \n modelPath:\t\t%s %s", 
            rst.c_str(), modelVersion, modelId, (int)algoType, modelPath.c_str(), rst.c_str());
#endif
        return std::string(buff);
    }
   
};


struct stImageBuf
{
    int width;
    int height;
    int channel;
    unsigned char* data;
};

struct stTaskInfo
{
    int modelId;
    int taskId;
    int orgImageId;
    TimeCost tt;
    long long preCostTime;
    long long inferCostTime;
    long long hostCostTime;
    long long totalCostTime;
    std::shared_ptr<void> inspParam;
    std::vector<cv::Mat> imageData;
    void* promiseResult;
    std::string Info() const {
        char buff[200];
#ifdef _WIN32
        sprintf_s(buff, "\n model id:\t\t%d\n image size:\t\t%d", modelId, imageData.size());
#elif __linux__
        snprintf(buff, sizeof(buff), "\n model id:\t\t%d\n image size:\t\t%d", modelId, imageData.size());
#endif
        // sprintf_s(buff, "\n model id:\t\t%d\n image size:\t\t%d", modelId, imageData.size());
        return std::string(buff);
    }
};

struct stPoint
{
    float x;
    float y;
    stPoint(float x_, float y_) : x(x_), y(y_) {}
};

struct stResultItem
{
    int code;
    int shape;
    float confidence;
    std::vector<stPoint> points;
    float                                       angle;
    std::vector<std::pair<std::string, double>> rec_conf_res;
    std::vector<float>                          res_ratio_res;
    std::vector<std::vector<cv::Point>>         char_single_pos;
    std::vector<int>                            ocr_char_index;
    std::string          ocr_str = "";
    cv::Mat              ocr_det;
    cv::Mat              msae_img;
    cv::Mat              heatMap;
    std::vector<std::vector<cv::Point>> mask;
    std::string Info() {
        std::string rst = "";
        char buff[200];
#ifdef _WIN32
        sprintf_s(buff, "\n code:\t\t %d \n shape:\t\t %d\n confidence:\t\t %g", code, shape, confidence);
#elif __linux__
        snprintf(buff, sizeof(buff), "\n code:\t\t %d \n shape:\t\t %d\n confidence:\t\t %g", code, shape, confidence);
#endif
        // sprintf_s(buff, "\n code:\t\t %d \n shape:\t\t %d\n confidence:\t\t %g", code, shape, confidence);
        if (points.size() == 2) {
#ifdef _WIN32
        sprintf_s(buff, "%s\n points:\t\t[%g  %g  %g  %g]", buff, points[0].x, points[0].y, points[1].x, points[1].y);
#elif __linux__
        snprintf(buff, sizeof(buff), "%s\n points:\t\t[%g  %g  %g  %g]", buff, points[0].x, points[0].y, points[1].x, points[1].y);
            // sprintf_s(buff, "%s\n points:\t\t[%g  %g  %g  %g]", buff, points[0].x, points[0].y, points[1].x, points[1].y);
#endif
        }
        else {
            rst += "[]";
        }
        rst = std::string(buff);
        return rst;
    }
};

struct stModelResult
{
    std::shared_ptr<stTaskInfo> taskInfo;
    std::vector<std::vector<stResultItem>> itemList;
};

struct stGPUInfo {
    int    gpuId;
    size_t totalMemorySize;
    size_t usedMemorySize;
    size_t avaliableMemorySize;
    float  gpuUtilRate;
};

using ModelResultPtr = std::shared_ptr<stModelResult>;
using TaskInfoPtr = std::shared_ptr<stTaskInfo>;

#endif // __DATA_STRUCT_H__