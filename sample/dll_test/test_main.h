#include "../../install/AIRuntimeDataStruct.h"
#include "../../install/AIRuntimeInterface.h"
#include "../../install/AIRuntimeUtils.h"
#include "../../3rdparty/argparse/argparse.hpp"

// #include <windows.h>
#ifdef _WIN32
    #include <windows.h>
#elif __linux__
#endif

class PR:IModelResultListener
{
public: 
       void  OnModelResult(ModelResultPtr spResult)
       {
           for(int i = 0; i < spResult->itemList.size();i++) {
               for (int j = 0; j < spResult->itemList[i].size();j++) {

                    std::string ret = (spResult->itemList[i])[j].Info();
                   std::cout<<ret<<std::endl;
               }
           }
       }

};    

cv::Mat GetRotateCropImage(const cv::Mat& srcimage, std::vector<std::vector<int>> box)
{
       std::vector<int> x_vec{ box[0][0], box[1][0], box[2][0], box[3][0] };
       std::vector<int> y_vec{ box[0][1], box[1][1], box[2][1], box[3][1] };
       int              x_min = *std::min_element(x_vec.begin(), x_vec.end());
       int              x_max = *std::max_element(x_vec.begin(), x_vec.end());

       int y_min = *std::min_element(y_vec.begin(), y_vec.end());
       int y_max = *std::max_element(y_vec.begin(), y_vec.end());
       if (x_max - x_min < 3 || y_max - y_min < 3)
           return cv::Mat();

       cv::Mat image;
       srcimage.copyTo(image);
       std::vector<std::vector<int>> points = box;

       int x_collect[4] = { box[0][0], box[1][0], box[2][0], box[3][0] };
       int y_collect[4] = { box[0][1], box[1][1], box[2][1], box[3][1] };
       int left         = int(*std::min_element(x_collect, x_collect + 4));
       int right        = int(*std::max_element(x_collect, x_collect + 4));
       int top          = int(*std::min_element(y_collect, y_collect + 4));
       int bottom       = int(*std::max_element(y_collect, y_collect + 4));

       cv::Mat img_crop;
       image(cv::Rect(left, top, right - left, bottom - top)).copyTo(img_crop);

       for (int i = 0; i < points.size(); i++) {
           points[i][0] -= left;
           points[i][1] -= top;
       }

       int img_crop_width  = int(sqrt(pow(points[0][0] - points[1][0], 2) + pow(points[0][1] - points[1][1], 2)));
       int img_crop_height = int(sqrt(pow(points[0][0] - points[3][0], 2) + pow(points[0][1] - points[3][1], 2)));

       cv::Point2f pts_std[4];
       pts_std[0] = cv::Point2f(0., 0.);
       pts_std[1] = cv::Point2f(img_crop_width, 0.);
       pts_std[2] = cv::Point2f(img_crop_width, img_crop_height);
       pts_std[3] = cv::Point2f(0.f, img_crop_height);

       cv::Point2f pointsf[4];
       pointsf[0] = cv::Point2f(points[0][0], points[0][1]);
       pointsf[1] = cv::Point2f(points[1][0], points[1][1]);
       pointsf[2] = cv::Point2f(points[2][0], points[2][1]);
       pointsf[3] = cv::Point2f(points[3][0], points[3][1]);

       cv::Mat M = cv::getPerspectiveTransform(pointsf, pts_std);

       cv::Mat dst_img;
       cv::warpPerspective(img_crop, dst_img, M, cv::Size(img_crop_width, img_crop_height), cv::BORDER_REPLICATE);

       if (float(dst_img.rows) >= float(dst_img.cols) * 1.5) {
           cv::Mat srcCopy = cv::Mat(dst_img.rows, dst_img.cols, dst_img.depth());
           cv::transpose(dst_img, srcCopy);
           cv::flip(srcCopy, srcCopy, 0);
           return srcCopy;
       }
       else {
           return dst_img;
       }
}

inline void test_sync() {
    //////////////////////////////////////////////////////////
    stAIConfigInfo      ai_cfg;
    ai_cfg.preProcessThreadPriority = 0;
    ai_cfg.inferThreadCnt = 1;
    ai_cfg.inferThreadPriority = 0;

    AIRuntimeInterface* ai_obj = GetAIRuntime();
    ai_obj->InitRuntime(ai_cfg);

    stAIModelInfo model_cfg;

    model_cfg.modelId = 0;
    model_cfg.modelName = "yolov8";
    model_cfg.modelPath = R"(E:\project\SBG\tv_algo_sbg\build\windows\x64\release\Model\sbg0402.trtmodel)";
    model_cfg.modelVersion = 1;
    model_cfg.modelBackend = "tensorrt";
    model_cfg.inferParam.confidenceThreshold = 0.45;
    model_cfg.inferParam.maxBatchSize = 1;
    model_cfg.inferParam.nmsThreshold = 0.6;
    model_cfg.inferParam.maxObjectNums = 200000;
    model_cfg.algoType = YOLO8;


    ai_obj->CreateModle(model_cfg);
////////////////////////////////////////////////////////////////

    model_cfg.modelId = 1;
    model_cfg.modelName = "rec";
    model_cfg.modelPath = "E:\\project\\SBG\\tv_algo_sbg\\build\\windows\\x64\\release\\Model\\ortmodels\\rec.trtmodel";
    model_cfg.modelVersion = 1;
    model_cfg.modelBackend = "tensorrt";
    model_cfg.inferParam.confidenceThreshold = 0.25;
    model_cfg.modleLabelPath = "E:\\project\\EFEM\\tv_algorithm\\build\\windows\\x64\\release\\ort_models\\ppocr_keys_v1.txt";
    model_cfg.inferParam.maxBatchSize = 1;
    model_cfg.algoType = OCR_REC;


    ai_obj->CreateModle(model_cfg);
////////////////////////////////////////////////////////////////
    model_cfg.modelId = 2;
    model_cfg.modelName = "det";
    model_cfg.modelPath = "E:\\project\\SBG\\tv_algo_sbg\\build\\windows\\x64\\release\\Model\\ortmodels\\det.trtmodel";
    model_cfg.modelVersion = 1;
    model_cfg.modelBackend = "tensorrt";
    model_cfg.inferParam.confidenceThreshold = 0.25;
    model_cfg.inferParam.maxBatchSize = 1;
    model_cfg.inferParam.enableDetMat = false;
    //model_cfg.inferParam.nmsThreshold        = 0.6;
    //model_cfg.inferParam.maxObjectNums       = 200000;
    model_cfg.algoType = OCR_DET;

    ai_obj->CreateModle(model_cfg);

///////////////////////////////////////////////////////////////
    model_cfg.modelId = 3;
    model_cfg.modelName = "stfpm";
    model_cfg.modelPath = "E:\\model\\fastflow\\stfpm.trtmodel";
    model_cfg.modelVersion = 1;
    model_cfg.modelBackend = "tensorrt";
    model_cfg.inferParam.confidenceThreshold = 0.25;
    model_cfg.inferParam.maxBatchSize = 1;
    //model_cfg.inferParam.nmsThreshold        = 0.6;
    //model_cfg.inferParam.maxObjectNums       = 200000;
    model_cfg.algoType = ANOMALIB;

    ai_obj->CreateModle(model_cfg);
///////////////////////////////////////////////////////////////
    model_cfg.modelId = 4;
    model_cfg.modelName = "yolov5";
    model_cfg.modelPath = R"(D:\R\trt\tensorRT_Pro-main\example-simple_yolo\workspace\yolov5s.FP32.trtmodel)";
    model_cfg.modelVersion = 1;
    model_cfg.modelBackend = "tensorrt";//tensorrt   onnxruntime
    model_cfg.inferParam.confidenceThreshold = 0.25;
    model_cfg.inferParam.maxBatchSize = 1;
    model_cfg.inferParam.nmsThreshold = 0.45;
    model_cfg.inferParam.maxObjectNums = 200000;
    model_cfg.algoType = YOLOV5;

    ai_obj->CreateModle(model_cfg);


    //model_cfg.modelId = 6;
    //model_cfg.modelName = "yolov5";
    //model_cfg.modelPath = R"(D:\R\trt\tensorRT_Pro-main\example-simple_yolo\workspace\yolov5s.FP32.trtmodel)";
    //model_cfg.modelVersion = 1;
    //model_cfg.modelBackend = "tensorrt";//tensorrt   onnxruntime
    //model_cfg.inferParam.confidenceThreshold = 0.25;
    //model_cfg.inferParam.maxBatchSize = 1;
    //model_cfg.inferParam.nmsThreshold = 0.45;
    //model_cfg.inferParam.maxObjectNums = 200000;
    //model_cfg.algoType = YOLOV5;

    //ai_obj->CreateModle(model_cfg);

    //model_cfg.modelId = 7;
    //model_cfg.modelName = "yolov5";
    //model_cfg.modelPath = R"(D:\R\trt\tensorRT_Pro-main\example-simple_yolo\workspace\yolov5s.FP32.trtmodel)";
    //model_cfg.modelVersion = 1;
    //model_cfg.modelBackend = "tensorrt";//tensorrt   onnxruntime
    //model_cfg.inferParam.confidenceThreshold = 0.25;
    //model_cfg.inferParam.maxBatchSize = 1;
    //model_cfg.inferParam.nmsThreshold = 0.45;
    //model_cfg.inferParam.maxObjectNums = 200000;
    //model_cfg.algoType = YOLOV5;

    //ai_obj->CreateModle(model_cfg);

    //model_cfg.modelId = 8;
    //model_cfg.modelName = "yolov5";
    //model_cfg.modelPath = R"(D:\R\trt\tensorRT_Pro-main\example-simple_yolo\workspace\yolov5s.FP32.trtmodel)";
    //model_cfg.modelVersion = 1;
    //model_cfg.modelBackend = "tensorrt";//tensorrt   onnxruntime
    //model_cfg.inferParam.confidenceThreshold = 0.25;
    //model_cfg.inferParam.maxBatchSize = 1;
    //model_cfg.inferParam.nmsThreshold = 0.45;
    //model_cfg.inferParam.maxObjectNums = 200000;
    //model_cfg.algoType = YOLOV5;

    //ai_obj->CreateModle(model_cfg);

    //model_cfg.modelId = 9;
    //model_cfg.modelName = "yolov5";
    //model_cfg.modelPath = R"(D:\R\trt\tensorRT_Pro-main\example-simple_yolo\workspace\yolov5s.FP32.trtmodel)";
    //model_cfg.modelVersion = 1;
    //model_cfg.modelBackend = "tensorrt";//tensorrt   onnxruntime
    //model_cfg.inferParam.confidenceThreshold = 0.25;
    //model_cfg.inferParam.maxBatchSize = 1;
    //model_cfg.inferParam.nmsThreshold = 0.45;
    //model_cfg.inferParam.maxObjectNums = 200000;
    //model_cfg.algoType = YOLOV5;

    //ai_obj->CreateModle(model_cfg);
//////////////////////////////////////////////////////////////

    model_cfg.modelId = 5;
    model_cfg.modelName = "mase";
    model_cfg.modelPath = R"(E:\project\ai_inference\Model\msae_hgz_a.trtmodel)";
    model_cfg.modelVersion = 1;
    model_cfg.modelBackend = "tensorrt";//tensorrt   onnxruntime
    model_cfg.inferParam.confidenceThreshold = 0.25;
    model_cfg.inferParam.maxBatchSize = 1;
    model_cfg.inferParam.nmsThreshold = 0.45;
    model_cfg.inferParam.maxObjectNums = 200000;

    model_cfg.algoType = MSAE;

    ai_obj->CreateModle(model_cfg);

    //ai_obj->UpdateModle(model_cfg);
    //model_cfg.modelId = 10;
    //model_cfg.modelName = "mase";
    //model_cfg.modelPath = R"(E:\project\ai_inference\Model\msae_hgz_a.trtmodel)";
    //model_cfg.modelVersion = 1;
    //model_cfg.modelBackend = "tensorrt";//tensorrt   onnxruntime
    //model_cfg.inferParam.confidenceThreshold = 0.25;
    //model_cfg.inferParam.maxBatchSize = 1;
    //model_cfg.inferParam.nmsThreshold = 0.45;
    //model_cfg.inferParam.maxObjectNums = 200000;
    //model_cfg.algoType = MSAE;

    //ai_obj->CreateModle(model_cfg);

    //model_cfg.modelId = 11;
    //model_cfg.modelName = "mase";
    //model_cfg.modelPath = R"(E:\project\ai_inference\Model\msae_hgz_a.trtmodel)";
    //model_cfg.modelVersion = 1;
    //model_cfg.modelBackend = "tensorrt";//tensorrt   onnxruntime
    //model_cfg.inferParam.confidenceThreshold = 0.25;
    //model_cfg.inferParam.maxBatchSize = 1;
    //model_cfg.inferParam.nmsThreshold = 0.45;
    //model_cfg.inferParam.maxObjectNums = 200000;
    //model_cfg.algoType = MSAE;

    //ai_obj->CreateModle(model_cfg);

    //model_cfg.modelId = 12;
    //model_cfg.modelName = "mase";
    //model_cfg.modelPath = R"(E:\project\ai_inference\Model\msae_hgz_a.trtmodel)";
    //model_cfg.modelVersion = 1;
    //model_cfg.modelBackend = "tensorrt";//tensorrt   onnxruntime
    //model_cfg.inferParam.confidenceThreshold = 0.25;
    //model_cfg.inferParam.maxBatchSize = 1;
    //model_cfg.inferParam.nmsThreshold = 0.45;
    //model_cfg.inferParam.maxObjectNums = 200000;
    //model_cfg.algoType = MSAE;

    //ai_obj->CreateModle(model_cfg);
////////////////////////////////////////////////
    cv::Mat     img = cv::imread(R"(./test_algo/IMG_0027.JPG)");
    TaskInfoPtr task = std::make_shared<stTaskInfo>();
    task->imageData = { img };
    task->modelId = 2;
    task->taskId = 0;
    ModelResultPtr spResul = ai_obj->RunInferTask(task);
    std::vector<std::vector<int>> ocr_det;
    cv::Mat cropImg;
    for (int i = 0; i < spResul->itemList.size(); i++) {
        auto ret = spResul->itemList[i];
        cropImg = ret[0].ocr_det;
    }

    task->imageData = { cropImg };
    task->modelId = 1;
    spResul = ai_obj->RunInferTask(task);
    //ai_obj->DestoryRuntime();
    return;
}
    
inline void test_ort_yolo8(){


	stAIConfigInfo      ai_cfg;
    ai_cfg.preProcessThreadPriority = 0;
    ai_cfg.inferThreadCnt           = 1;
    ai_cfg.inferThreadPriority      = 0;

	AIRuntimeInterface* ai_obj = GetAIRuntime();
    ai_obj->InitRuntime(ai_cfg);

    stAIModelInfo model_cfg;

    model_cfg.modelId = 0;
    model_cfg.modelName = "tolov8_collet.onnx";
    model_cfg.modelPath = R"(E:\demo\rep\AIFramework\models\ort_models\tolov8_collet.onnx)";
    model_cfg.modelVersion = 1;
    model_cfg.modelBackend = "onnxruntime";
    model_cfg.inferParam.confidenceThreshold = 0.45;
    model_cfg.inferParam.maxBatchSize        = 1;
    model_cfg.inferParam.nmsThreshold        = 0.6;
    model_cfg.inferParam.maxObjectNums       = 200000;
    model_cfg.algoType                       = YOLO8;


    ai_obj->CreateModle(model_cfg);

    PR print_ret;
    ai_obj->RegisterResultListener(0, (IModelResultListener*)&print_ret);

    cv::Mat     img = cv::imread(R"(E:\demo\rep\AIFramework\data\test_img\20231115-141947.jpg)");
    TaskInfoPtr task = std::make_shared<stTaskInfo>();
    task->imageData = { img };
    task->modelId   = 0;
    task->taskId     = 0;
    ai_obj->CommitInferTask(task);

    // Sleep(50000);
}
// const char* model1_labels = {"xxx1", "xxx2"};
const std::vector<std::string> labels{
"0P510",
"0P705",
"0B510",
"0I302",
"0I304",
"0P306",
"0P501"
    };
cv::Mat draw_rst(cv::Mat& image, const stResultItem& box)
{
    cv::Scalar color(255, 255, 0);
    cv::rectangle(image, cv::Point(box.points[0].x, box.points[0].y), cv::Point(box.points[1].x, box.points[1].y), color, 3);
    auto& name       = labels[box.code];
    auto  caption    = cv::format("%s %.2f", name.c_str(), box.confidence);
    int   text_width = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
    cv::rectangle(image, cv::Point(box.points[0].x - 3, box.points[0].y - 33), cv::Point(box.points[0].x + text_width, box.points[0].y), color, -1);
    cv::putText(image, caption, cv::Point(box.points[0].x, box.points[0].y - 5), 0, 1, cv::Scalar::all(0), 2, 16);
    return image;
}

inline void test_classify()
{
    stAIConfigInfo ai_cfg;
    ai_cfg.preProcessThreadPriority = 0;
    ai_cfg.inferThreadCnt           = 1;
    ai_cfg.inferThreadPriority      = 0;

    AIRuntimeInterface* ai_obj = GetAIRuntime();
    ai_obj->InitRuntime(ai_cfg);

    stAIModelInfo model_cfg;

    model_cfg.modelId      = 0;
    model_cfg.modelName    = "classify";
    model_cfg.modelPath    = "D:/work/0_HF/AI/model/1127/lr_dyn.engine";
    model_cfg.modelVersion = 1;
    model_cfg.modelBackend = "tensorrt";
    model_cfg.algoType                = CLASSIFY;
    model_cfg.inferParam.optBatchSize = 12;
    model_cfg.inferParam.construct_dim();

    ai_obj->CreateModle(model_cfg);

    PR print_ret;
    ai_obj->RegisterResultListener(0, (IModelResultListener*)&print_ret);

    
    //cv::Mat img12 = cv::imread("D:/work/0_HF/svm_trainSample/sbzg/sbzg_1851408151/NG/0_A_639_1744.jpg");

    cv::Mat              img0 = cv::imread("D:/work/0_HF/AI/images/test_cls/Image_20241025134455122_1_4.jpg");
    cv::Mat              img1 = cv::imread("D:/work/0_HF/AI/images/test_cls/Image_20241025134455122_3_4.jpg");
    cv::Mat              img2 = cv::imread("D:/work/0_HF/AI/images/test_cls/Image_20241025134455122_4_5.jpg");
    cv::Mat              img3 = cv::imread("D:/work/0_HF/AI/images/test_cls/Image_20241025134455122_5_4.jpg");
    cv::Mat              img4 = cv::imread("D:/work/0_HF/AI/images/test_cls/Image_20241025134455122_5_5.jpg");
    cv::Mat              img5 = cv::imread("D:/work/0_HF/AI/images/test_cls/Image_20241025134455122_7_4.jpg");
    cv::Mat              img6 = cv::imread("D:/work/0_HF/AI/images/test_cls/Image_20241025134455122_11_5.jpg");
    cv::Mat              img7 = cv::imread("D:/work/0_HF/AI/images/test_cls/Image_20241025134455122_12_5.jpg");
    cv::Mat              img8 = cv::imread("D:/work/0_HF/AI/images/test_cls/Image_20241025134455122_13_5.jpg");
    cv::Mat              img9 = cv::imread("D:/work/0_HF/AI/images/test_cls/Image_20241025134455122_15_5.jpg");
    cv::Mat              img10 = cv::imread("D:/work/0_HF/AI/images/test_cls/Image_20241025134455122_19_5.jpg");
    cv::Mat              img11 = cv::imread("D:/work/0_HF/AI/images/test_cls/Image_20241025134455122_20_4.jpg");
    cv::Mat              img12 = cv::imread("D:/work/0_HF/AI/test/6fe7e5c974754aa0_0_20241127093336996_0_0_1.jpg");

    std::vector<cv::Mat> img_list;
    img_list.push_back(img12);
    img_list.push_back(img12);
    img_list.push_back(img12);
    img_list.push_back(img12);
    img_list.push_back(img12);
    img_list.push_back(img12);
    img_list.push_back(img12);
    img_list.push_back(img12);
    img_list.push_back(img12);
    img_list.push_back(img12);
    img_list.push_back(img12);
    img_list.push_back(img12);


    for (int loop = 0; loop < 20; loop++) {
        auto        start = std::chrono::high_resolution_clock::now();
        TaskInfoPtr task = std::make_shared<stTaskInfo>();
        task->imageData  = { img_list };
        task->modelId    = 0;
        task->taskId     = 0;

        ModelResultPtr clsResultPtr = GetAIRuntime()->RunInferTask(task);

        // 输出结果
        if (clsResultPtr->itemList.size() == 0) {
            std::cout << "itemList.size() == 0" << std::endl;
        }

        json jsonArray;
        for (int i = 0; i < clsResultPtr->itemList.size();i++){
            auto clsRstList = clsResultPtr->itemList[i];
            if (clsRstList.size() == 0) {
                std::cout << "clsRstList.size() == 0" << std::endl;
            }
            else {
                for (int j = 0; j < clsRstList.size(); j++) {
                    //std::cout << "clsRstList[0].code:" << clsRstList[0].code << std::endl;
                    jsonArray.push_back(clsRstList[j].code);
                }
            }
        }
        std::cout << "jsonArray:" << jsonArray.dump() << std::endl;

        auto end      = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;
        std::cout << "推理运行时间: " << duration.count() << " 毫秒" << std::endl;
    }
}

//inline void build_trt(int argc, char** argv) 
inline void build_trt()
{
    /*argparse::ArgumentParser program("model_conversion_cmd");
    program.add_argument("--onnx_model")
        .help("onnx模型名(不包含后缀.onnx)")
        .default_value(std::string(""));
    program.add_argument("--modelfile_path")
        .help("onnx模型路径(trt模型生成在该目录下)")
        .default_value(std::string("./"));
    try {
        program.parse_args(argc, argv);
    }
    catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        std::exit(1);
    }*/
    //std::string onnx_model = program.get<std::string>("--onnx_model");
    //std::string trt_model  = program.get<std::string>("--modelfile_path");
    std::string onnx_model = "seg_6";
    std::string trt_model  = "D:/work/0_HF/AI/model";

    if (onnx_model != "") {
        // std::string tapp_path = "./models/";
        std::string onnx_path = trt_model + "/" + onnx_model + ".onnx";
        std::string trt_path  = trt_model + "/" + onnx_model + ".trtmodel";
        // encode_onnx_model_to_binn(onnx_path.c_str(), trt_path.c_str());
        build_model(0, 6, onnx_path.c_str(), trt_path.c_str());
        std::cout << "conversion model succeesed!!!" << std::endl;
    }
    else {
        std::cout << "conversion model Fail。。。 " << std::endl;
    }
}

cv::Mat draw_rst_withmask(cv::Mat& image, const stResultItem& box)
{
    cv::Scalar color(255, 255, 0);
    cv::rectangle(image, cv::Point(box.points[0].x, box.points[0].y), cv::Point(box.points[1].x, box.points[1].y), color, 3);
    auto& name       = labels[box.code];
    auto  caption    = cv::format("%s %.2f", name.c_str(), box.confidence);
    int   text_width = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
    cv::rectangle(image, cv::Point(box.points[0].x - 3, box.points[0].y - 33), cv::Point(box.points[0].x + text_width, box.points[0].y), color, -1);
    cv::putText(image, caption, cv::Point(box.points[0].x, box.points[0].y - 5), 0, 1, cv::Scalar::all(0), 2, 16);
    cv::fillPoly(image, box.mask, cv::Scalar(0, 0, 255), 8, 0);
    return image;
}

inline void test_trt_yolo8seg()
{
    stAIConfigInfo ai_cfg;
    ai_cfg.preProcessThreadPriority = 0;
    ai_cfg.inferThreadCnt           = 1;
    ai_cfg.inferThreadPriority      = 0;

    AIRuntimeInterface* ai_obj = GetAIRuntime();
    ai_obj->InitRuntime(ai_cfg);

    stAIModelInfo model_cfg;

    model_cfg.modelId                        = 0;
    model_cfg.modelName                      = "best.trt.engine";
    model_cfg.modelPath                      = R"(D:\work\0_HF\AI\model\1120\seg_dyn.engine)";
    model_cfg.modelVersion                   = 1;
    model_cfg.modelBackend                   = "tensorrt";
    model_cfg.inferParam.confidenceThreshold = 0.3;
    model_cfg.inferParam.maxBatchSize        = 1;
    model_cfg.inferParam.optBatchSize        = 6;
    model_cfg.inferParam.nmsThreshold        = 0.5;
    model_cfg.inferParam.maxObjectNums       = 200000;
    model_cfg.algoType                       = YOLOV8_SEG;
    model_cfg.inferParam.segThreshold        = 0.5;
    model_cfg.inferParam.construct_dim();

    ai_obj->CreateModle(model_cfg);

    PR print_ret;
    ai_obj->RegisterResultListener(0, (IModelResultListener*)&print_ret);
    std::vector<std::string > path_list;
    path_list.push_back("D:/work/0_HF/AI/images/test_seg/test_rst/Image_20241031100910652_0.jpg");
    path_list.push_back("D:/work/0_HF/AI/images/test_seg/test_rst/Image_20241031100910652_1.jpg");
    path_list.push_back("D:/work/0_HF/AI/images/test_seg/test_rst/Image_20241031100910652_2.jpg");
    path_list.push_back("D:/work/0_HF/AI/images/test_seg/test_rst/Image_20241031100910652_3.jpg");
    path_list.push_back("D:/work/0_HF/AI/images/test_seg/test_rst/Image_20241031100910652_4.jpg");
    path_list.push_back("D:/work/0_HF/AI/images/test_seg/test_rst/Image_20241031100910652_5.jpg");


    cv::Mat     img0     = cv::imread(R"(D:\work\0_HF\AI\images\test_seg\test\Image_20241031100910652_0.jpg)");
    cv::Mat     img1     = cv::imread(R"(D:\work\0_HF\AI\images\test_seg\test\Image_20241031100910652_1.jpg)");
    cv::Mat     img2     = cv::imread(R"(D:\work\0_HF\AI\images\test_seg\test\Image_20241031100910652_2.jpg)");
    cv::Mat     img3     = cv::imread(R"(D:\work\0_HF\AI\images\test_seg\test\Image_20241031100910652_3.jpg)");
    cv::Mat     img4     = cv::imread(R"(D:\work\0_HF\AI\images\test_seg\test\Image_20241031100910652_4.jpg)");
    cv::Mat     img5     = cv::imread(R"(D:\work\0_HF\AI\images\test_seg\test\Image_20241031100910652_5.jpg)");
    cv::Mat     img6     = cv::imread(R"(D:\work\0_HF\AI\images\1108\sample_crop\black\1\Image_20241108150253101_0.jpg)");

    std::vector<cv::Mat> img_list;
    img_list.push_back(img6);
    img_list.push_back(img6);
    img_list.push_back(img6);
    img_list.push_back(img6);
    img_list.push_back(img6);
    img_list.push_back(img6);

    
    

    for (int loop = 0; loop < 15; loop++) {
        auto start          = std::chrono::high_resolution_clock::now();

        TaskInfoPtr        task     = std::make_shared<stTaskInfo>();
        task->imageData     = { img_list };
        task->modelId       = 0;
        task->taskId        = 0;
        task->promiseResult = new std::promise<ModelResultPtr>();
        ai_obj->CommitInferTask(task);

        std::promise<ModelResultPtr>* promiseResult = static_cast<std::promise<ModelResultPtr>*>(task->promiseResult);
        std::future<ModelResultPtr>   futureRst     = promiseResult->get_future();

        ModelResultPtr rst         = futureRst.get();
        cv::Mat        binImg      = cv::Mat::zeros(img0.rows, img0.cols, CV_8UC1);
        int            colletCount = 0;
        for (int i = 0; i < rst->itemList.size(); i++) {
            cv::Mat src = img_list[i].clone();
            for (auto& box : rst->itemList[i]) {
                std::cout << box.confidence << box.code << box.shape << std::endl;
                std::cout << box.points[0].x << " " << box.points[0].y << " " << box.points[1].x << " " << box.points[1].y << std::endl;
                // draw_rst(img, box);
                draw_rst_withmask(src, box);
                int a = 1;
            }
            //cv::imwrite(path_list[i], src);
            colletCount++;
            std::cout << "图片个数:" << colletCount << std::endl;
        }
        auto                                      end      = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;
        std::cout << "推理运行时间: " << duration.count() << " 毫秒" << std::endl;
    }
}

inline void test_trt_yolo8(){

    stAIConfigInfo ai_cfg;
    ai_cfg.preProcessThreadPriority = 0;
    ai_cfg.inferThreadCnt           = 1;
    ai_cfg.inferThreadPriority      = 0;

    AIRuntimeInterface* ai_obj = GetAIRuntime();
    ai_obj->InitRuntime(ai_cfg);

    stAIModelInfo model_cfg;

    model_cfg.modelId                        = 0;
    model_cfg.modelName                      = "best.trt.engine";
    model_cfg.modelPath                      = R"(E:\demo\3rdparty\TensorRT-8.4.1.5\bin\best.trt.engine)";
    model_cfg.modelVersion                   = 1;
    model_cfg.modelBackend                   = "tensorrt";
    model_cfg.inferParam.confidenceThreshold = 0.52;
    model_cfg.inferParam.maxBatchSize        = 1;
    model_cfg.inferParam.nmsThreshold        = 0.6;
    model_cfg.inferParam.maxObjectNums       = 200000;
    model_cfg.algoType                       = YOLO8;

    ai_obj->CreateModle(model_cfg);

    PR print_ret;
    ai_obj->RegisterResultListener(0, (IModelResultListener*)&print_ret);

    cv::Mat     img  = cv::imread(R"(C:\Users\Administrator\Desktop\EvopactHVX-210\Front\temp\705.jpg)");
    TaskInfoPtr task = std::make_shared<stTaskInfo>();
    task->imageData  = { img };
    task->modelId    = 0;
    task->taskId     = 0;
    task->promiseResult = new std::promise<ModelResultPtr>();
    ai_obj->CommitInferTask(task);

    
    std::promise<ModelResultPtr>* promiseResult = static_cast<std::promise<ModelResultPtr>*>(task->promiseResult);
    std::future<ModelResultPtr>   futureRst  = promiseResult->get_future();
   
    ModelResultPtr rst         = futureRst.get();
    cv::Mat        binImg      = cv::Mat::zeros(img.rows, img.cols, CV_8UC1);
    int            colletCount = 0;
    for (int i = 0; i < rst->itemList.size(); i++) {
           for (auto& box : rst->itemList[i]) {
               std::cout <<box.confidence <<box.code <<box.shape << std::endl;
               std::cout << box.points[0].x << " " << box.points[0].y << " " << box.points[1].x << " "<<box.points[1].y << std::endl;
               draw_rst(img, box);

           }
    }

}

inline void test_trt_yolov5(){

    stAIConfigInfo ai_cfg;
    ai_cfg.preProcessThreadPriority = 0;
    ai_cfg.inferThreadCnt           = 1;
    ai_cfg.inferThreadPriority      = 0;

    AIRuntimeInterface* ai_obj = GetAIRuntime();
    ai_obj->InitRuntime(ai_cfg);

    stAIModelInfo model_cfg;

    model_cfg.modelId                        = 0;
    model_cfg.modelName                      = "best.trt.engine";
    model_cfg.modelPath                      = R"(E:\\projdata\\ZXY_RPA\\best.onnx)";
    model_cfg.modelVersion                   = 1;
    model_cfg.modelBackend                   = "onnxruntime";//tensorrt   onnxruntime
    model_cfg.inferParam.confidenceThreshold = 0.25;
    model_cfg.inferParam.maxBatchSize        = 1;
    model_cfg.inferParam.nmsThreshold        = 0.45;
    model_cfg.inferParam.maxObjectNums       = 200000;
    model_cfg.algoType                       = YOLOV5;

    ai_obj->CreateModle(model_cfg);

    // PR print_ret;
    // ai_obj->RegisterResultListener(0, (IModelResultListener*)&print_ret);

    cv::Mat     img  = cv::imread(R"(E:\projdata\ZXY_RPA\detect\POC_20240325\0B510\20231031232122_78474_C81S3A048525_A0004.jpg)");
    TaskInfoPtr task = std::make_shared<stTaskInfo>();
    task->imageData  = { img };
    task->modelId    = 0;
    task->taskId     = 0;
    task->promiseResult = new std::promise<ModelResultPtr>();
    ai_obj->CommitInferTask(task);

    
    std::promise<ModelResultPtr>* promiseResult = static_cast<std::promise<ModelResultPtr>*>(task->promiseResult);
    std::future<ModelResultPtr>   futureRst  = promiseResult->get_future();
   
    ModelResultPtr rst         = futureRst.get();
    cv::Mat        binImg      = cv::Mat::zeros(img.rows, img.cols, CV_8UC1);
    int            colletCount = 0;
    for (int i = 0; i < rst->itemList.size(); i++) {
           for (auto& box : rst->itemList[i]) {
               std::cout <<box.confidence <<box.code <<box.shape << std::endl;
               std::cout << box.points[0].x << " " << box.points[0].y << " " << box.points[1].x << " "<<box.points[1].y << std::endl;
               draw_rst(img, box);
               cv::imshow("111",img);
               cv::waitKey(0);
           }
    }
ai_obj->DestroyModle(0);
// ai_obj->DestoryRuntime();    
}

inline void test_trt_ocr()
{
    stAIConfigInfo ai_cfg;
    ai_cfg.preProcessThreadPriority = 0;
    ai_cfg.inferThreadCnt           = 1;
    ai_cfg.inferThreadPriority      = 0;

    AIRuntimeInterface* ai_obj = GetAIRuntime();
    ai_obj->InitRuntime(ai_cfg);

    stAIModelInfo model_cfg;

    model_cfg.modelId                        = 0;
    model_cfg.modelName                      = "det.trt.engine";
    model_cfg.modelPath                      = "E:\\demo\\rep\\AIFramework\\models\\ort_models\\ch_PP-OCRv4_det_infer\\reshape\\det.trt.engine";
    model_cfg.modelVersion                   = 1;
    model_cfg.modelBackend                   = "tensorrt";
    model_cfg.inferParam.confidenceThreshold = 0.25;
    model_cfg.inferParam.maxBatchSize        = 1;
    //model_cfg.inferParam.nmsThreshold        = 0.6;
    //model_cfg.inferParam.maxObjectNums       = 200000;
    model_cfg.algoType                       = OCR_DET;

    ai_obj->CreateModle(model_cfg);

   

    cv::Mat     img     = cv::imread(R"(C:\Users\Administrator\Desktop\333.jpg)");
    TaskInfoPtr task    = std::make_shared<stTaskInfo>();
    task->imageData     = { img };
    task->modelId       = 0;
    task->taskId        = 0;
    task->promiseResult = new std::promise<ModelResultPtr>();
    ai_obj->CommitInferTask(task);

    std::promise<ModelResultPtr>* promiseResult = static_cast<std::promise<ModelResultPtr>*>(task->promiseResult);
    std::future<ModelResultPtr>   futureRst     = promiseResult->get_future();

    ModelResultPtr rst         = futureRst.get();
    cv::Mat        binImg      = cv::Mat::zeros(img.rows, img.cols, CV_8UC1);
    int            colletCount = 0;
    std::vector<cv::Mat> ret_img;

    for (int i = 0; i < rst->itemList.size(); i++) {
           auto ret = rst->itemList[i];
           for (int j = 0; j < ret.size(); j++) {
               std::vector<std::vector<int>> ocr_det;
               std::vector<int>              tmp_1;
               tmp_1.emplace_back(ret[j].points[0].x);
               tmp_1.emplace_back(ret[j].points[0].y);

               std::vector<int> tmp_2;
               tmp_2.emplace_back(ret[j].points[1].x);
               tmp_2.emplace_back(ret[j].points[1].y);

               std::vector<int> tmp_3;
               tmp_3.emplace_back(ret[j].points[2].x);
               tmp_3.emplace_back(ret[j].points[2].y);

               std::vector<int> tmp_4;
               tmp_4.emplace_back(ret[j].points[3].x);
               tmp_4.emplace_back(ret[j].points[3].y);
           
               ocr_det.emplace_back(tmp_1);
               ocr_det.emplace_back(tmp_2);
               ocr_det.emplace_back(tmp_3);
               ocr_det.emplace_back(tmp_4);

               cv::Mat crop_img_tmp = GetRotateCropImage(img, ocr_det);
               if (!crop_img_tmp.empty())
                   ret_img.emplace_back(crop_img_tmp);
           }
            
    }
}

inline void test_ort_ocr(){


    stAIConfigInfo ai_cfg;
    ai_cfg.preProcessThreadPriority = 0;
    ai_cfg.inferThreadCnt           = 1;
    ai_cfg.inferThreadPriority      = 0;

    AIRuntimeInterface* ai_obj = GetAIRuntime();
    ai_obj->InitRuntime(ai_cfg);

    stAIModelInfo model_cfg;

    model_cfg.modelId                        = 0;
    model_cfg.modelName                      = "det.trt.engine";
    model_cfg.modelPath                      = "E:\\project\\EFEM\\tv_algorithm\\build\\windows\\x64\\release\\ort_models\\ch_PP-OCRv4_rec_infer\\reshape\\rec_new.onnx";
    model_cfg.modelVersion                   = 1;
    model_cfg.modelBackend                   = "onnxruntime";
    model_cfg.inferParam.confidenceThreshold = 0.25;
    model_cfg.modleLabelPath                 = "E:\\project\\EFEM\\tv_algorithm\\build\\windows\\x64\\release\\ort_models\\ppocr_keys_v1.txt";
    model_cfg.inferParam.maxBatchSize        = 1;
    // model_cfg.inferParam.nmsThreshold        = 0.6;
    // model_cfg.inferParam.maxObjectNums       = 200000;
    model_cfg.algoType = OCR_REC;

    ai_obj->CreateModle(model_cfg);

    cv::Mat     img     = cv::imread(R"(E:\\project\\EFEM\\tv_algorithm\\build\\windows\\x64\\release\\test_algo\\ocr.png)");
    TaskInfoPtr task    = std::make_shared<stTaskInfo>();
    task->imageData     = { img };
    task->modelId       = 0;
    task->taskId        = 0;
    task->promiseResult = new std::promise<ModelResultPtr>();
    ai_obj->CommitInferTask(task);

    std::promise<ModelResultPtr>* promiseResult = static_cast<std::promise<ModelResultPtr>*>(task->promiseResult);
    std::future<ModelResultPtr>   futureRst     = promiseResult->get_future();

    ModelResultPtr       rst         = futureRst.get();
 /*   cv::Mat              binImg      = cv::Mat::zeros(img.rows, img.cols, CV_8UC1);
    int                  colletCount = 0;
    std::vector<cv::Mat> ret_img;*/

     for (int i = 0; i < rst->itemList.size(); i++) {
           for (auto& box : rst->itemList[i]) {
               
            std::cout << "model: " << box.confidence << " " << box.ocr_str << " " << std::endl;
            // double area = std::abs(box.points[0].x - box.points[1].x) * std::abs(box.points[0].y - box.points[1].y);
                  
               
           }
    }
}