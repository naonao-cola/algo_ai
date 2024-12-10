#ifndef YOLO
#define YOLO

#include <vector>
#include <memory>
#include <string>
#include <future>
#include <opencv2/opencv.hpp>

#include "../trt_common/trt-tensor.hpp"
#include "../../airuntime/inference.h"

/**
 * @brief Support YoloX and YoloV5
 */
namespace Yolo{

    using namespace std;
    using namespace Algo;

    enum class Type : int{
        V5 = 0,
        X  = 1,
        V3 = V5,
    };

    //Type get_type_by_int(int t) {
    //    if (t == 0) return Type::V5;
    //    if (t == 1) return Type::X;
    //    if (t == 2) return Type::V3;
    //    return Type::V5;
    //}

    enum class NMSMethod : int{
        CPU = 0,         // General, for estimate mAP
        FastGPU = 1      // Fast NMS with a small loss of accuracy in corner cases
    };

    /**
     * @brief tranform image to tensor 
     * yolo image preprocess with normalize and affine transform by cuda.
     * @param image input image
     * @param tensor output
     * @param type [yolov5 | yolov3 | yolovx]
     * @param ibatch 
     */
    void image_to_tensor(const cv::Mat& image, shared_ptr<TRT::Tensor>& tensor, Type type, int ibatch);

    /**
     * @brief Create a yolo infer object
     * 
     * @param engine_file tensorRT file path
     * @param type yolo algorithm type [yolov3 | yolov5 | yolox]
     * @param gpuid gpu id which inference
     * @param confidence_threshold hostprocess confidence threshold
     * @param nms_threshold hostprocess nms thrshold
     * @param nms_method method of nms [cpu | FastGPU]
     * @param max_objects hostprocess. maxinum object number
     * @param use_multi_preprocess_stream 
     * @return shared_ptr<AlgoInterface::Infer> if create fail, will return nullptr.
     */
    shared_ptr<Algo::Infer> create_infer(
        const string& engine_file, Type type, int gpuid,
        float confidence_threshold=0.25f, float nms_threshold=0.5f,
        NMSMethod nms_method = NMSMethod::FastGPU, int max_objects = 1024,
        bool use_multi_preprocess_stream = false
    );

    /**
     * @brief transform yolo algorithm enmum Type to const char*
     * 
     * @param type 
     * @return const char* ["yolov5", "yolox", "UNKNOWN"]
     */
    const char* type_name(Type type);

}; // namespace Yolo

#endif // !YOLO
