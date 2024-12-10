#ifndef YOLO8_OBB_HPP
#define YOLO8_OBB_HPP

#include <vector>
#include <memory>
#include <string>
#include <future>
#include <opencv2/opencv.hpp>
#include "../../airuntime/inference.h"

namespace yolo8_obb {
    using namespace cv;
    using namespace std;
    using BoxArray = Algo::BoxArray;
    using Box      = Algo::Box;
    using Infer    = Algo::Infer;
    using namespace Algo;


    enum class NMSMethod : int{
        CPU = 0,         // General, for estimate mAP
        FastGPU = 1      // Fast NMS with a small loss of accuracy in corner cases
    };

    void image_to_tensor(const cv::Mat& image, shared_ptr<TRT::Tensor>& tensor, int ibatch);

    shared_ptr<Infer> create_infer(
        const string& engine_file, int gpuid,
        float confidence_threshold=0.25f, float nms_threshold=0.5f,
        NMSMethod nms_method = NMSMethod::FastGPU, int max_objects = 1024,
        bool use_multi_preprocess_stream = false
    );

}; // namespace yolo8_obb
#endif // !__TRT_YOLO8_OBB_H__
