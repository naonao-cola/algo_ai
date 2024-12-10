#ifdef _WIN32
#if USE_ORT

#include "../../../include/private/ort/ort_app_yolo8/ort_yolo8.h"
#include "../../../include/private/airuntime/logger.h"
#include "../../../include/private/ort/ort_common/ort_infer.hpp"
#include "../../../include/private/ort/ort_common/ort_infer_schedule.hpp"
#include "../../../include/private/ort/ort_common/ort_utility.h"

namespace ort::yolo8 {

using BoxArray = Algo::BoxArray;
using Box      = Algo::Box;
using ControllerImpl =
    ort::InferController<cv::Mat, BoxArray, std::tuple<std::string, int>, int>;

using YoloV5ScaleParams = struct
{
    float r;
    int   dw;
    int   dh;
    int   new_unpad_w;
    int   new_unpad_h;
    bool  flag;
};

enum NMS
{
    HARD   = 0,
    BLEND  = 1,
    OFFSET = 2
};

/*辅助函数前向声明*/
void resize_unscale(const cv::Mat& mat, cv::Mat& mat_rs, int target_height, int target_width, YoloV5ScaleParams& scale_params);
void generate_bboxes(const YoloV5ScaleParams& scale_params, BoxArray& bbox_collection, std::vector<Ort::Value>& output_tensors, std::vector<std::vector<int64_t>> output_node_dims, float score_threshold, int img_height, int img_width, int max_nms);
void nms(BoxArray& input, BoxArray& output, float iou_threshold, unsigned int topk, unsigned int nms_type);

// 子类
class InferImpl : public Algo::Infer, public ControllerImpl
{
public:
    virtual ~InferImpl() override { stop(); }

    virtual bool startup(const std::string& onnx_file, float confidence_threshold = 0.25f, float nms_threshold = 0.45f, int max_objects = 1024)
    {
        _mean_val  = 0.f;
        _scale_val = 1 / 255.f;

        _confidence_threshold = confidence_threshold;
        _nms_threshold        = nms_threshold;
        _max_objects          = max_objects;
        return ControllerImpl::startup(std::make_tuple(onnx_file, 0));
    }

    bool set_param(const json& config) override
    {
        _confidence_threshold =
            get_param<float>(config, "confidence_threshold", _confidence_threshold);
        _nms_threshold  = get_param<float>(config, "nms_threshold", _nms_threshold);
        _max_objects    = get_param<int>(config, "max_objects", _max_objects);
        _max_batch_size = get_param<int>(config, "max_batch_size", _max_batch_size);
        return true;
    }

    bool preprocess(Job& job, const cv::Mat& image) override
    {
        job.preTime.start();
        if (image.empty()) {
            LOG_INFOE("Image is empty");
            return false;
        }
        cv::Mat mat_rs;
        resize_unscale(image, mat_rs, _input_shapes.at(0).at(2), _input_shapes.at(0).at(3), _scale_params);
        cv::Mat canvas;
        cv::cvtColor(mat_rs, canvas, cv::COLOR_BGR2RGB);
        ort::utils::normalize_inplace(canvas, _mean_val, _scale_val);

        job.input_value = std::shared_ptr<float[]>(
            new float[canvas.rows * canvas.cols * canvas.channels()],
            [](float* s) { delete s; });
        ort::utils::permute(canvas, job.input_value.get());
        job.input = image.clone();
        job.preTime.stop();
        return true;
    }

    bool postprocess(Job& job, std::vector<Ort::Value>& output_tensors) override
    {
        TimeCost post_time_cost;
        post_time_cost.start();
        job.hostTime.start();
        BoxArray bbox_collection;
        generate_bboxes(_scale_params, bbox_collection, output_tensors, _output_shapes, _confidence_threshold, job.input.rows, job.input.cols, _max_nms);
        BoxArray detected_boxes;
        nms(bbox_collection, detected_boxes, _nms_threshold, _max_objects, NMS::HARD);

        job.hostTime.stop();
        detected_boxes.pre_time   = job.preTime.get_cost_time();
        detected_boxes.infer_time = job.inferTime.get_cost_time();
        detected_boxes.host_time  = job.hostTime.get_cost_time();
        detected_boxes.total_time = detected_boxes.pre_time +
                                    detected_boxes.infer_time +
                                    detected_boxes.host_time;
        job.pro->set_value(detected_boxes);

        return true;
    }

    std::vector<std::shared_future<BoxArray>>
    commits(const std::vector<cv::Mat>& images) override
    {
        return ControllerImpl::commits(images);
    }

    std::shared_future<BoxArray> commit(const cv::Mat& image) override
    {
        return ControllerImpl::commit(image);
    }

    json infer_info() override { return _model_info; }

private:
    float _confidence_threshold = 0;
    float _nms_threshold        = 0;
    int   _max_objects          = 1024;

    int                _input_height;
    int                _input_width;
    YoloV5ScaleParams  _scale_params;
    float              _mean_val  = 0.f;
    float              _scale_val = 1 / 255.f;
    const unsigned int _max_nms = 30000;
};

void resize_unscale(const cv::Mat& mat, cv::Mat& mat_rs, int target_height, int target_width, YoloV5ScaleParams& scale_params)
{
    if (mat.empty())
        return;
    auto img_height = mat.rows;
    auto img_width  = mat.cols;

    mat_rs =
        cv::Mat(target_height, target_width, CV_8UC3, cv::Scalar(114, 114, 114));
    // scale ratio (new / old) new_shape(h,w)
    float w_r = (float)target_width / (float)img_width;
    float h_r = (float)target_height / (float)img_height;
    float r   = std::min(w_r, h_r);
    // compute padding
    auto new_unpad_w = static_cast<int>((float)img_width * r);   // floor
    auto new_unpad_h = static_cast<int>((float)img_height * r);  // floor
    int  pad_w       = target_width - new_unpad_w;               // >=0
    int  pad_h       = target_height - new_unpad_h;              // >=0

    int dw = pad_w / 2;
    int dh = pad_h / 2;

    // resize with unscaling
    cv::Mat new_unpad_mat;
    // cv::Mat new_unpad_mat = mat.clone(); // may not need clone.
    cv::resize(mat, new_unpad_mat, cv::Size(new_unpad_w, new_unpad_h));
    new_unpad_mat.copyTo(mat_rs(cv::Rect(dw, dh, new_unpad_w, new_unpad_h)));

    // record scale params.
    scale_params.r           = r;
    scale_params.dw          = dw;
    scale_params.dh          = dh;
    scale_params.new_unpad_w = new_unpad_w;
    scale_params.new_unpad_h = new_unpad_h;
    scale_params.flag        = true;
}

void generate_bboxes(const YoloV5ScaleParams& scale_params, BoxArray& bbox_collection, std::vector<Ort::Value>& output_tensors, std::vector<std::vector<int64_t>> output_node_dims, float score_threshold, int img_height, int img_width, int max_nms)
{
    // 图片的输出
    float* all_data = output_tensors.at(0).GetTensorMutableData<float>();

    auto    pred_dims         = output_node_dims.at(0);                      // (1,84,8400)，第一个输出的维度
    int64_t one_output_length = pred_dims[0] * pred_dims[1] * pred_dims[2];  // 一张图片的输出长度
    int     net_width         = pred_dims[1];
    // 目前输出结果是 1 *84 * 8500，中间的84， 4是矩形框的中心坐标+宽高，80是类别
    // 输入为1*3*3200*3200时，输出为1*5*210000， 中间的5,。 4是 矩形框的中心坐标+ 宽高。 1是类别。
    //  [1, 84, 8400]->[1, 8400, 84], 1*5*210000 -> 1*210000*5
    cv::Mat output0 = cv::Mat(cv::Size((int)pred_dims[2], (int)pred_dims[1]), CV_32F, all_data).t();
    float*  pdata   = (float*)output0.data;
    int     rows    = output0.rows;  // 预测框的数量 8400.现在是210000

    float r_  = scale_params.r;
    int   dw_ = scale_params.dw;
    int   dh_ = scale_params.dh;

    bbox_collection.clear();
    unsigned int count = 0;
    // 一张图片的预测框
    for (int i = 0; i < rows; i++) {
        cv::Mat   scores(1, pred_dims[1] - 4, CV_32F, pdata + 4);
        cv::Point class_id_point;
        double    max_class_socse;
        cv::minMaxLoc(scores, 0, &max_class_socse, 0, &class_id_point);
        max_class_socse = (float)max_class_socse;

        if (max_class_socse >= score_threshold) {
            // 映射到原图。
            float cx = pdata[0];
            float cy = pdata[1];
            float w  = pdata[2];
            float h  = pdata[3];
            float x1 = ((cx - w / 2.f) - (float)dw_) / r_;
            float y1 = ((cy - h / 2.f) - (float)dh_) / r_;
            float x2 = ((cx + w / 2.f) - (float)dw_) / r_;
            float y2 = ((cy + h / 2.f) - (float)dh_) / r_;
            Box   box;
            box.left        = std::max(0.f, x1);
            box.top         = std::max(0.f, y1);
            box.right       = std::min(x2, (float)img_width - 1.f);
            box.bottom      = std::min(y2, (float)img_height - 1.f);
            box.confidence  = max_class_socse;
            box.class_label = class_id_point.x;
            bbox_collection.push_back(box);
            count += 1;  // limit boxes for nms.
            if (count > max_nms)
                break;
        }
        pdata += net_width;  // 下一个预测框
    }
}

void nms(BoxArray& input, BoxArray& output, float iou_threshold, unsigned int topk, unsigned int nms_type)
{
    if (nms_type == NMS::BLEND)
        ort::utils::blending_nms(input, output, iou_threshold, topk);
    else if (nms_type == NMS::OFFSET)
        ort::utils::offset_nms(input, output, iou_threshold, topk);
    else
        ort::utils::hard_nms(input, output, iou_threshold, topk);
}

std::shared_ptr<Algo::Infer> create_infer(const std::string& onnx_file, float confidence_threshold, float nms_threshold, int max_objects)
{
    std::shared_ptr<InferImpl> instance = std::make_shared<InferImpl>();
    if (!instance->startup(onnx_file, confidence_threshold, nms_threshold, max_objects)) {
        instance.reset();
    }
    return instance;
}
}  // namespace ort::yolo8


#endif //USE_ORT
#elif __linux__
#endif