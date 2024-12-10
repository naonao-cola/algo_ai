#ifdef _WIN32
#if USE_ORT

#include "../../../include/private/ort/ort_app_ocr/ort_ocr_det.h"
#include "../../../include/private/airuntime/logger.h"
#include "../../../include/private/ort/ort_common/ort_infer.hpp"
#include "../../../include/private/ort/ort_common/ort_infer_schedule.hpp"
#include "../../../include/private/ort/ort_common/ort_utility.h"
#include "../../../include/private/trt/trt_app_ocr/ocr_utility.h"

namespace ort::ocr::det {

using BoxArray = Algo::BoxArray;
using Box      = Algo::Box;
using ControllerImpl =
    ort::InferController<cv::Mat, BoxArray, std::tuple<std::string, int>, int>;

// 子类
class InferImpl : public Algo::Infer, public ControllerImpl
{
public:
    ~InferImpl() override { stop(); }

    virtual bool startup(const std::string& onnx_file, int gpuid, int kernel_size, bool use_dilation, bool enable_detmat, float det_db_box_thresh, float det_db_unclip_ratio, int max_side_len)
    {
        kernel_size_ = kernel_size;
        _use_dilation = use_dilation;
        enable_detmat_ = enable_detmat;
        _det_db_box_thresh = det_db_box_thresh;
        _det_db_unclip_ratio = det_db_unclip_ratio;
        _max_side_len = max_side_len;
        return ControllerImpl::startup(std::make_tuple(onnx_file, 0));
    }

    bool set_param(const json& config) override
    {
        _det_db_box_thresh = get_param<float>(config, "det_db_box_thresh", _det_db_box_thresh);
        _det_db_unclip_ratio = get_param<float>(config, "det_db_unclip_ratio", _det_db_unclip_ratio);
        enable_detmat_ = get_param<bool>(config, "enableDetMat", enable_detmat_);
        _use_dilation = get_param<float>(config, "useDilat", _use_dilation);
        _max_batch_size = get_param<int>(config, "max_batch_size", _max_batch_size);
        _max_side_len = get_param<int>(config, "max_side_len", _max_side_len);
        return true;
    }

    bool preprocess(Job& job, const cv::Mat& image) override
    {
        job.preTime.start();
        if (image.empty()) {
            LOG_INFOE("Image is empty");
            return false;
        }
        job.input = image.clone();

        //cv::Mat srcimg;
        cv::Mat resize_img;
        //image.copyTo(srcimg);
        // 加一个resize
        cv::resize(image, resize_img, cv::Size(_input_shapes.at(0).at(3), _input_shapes.at(0).at(2)));
        ort::utils::resize_img_type0(resize_img, resize_img, this->_max_side_len, _ratio_h, _ratio_w);
        /*记录宽高*/
        _ratio_h = resize_img.rows * 1.0f / image.rows;
        _ratio_w = resize_img.cols * 1.0f / image.cols;
        ort::utils::normalize_img_type0(&resize_img, this->_mean, this->_scale, true);
        job.input_value = std::shared_ptr<float[]>(
            new float[resize_img.rows * resize_img.cols * resize_img.channels()],
            [](float* s) { delete s; });
        ort::utils::permute(resize_img, job.input_value.get());
        _resize_img_h = resize_img.rows;
        _resize_img_w = resize_img.cols;
        job.preTime.stop();
        return true;
    }

    bool postprocess(Job& job, std::vector<Ort::Value>& output_tensors) override
    {
        TimeCost post_time_cost;
        post_time_cost.start();
        job.hostTime.start();
        int   n2           = _output_shapes.at(0).at(2);
        int   n3           = _output_shapes.at(0).at(3);
        int   n            = n2 * n3;  // output_h * output_w
        auto& output_boxes = job.output;

        std::vector<float>                         pred(n, 0.0);
        std::vector<unsigned char>                 cbuf(n, ' ');
        std::vector<std::vector<std::vector<int>>> boxes;
        Ort::Value&                                pred_output = output_tensors.at(0);
        /*float*                                       pred_float  =
         * output_tensors[0].GetTensorMutableData<float>();*/

        // 输出维度为1*1*640*640
        for (int i = 0; i < n; i++) {
            pred[i] = float(pred_output.At<float>({ 0, 0, i / n2, i % n3 }));
            cbuf[i] =
                (unsigned char)(pred_output.At<float>({ 0, 0, i / n2, i % n3 }) * 255);
            /* pred[i] = float(pred_float[i/n2 + i%n3]);
            cbuf[i] = (unsigned char)(pred_float[i / n2 + i % n3] * 255);*/
        }
        cv::Mat      cbuf_map(n2, n3, CV_8UC1, (unsigned char*)cbuf.data());
        cv::Mat      pred_map(n2, n3, CV_32F, (float*)pred.data());
        const double threshold = this->_det_db_thresh * 255;
        const double maxvalue  = 255;
        cv::Mat      bit_map;
        cv::threshold(cbuf_map, bit_map, threshold, maxvalue, cv::THRESH_BINARY);
        if (this->_use_dilation) {
            cv::Mat dila_ele =
                cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernel_size_, kernel_size_));
            cv::dilate(bit_map, bit_map, dila_ele);
        }
        boxes = OCR::utility::BoxesFromBitmap(
            pred_map, bit_map, this->_det_db_box_thresh, this->_det_db_unclip_ratio,
            this->_use_polygon_score);

        //cv::Mat tmp_src = job.input.clone();
        /*cv::Mat tmp_src =
            cv::Mat::zeros(cv::Size(_resize_img_w, _resize_img_h), CV_8UC3);*/
        boxes = OCR::utility::FilterTagDetRes(
            boxes, _ratio_h, _ratio_w,
            job.input.clone());  // 将resize_img中得到的bbox 映射回srcing中的bbox

        for (int i = 0; i < boxes.size(); i++) {
            Box box;
            //box.ocr_det = boxes[i];
            if (!enable_detmat_) {
                box.ocr_det = boxes[i];
            }
            else {
                std::vector<std::vector<int>> ocr_det1;
                for (int k = 0; k < 4; k++) {
                    ocr_det1.push_back({ boxes[i][k][0], boxes[i][k][1] });
                }
                box.detMat = OCR::utility::GetRotateCropImage(job.input, ocr_det1);
            }

            output_boxes.emplace_back(box);
        }

        job.hostTime.stop();
        output_boxes.pre_time   = job.preTime.get_cost_time();
        output_boxes.infer_time = job.inferTime.get_cost_time();
        output_boxes.host_time  = job.hostTime.get_cost_time();
        output_boxes.total_time = output_boxes.pre_time + output_boxes.infer_time +
                                  output_boxes.host_time;
        _model_info["infer_time"] = output_boxes.total_time;
        job.pro->set_value(output_boxes);
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
    bool               _use_polygon_score    = false;
    double             _det_db_box_thresh    = 0.5;
    double             _det_db_unclip_ratio  = 2.0;
    bool               _use_dilation         = false;
    bool                enable_detmat_       = false;
    int                 kernel_size_         = 3;
    double             _det_db_thresh        = 0.3;
    float              _confidence_threshold = 0;
    int                _max_side_len         = 640;
    float              _ratio_h              = 0.f;
    float              _ratio_w              = 0.f;
    int                _resize_img_h         = 0;
    int                _resize_img_w         = 0;
    std::vector<float> _mean                 = { 0.485f, 0.456f, 0.406f };
    std::vector<float> _scale                = { 1 / 0.229f, 1 / 0.224f, 1 / 0.225f };
};

std::shared_ptr<Algo::Infer> create_infer(const std::string& onnx_file, int gpuid, int kernel_size, bool use_dilation, bool enable_detmat, float det_db_box_thresh, float det_db_unclip_ratio, int max_side_len)
{
    std::shared_ptr<InferImpl> instance = std::make_shared<InferImpl>();
    if (!instance->startup(onnx_file, gpuid, kernel_size, use_dilation, enable_detmat, det_db_box_thresh, det_db_unclip_ratio, max_side_len)) {
        instance.reset();
    }
    return instance;
}

}  // namespace ort::ocr::det

#endif //USE_ORT
#elif __linux__
#endif