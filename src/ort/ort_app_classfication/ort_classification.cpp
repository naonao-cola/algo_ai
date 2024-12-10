#ifdef _WIN32
#if USE_ORT

#include "../../../include/private/ort/ort_app_classification/ort_classification.h"
#include "../../../include/private/airuntime/logger.h"
#include "../../../include/private/ort/ort_common/ort_infer.hpp"
#include "../../../include/private/ort/ort_common/ort_infer_schedule.hpp"
#include "../../../include/private/ort/ort_common/ort_utility.h"

namespace ort::classification {

using BoxArray = Algo::BoxArray;
using Box      = Algo::Box;
using ControllerImpl =
    ort::InferController<cv::Mat, BoxArray, std::tuple<std::string, int>, int>;

// ����
class InferImpl : public Algo::Infer, public ControllerImpl
{
public:
    virtual ~InferImpl() override { stop(); }

    virtual bool startup(const std::string& onnx_file, float confidence_threshold = 0.25f, int gpuid = 0)
    {
        _mean_val             = 0.f;
        _scale_val            = 1 / 255.f;
        _confidence_threshold = confidence_threshold;
        return ControllerImpl::startup(std::make_tuple(onnx_file, 0));
    }

    bool set_param(const json& config) override
    {
        _confidence_threshold =
            get_param<float>(config, "confidence_threshold", _confidence_threshold);
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

        _input_width  = _input_shapes.at(0).at(3);
        _input_height = _input_shapes.at(0).at(2);

        cv::Mat canvas;
        cv::cvtColor(image, canvas, cv::COLOR_BGR2RGB);
        cv::resize(canvas, canvas, cv::Size(_input_width, _input_height));
        ort::utils::normalize_inplace(canvas, _mean_val, _scale_val);
        std::vector<float> input_values_handler;
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

        // ��������
        int         number_class = _output_shapes.at(0).at(1);
        Ort::Value& pred_out     = output_tensors.at(0);

        std::vector<float> featureVector;
        featureVector.resize(number_class);
        for (int i = 0; i < number_class; i++) {
            featureVector[i] = float(pred_out.At<float>({ 0, i }));
        }

        int predict_label =
            std::max_element(featureVector.begin(), featureVector.end()) -
            featureVector.begin();
        float confidence = 1 / (1 + exp(-featureVector[predict_label]));

        Box      box(0, 0, 0, 0, confidence, predict_label);
        BoxArray detected_boxes;
        detected_boxes.emplace_back(box);

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
    int   _input_height;
    int   _input_width;
    float _mean_val  = 0.f;
    float _scale_val = 1 / 255.f;
};

std::shared_ptr<Algo::Infer> create_infer(const std::string& onnx_file, float confidence_threshold, int gpuid)
{
    std::shared_ptr<InferImpl> instance = std::make_shared<InferImpl>();
    if (!instance->startup(onnx_file, confidence_threshold, gpuid)) {
        instance.reset();
    }
    return instance;
}
}  // namespace ort::classification


#endif //USE_ORT

#elif __linux__
#endif