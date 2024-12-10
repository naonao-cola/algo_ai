#ifdef _WIN32
#    if USE_ORT

#        include "../../../include/private/ort/ort_app_ocr/ort_ocr_rec.h"
#        include "../../../include/private/airuntime/logger.h"
#        include "../../../include/private/ort/ort_common/ort_infer.hpp"
#        include "../../../include/private/ort/ort_common/ort_infer_schedule.hpp"
#        include "../../../include/private/ort/ort_common/ort_utility.h"
#        include "../../../include/private/trt/trt_app_ocr/ocr_utility.h"

namespace ort::ocr::rec {

using BoxArray = Algo::BoxArray;
using Box      = Algo::Box;
using ControllerImpl =
    ort::InferController<cv::Mat, BoxArray, std::tuple<std::string, int>, int>;


class InferImpl : public Algo::Infer, public ControllerImpl
{
public:
    ~InferImpl() override { stop(); }

    virtual bool startup(const std::string& onnx_file, const std::string& label_file, int gpuid, float confidence_threshold)
    {
        _label_list = OCR::utility::read_dict(label_file);
        return ControllerImpl::startup(std::make_tuple(onnx_file, 0));
    }

    bool set_param(const json& config) override
    {
        _max_batch_size    = get_param<int>(config, "max_batch_size", _max_batch_size);
        bEnableSingleChar_ = get_param<bool>(config, "enable_single", bEnableSingleChar_);
        _shif_ratio        = get_param<float>(config, "single_char_shif_ratio", _shif_ratio);
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

        cv::Mat resize_img;
        OCR::utility::crnn_resize_img(image, resize_img, _input_shapes.at(0).at(3) * 1.0 / _input_shapes.at(0).at(2), this->_rec_image_shape);
        OCR::utility::normalize(&resize_img, this->_mean, this->_scale, true);
        std::vector<cv::Mat> norm_img_batch;
        norm_img_batch.push_back(resize_img);
        job.input_value = std::shared_ptr<float[]>(
            new float[_input_shapes.at(0).at(3) * _input_shapes.at(0).at(2) * _input_shapes.at(0).at(1)],
            [](float* s) { delete s; });
        OCR::utility::permute_batch(norm_img_batch, job.input_value.get());
        job.preTime.stop();
        return true;
    }

    bool postprocess(Job& job, std::vector<Ort::Value>& output_tensors) override
    {
        TimeCost post_time_cost;
        post_time_cost.start();
        job.hostTime.start();
        auto& output_boxes = job.output;

        int                n2 = _output_shapes.at(0).at(1);
        int                n3 = _output_shapes.at(0).at(2);
        int                n  = n2 * n3;
        std::vector<float> featureVector;
        featureVector.resize(n2 * n3);
        std::vector<std::vector<float>> featureMatrix(n2, std::vector<float>(n3));

        Ort::Value& pred_out = output_tensors.at(0);

        for (int i = 0; i < n; i++) {
            featureVector[i] = float(pred_out.At<float>({ 0, i / n3, i % n3 }));
        }

        std::vector<std::vector<float>>().swap(_featureVectors);
        _featureVectors.clear();
        _featureVectors.emplace_back(std::move(featureVector));

        std::vector<int>                            predict_shape = { 1, n2, n3 };
        std::vector<std::pair<std::string, double>> rec_conf_res;
        std::vector<std::vector<cv::Point>>         ocr_single_det;
        std::vector<int>                            ocr_char_index;
        int                                         argmax_idx;
        int                                         last_index  = 0;
        int                                         count       = 0;
        int                                         m           = 0;
        int                                         sum         = 0;
        float                                       score       = 0.f;
        float                                       max_value   = 0.0f;
        float                                       ratio       = float(job.input.rows) / float(_input_shapes.at(0).at(2));
        float                                       src_ratio   = float(job.input.cols) / float(job.input.rows);
        std::vector<float>                          character_ratio;
        std::string                                 final_ret = "";

        for (int n = 0; n < predict_shape[1]; n++) {
            int selection = (m * predict_shape[1] + n) * predict_shape[2];
            argmax_idx    = int(OCR::utility::argmax(
                &_featureVectors[0][(m * predict_shape[1] + n) * predict_shape[2]],
                &_featureVectors[0][(m * predict_shape[1] + n + 1) * predict_shape[2]]));
            max_value     = float(*std::max_element(
                &_featureVectors[0][(m * predict_shape[1] + n) * predict_shape[2]],
                &_featureVectors[0][(m * predict_shape[1] + n + 1) * predict_shape[2]]));

            // label_list_超限判断
            if (argmax_idx > 0 && (!(n > 0 && argmax_idx == last_index)) && argmax_idx < _label_list.size()) {
                ocr_char_index.push_back(n);
                //character_ratio.push_back(float((float(n) - _shif_ratio) * 8.0f * ratio));
                if (float(_input_shapes.at(0).at(2)) * src_ratio > float(_input_shapes.at(0).at(3))) {
                    ratio = float(job.input.cols) / float(_input_shapes.at(0).at(3));
                }  
                character_ratio.push_back(float(n - _shif_ratio) * float(_input_shapes.at(0).at(3)) / float(n2) * ratio);
                score += max_value;
                count += 1;
                // ocr结果拆分成单个字符
                if (bEnableSingleChar_) {
                    rec_conf_res.push_back(std::make_pair(_label_list[argmax_idx - 1], max_value));
                }
                final_ret = final_ret + _label_list[argmax_idx - 1];
            }
            last_index = argmax_idx;
        }

        float mean_count = 0;
        for (int i = 0; i < character_ratio.size(); i++) {
            std::vector<cv::Point> singlePoint;
            if (character_ratio[i] < 0) {
                character_ratio[i] = 0;
            }
            if (i == character_ratio.size() - 1) {
                singlePoint.push_back(cv::Point(character_ratio[i], 3));
                singlePoint.push_back(cv::Point(character_ratio[i] + mean_count, job.input.rows - 4));
                // cv::rectangle(job.input, cv::Point(character_ratio[i], 3), cv::Point(character_ratio[i] + mean_count, job.input.rows - 4), cv::Scalar(0, 255, 0));
            }
            else {
                singlePoint.push_back(cv::Point(character_ratio[i], 3));
                singlePoint.push_back(cv::Point(character_ratio[i + 1], job.input.rows - 4));
                // cv::rectangle(job.input, cv::Point(character_ratio[i], 3), cv::Point(character_ratio[i + 1], job.input.rows - 4), cv::Scalar(0, 255, 0));
            }
            ocr_single_det.push_back(singlePoint);
            mean_count = (character_ratio[i + 1] - character_ratio[i]);
        }
        score /= count;
        Box box(score, final_ret);
        if (bEnableSingleChar_ && !rec_conf_res.empty()) {
            box.rec_conf_res  = rec_conf_res;
            box.rec_ratio_res = character_ratio;
            box.ocr_single_pos = ocr_single_det;
            box.ocr_char_index = ocr_char_index;
        }

        output_boxes.emplace_back(box);

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
    bool                            bEnableSingleChar_ = true;
    int                             _rec_img_h         = 48;  ////32
    int                             _rec_img_w         = 640;
    float                           _shif_ratio        = 1.5f;
    std::vector<int>                _rec_image_shape   = { 3, _rec_img_h, _rec_img_w };
    std::vector<float>              _mean              = { 0.5f, 0.5f, 0.5f };
    std::vector<float>              _scale             = { 1 / 0.5f, 1 / 0.5f, 1 / 0.5f };
    std::vector<std::string>        _label_list;
    std::vector<std::vector<float>> _featureVectors;
};

std::shared_ptr<Algo::Infer> create_infer(const std::string& onnx_file, const std::string& label_file, int gpuid, float confidence_threshold)
{
    std::shared_ptr<InferImpl> instance = std::make_shared<InferImpl>();
    if (!instance->startup(onnx_file, label_file, gpuid, confidence_threshold)) {
        instance.reset();
    }
    return instance;
}

}  // namespace ort::ocr::rec

#    endif  // USE_ORT
#elif __linux__
#endif