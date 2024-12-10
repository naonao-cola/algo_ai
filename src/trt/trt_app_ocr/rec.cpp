#if USE_TRT

#    include <atomic>
#    include <condition_variable>
#    include <mutex>
#    include <queue>

#    include "../../../include/private/airuntime/logger.h"
#    include "../../../include/public/AIRuntimeUtils.h"

#    include "../../../include/private/trt/trt_common/cuda-tools.hpp"
#    include "../../../include/private/trt/trt_common/monopoly_allocator.hpp"
#    include "../../../include/private/trt/trt_common/time_cost.h"
#    include "../../../include/private/trt/trt_common/trt_infer.hpp"
#    include "../../../include/private/trt/trt_common/trt_infer_schedule.hpp"

// #include "algo/algo_interface.h"
#    include "../../../include/private/trt/trt_app_ocr/ocr_utility.h"
#    include "../../../include/private/trt/trt_app_ocr/rec.hpp"

namespace OCR {
namespace rec {
using namespace cv;
using namespace std;
using BoxArray = Algo::BoxArray;
using Box      = Algo::Box;
using Infer    = Algo::Infer;

using ControllerImpl =
    InferController<Mat, /*input*/ BoxArray,
                    /*output*/ tuple<string, int> /* start param*/>;
class InferImpl : public Infer, public ControllerImpl
{
public:
    virtual ~InferImpl() { stop(); }

    virtual bool startup(const string& file, int gpuid, float confidence_threshold, std::string label_file)
    {
        confidence_threshold_ = confidence_threshold;
        label_list_           = OCR::utility::read_dict(label_file);
        return ControllerImpl::startup(make_tuple(file, gpuid));
    }

    virtual bool set_param(const json& config) override
    {
        bEnableSingleChar_ = get_param<bool>(config, "enableSingle", bEnableSingleChar_);
        _shif_ratio        = get_param<float>(config, "single_char_shif_ratio", _shif_ratio);
        return true;
    }

    virtual void worker(promise<bool>& result) override
    {
        string file  = get<0>(start_param_);
        int    gpuid = get<1>(start_param_);

        TRT::set_device(gpuid);
        auto engine = TRT::load_infer(file);
        if (engine == nullptr) {
            LOG_INFOE("Engine {} load failed", file.c_str());
            result.set_value(false);
            return;
        }

        engine->print();

        int max_batch_size = engine->get_max_batch_size();

        auto input  = engine->tensor(engine->get_input_name(0));   // engine->get_input_name(0) //"x"
        auto output = engine->tensor(engine->get_output_name(0));  // engine->get_output_name(0) "softmax_11.tmp_0"

        input_height_              = input->size(2);
        input_width_               = input->size(3);
        model_info_["memory_size"] = engine->get_device_memory_size() >> 20;
        model_info_["dims"]        = { input->size(0), input->size(1), input->size(2), input->size(3) };

        tensor_allocator_ =
            make_shared<MonopolyAllocator<TRT::Tensor>>(max_batch_size * 2);
        stream_ = engine->get_stream();
        gpu_    = gpuid;
        result.set_value(true);
        input->resize_single_dim(0, max_batch_size).to_gpu();

        vector<Job> fetch_jobs;

        while (get_jobs_and_wait(fetch_jobs, max_batch_size)) {
            int infer_batch_size = fetch_jobs.size();
            input->resize_single_dim(0, infer_batch_size);
            for (int ibatch = 0; ibatch < infer_batch_size; ++ibatch) {
                auto& job  = fetch_jobs[ibatch];
                auto& mono = job.mono_tensor->data();

                if (mono->get_stream() != stream_) {
                    // synchronize preprocess stream finish
                    checkCudaRuntime(cudaStreamSynchronize(mono->get_stream()));
                }
                input->copy_from_gpu(input->offset(ibatch), mono->gpu(), mono->count());
                job.mono_tensor->release();
            }
            TRT::TimeCost infer_time_cost;
            infer_time_cost.start();
            engine->forward();
            infer_time_cost.stop();

            TRT::TimeCost post_time_cost;
            post_time_cost.start();

            int output_size = output->size(1) * output->size(2);

            for (size_t ibatch = 0; ibatch < infer_batch_size; ibatch++) {
                std::vector<float> featureVector;
                featureVector.resize(output_size);
                auto& job = fetch_jobs[ibatch];
                checkCudaRuntime(cudaMemcpyAsync(
                    featureVector.data(), output->gpu<float>(ibatch),
                    output_size * sizeof(float), cudaMemcpyDeviceToHost, stream_));
                std::vector<std::vector<float>>().swap(featureVectors_);
                featureVectors_.clear();
                featureVectors_.emplace_back(std::move(featureVector));

                std::vector<int>                            predict_shape = { 1, output->size(1), output->size(2) };
                std::vector<std::pair<std::string, double>> rec_conf_res;
                std::vector<float>                          character_ratio;
                //std::vector<cv::Rect>                       char_rect;
                std::vector<std::vector<cv::Point>>         ocr_single_det;
                std::vector<int>                            ocr_char_index;

                int                                         argmax_idx;
                int                                         last_index = 0;
                float                                       score      = 0.f;
                int                                         count      = 0;
                float                                       max_value  = 0.0f;
                float                                       ratio      = float(job.input.rows) / float(input_height_);
                int                                         m          = 0;
                std::string                                 final_ret  = "";
                float                                       src_ratio  = float(job.input.cols) / float(job.input.rows);
                for (int n = 0; n < predict_shape[1]; n++) {
                    argmax_idx = int(OCR::utility::argmax(
                        &featureVectors_[0]
                                        [(m * predict_shape[1] + n) * predict_shape[2]],
                        &featureVectors_[0][(m * predict_shape[1] + n + 1) * predict_shape[2]]));
                    max_value  = float(*std::max_element(
                        &featureVectors_[0]
                                        [(m * predict_shape[1] + n) * predict_shape[2]],
                        &featureVectors_[0][(m * predict_shape[1] + n + 1) * predict_shape[2]]));

                    // label_list_超限判断
                    if (argmax_idx > 0 && (!(n > 0 && argmax_idx == last_index)) && argmax_idx < label_list_.size()) {
                        ocr_char_index.push_back(n);
                        if (float(input_height_) * src_ratio > float(input_width_))
                            ratio = float(job.input.cols) / float(input_width_);
                        character_ratio.push_back(float(n - _shif_ratio) * float(input_width_) / float(output->size(1)) * ratio);
                        score += max_value;
                        count += 1;
                        // ocr结果拆分成单个字符
                        if (bEnableSingleChar_) {
                            rec_conf_res.push_back(std::make_pair(label_list_[argmax_idx - 1], max_value));
                        }
                        final_ret = final_ret + label_list_[argmax_idx - 1];
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
                        //cv::rectangle(job.input, cv::Point(character_ratio[i], 3), cv::Point(character_ratio[i] + mean_count, job.input.rows - 4), cv::Scalar(0, 255, 0));
                    }
                    else {
                        singlePoint.push_back(cv::Point(character_ratio[i], 3));
                        singlePoint.push_back(cv::Point(character_ratio[i + 1], job.input.rows - 4));
                        //cv::rectangle(job.input, cv::Point(character_ratio[i], 3), cv::Point(character_ratio[i + 1], job.input.rows - 4), cv::Scalar(0, 255, 0));
                    }
                    ocr_single_det.push_back(singlePoint);
                    mean_count = (character_ratio[i + 1] - character_ratio[i]);
                }
                score /= count;
                post_time_cost.stop();
                auto& ocr_rec_ret = job.output;

                Box box(score, final_ret);
                if (bEnableSingleChar_ && !rec_conf_res.empty()) {
                    box.rec_conf_res  = rec_conf_res;
                    box.rec_ratio_res = character_ratio;
                    box.ocr_single_pos = ocr_single_det;
                    box.ocr_char_index = ocr_char_index;
                }

                ocr_rec_ret.emplace_back(box);
                ocr_rec_ret.pre_time      = job.preTime.get_cost_time() / infer_batch_size;
                ocr_rec_ret.infer_time    = infer_time_cost.get_cost_time() / infer_batch_size;
                ocr_rec_ret.host_time     = post_time_cost.get_cost_time() / infer_batch_size;
                ocr_rec_ret.total_time    = ocr_rec_ret.pre_time + ocr_rec_ret.infer_time + ocr_rec_ret.host_time;
                model_info_["infer_time"] = ocr_rec_ret.total_time;
                job.pro->set_value(ocr_rec_ret);
            }
            post_time_cost.stop();
            fetch_jobs.clear();
        }
        stream_ = nullptr;
        tensor_allocator_.reset();
        INFO("Engine destroy.");
    }

    virtual bool preprocess(Job& job, const Mat& image) override
    {
        job.preTime.start();
        if (tensor_allocator_ == nullptr) {
            LOG_INFOE("tensor_allocator_ is nullptr");
            return false;
        }

        job.mono_tensor = tensor_allocator_->query();
        if (job.mono_tensor == nullptr) {
            LOG_INFOE("Tensor allocator query failed.");
            return false;
        }
        job.input         = image;
        int     batch_num = 1;
        cv::Mat resize_img;
        OCR::utility::crnn_resize_img(image, resize_img, input_width_ * 1.0 / input_height_, this->rec_image_shape_);
        //job.input = resize_img.clone();
        OCR::utility::normalize(&resize_img, this->mean_, this->scale_, true);
        std::vector<cv::Mat> norm_img_batch;
        norm_img_batch.push_back(resize_img);

        int                    data_size = norm_img_batch.size() * 3 * input_width_ * input_height_;
        std::shared_ptr<float> inBlob(new float[3 * input_width_ * input_height_], [](float* s) { delete[] s; });

        OCR::utility::permute_batch(norm_img_batch, inBlob.get());

        CUDATools::AutoDevice auto_device(gpu_);
        auto&                 tensor            = job.mono_tensor->data();
        TRT::CUStream         preprocess_stream = nullptr;
        if (tensor == nullptr) {
            tensor = make_shared<TRT::Tensor>();
            tensor->set_workspace(make_shared<TRT::MixMemory>());
            preprocess_stream = stream_;
            tensor->set_stream(preprocess_stream, false);
            tensor->resize(batch_num, 3, input_height_, input_width_);
        }
        // ���������Ĵ�С
        tensor->resize(batch_num, 3, input_height_, input_width_);
        // ����cpu�������ڴ�
        tensor->get_data()->cpu(data_size * sizeof(float));
        memcpy(tensor->cpu(), inBlob.get(), data_size * sizeof(float));
        tensor->to_gpu();
        job.preTime.stop();
        return true;
    }

    virtual vector<shared_future<BoxArray>>
    commits(const vector<Mat>& images) override
    {
        return ControllerImpl::commits(images);
    }

    virtual std::shared_future<BoxArray> commit(const Mat& image) override
    {
        return ControllerImpl::commit(image);
    }

    virtual json infer_info() override { return model_info_; }

private:
    bool use_multi_preprocess_stream_ = false;

    bool                            bEnableSingleChar_    = true;
    float                           confidence_threshold_ = 0;
    float                           nms_threshold_        = 0;
    float                           _shif_ratio           = 1.5f;
    int                             gpu_                  = 0;
    int                             max_objects_          = 1024;
    int                             input_width_          = 0;
    int                             input_height_         = 0;
    int                             output_width_         = 0;
    int                             output_height_        = 0;
    int                             rec_img_h_            = 48;
    int                             rec_img_w_            = 640;
    std::vector<std::vector<float>> featureVectors_;
    std::vector<std::string>        label_list_;
    std::vector<int>                rec_image_shape_ = { 3, rec_img_h_, rec_img_w_ };
    std::vector<float>              mean_            = { 0.5f, 0.5f, 0.5f };
    std::vector<float>              scale_           = { 1 / 0.5f, 1 / 0.5f, 1 / 0.5f };
    TRT::CUStream                   stream_          = nullptr;
    json                            model_info_;
};

shared_ptr<Algo::Infer> create_infer(const string& engine_file, int gpuid, float confidence_threshold, std::string label_file)
{
    shared_ptr<InferImpl> instance(new InferImpl());
    if (!instance->startup(engine_file, gpuid, confidence_threshold, label_file)) {
        instance.reset();
    }
    return instance;
}
}  // namespace rec
};  // namespace OCR

#endif  // USE_TRT