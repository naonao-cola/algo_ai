#if USE_TRT

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <queue>

#include "../../../include/private/airuntime/logger.h"
#include "../../../include/public/AIRuntimeUtils.h"

#include "../../../include/private/trt/trt_common/cuda-tools.hpp"
#include "../../../include/private/trt/trt_common/monopoly_allocator.hpp"
#include "../../../include/private/trt/trt_common/time_cost.h"
#include "../../../include/private/trt/trt_common/trt_infer.hpp"
#include "../../../include/private/trt/trt_common/trt_infer_schedule.hpp"

// #include "algo/algo_interface.h"
#include "../../../include/private/trt/trt_app_ocr/ocr_utility.h"
#include "../../../include/private/trt/trt_app_msae/msae.hpp"

namespace msae {



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

    virtual bool startup(const string& file, int gpuid, float confidence_threshold, float nms_threshold, int max_objects)
    {
        confidence_threshold_ = confidence_threshold;
        nms_threshold_        = nms_threshold;
        max_objects_          = max_objects;
        return ControllerImpl::startup(make_tuple(file, gpuid));
    }

    virtual bool set_param(const json& config) override
    {
        confidence_threshold_ = get_param<float>(config, "confidence_threshold", confidence_threshold_);
        nms_threshold_ = get_param<float>(config, "nms_threshold", nms_threshold_);
        max_objects_   = get_param<float>(config, "max_objects", max_objects_);
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

        int  max_batch_size = engine->get_max_batch_size();
        auto input          = engine->tensor(engine->get_input_name(0));//"inputs"
        auto output         = engine->tensor(engine->get_output_name(0));//"outputs"

        input_height_ = input->size(2);
        input_width_  = input->size(3);
        model_info_["memory_size"] = engine->get_device_memory_size() >> 20;
        model_info_["dims"] = {input->size(0), input->size(1), input->size(2), input->size(3)};

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
                checkCudaRuntime(cudaMemcpyAsync(featureVector.data(), output->gpu<float>(ibatch), output_size * sizeof(float), cudaMemcpyDeviceToHost, stream_));
               
                //边缘弱化，看参数是否需要修改
                const float border_ratio = 0.05;
                const float border_coef  = 1.618;
                int out_h = output->size(1);
                int out_w = output->size(2);
                int   border_w    = round(out_w * border_ratio);
                int   border_h    = round(out_h * border_ratio);
                float       coef_factor  = std::sqrt(border_coef);
                for (int row = 0; row < out_h; row++) 
                {
                    for (int col = 0; col < out_w; col++) 
                    {
                        float val = featureVector[row * out_w + col];
                        if (row < border_h || row > out_h - border_h - 1 || col < border_w || col > out_w - border_w - 1) 
                        {
                            val /= coef_factor;
                        }
                        if (row < 2 * border_h || row > (out_h - 2 * border_h - 1) || col < 2 * border_w || col > (out_w - 2 * border_w - 1)) 
                        {
                            val /= coef_factor;
                        }
                        featureVector[row * out_w + col] = val;
                    }
                }
                //数组转mat
                cv::Mat outmap(out_h, out_w, CV_32FC1, cv::Scalar(0));
                memcpy(outmap.data, featureVector.data(), featureVector.size() * sizeof(float));
                double                   max_value, min_value;
                cv::minMaxLoc(outmap, &min_value, &max_value);
               // float v_min = 0, v_max = 10;
                outmap = (outmap - min_value) / (max_value - min_value);
                outmap.setTo(1, outmap > 1);
                outmap.convertTo(outmap, CV_8UC1, 255);
               
                cv::Mat amap;
                cv::resize(outmap, amap, cv::Size(origin_img_w_, origin_img_h_));
                //
                cv::GaussianBlur(amap, amap, cv::Size(5, 5), 1.0);
                //12/26  hjf
                //  二值化
                float   bin_threshold = 70;
                float   area_threshold = 8000;
                cv::Mat bin_map;
                cv::threshold(amap, bin_map, bin_threshold, 255, cv::THRESH_BINARY);
                cv::medianBlur(bin_map, bin_map, 7);

                std::vector<std::vector<cv::Point>> contours;
                std::vector<cv::Vec4i>              hierarchy;
                cv::findContours(bin_map, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

                auto& job     = fetch_jobs[ibatch];
                BoxArray detected_boxes;

                json all_out = json::array();
                for (auto cnt : contours) {
                    double area = cv::contourArea(cnt);
                    if (area > area_threshold) {
                        cv::Rect  box = cv::boundingRect(cnt);
                        std::vector<std::vector<cv::Point>> draw_cnt{ cnt };
                        cv::Mat mask = cv::Mat::zeros(amap.rows, amap.cols, CV_8UC1);
                        cv::drawContours(mask, draw_cnt,0,255,-1);
                        cv::Scalar avg       = cv::mean(amap, mask);
                        float      area_conf = area / (30000);
                        area_conf            = (area_conf < 1.0) ? area_conf : 1.0;
                        float conf           = (avg[0] / (bin_threshold * 1.8)) * area_conf;
                        conf                 = (conf < 1.0) ? conf : 1.0;

                        //TODO
                        Box msae_box;
                        msae_box.left     = box.x;
                        msae_box.right    = box.x +box.width;
                        msae_box.top      = box.y;
                        msae_box.bottom   = box.y+box.height;
                        msae_box.confidence = conf;
                        msae_box.msae_img = amap.clone();
                        //添加点
                        detected_boxes.emplace_back(msae_box);

                    }
                }
                post_time_cost.stop();
                //返回mat的数据
                detected_boxes.pre_time = job.preTime.get_cost_time() / infer_batch_size;
                detected_boxes.infer_time =infer_time_cost.get_cost_time() / infer_batch_size;
                detected_boxes.host_time = post_time_cost.get_cost_time() / infer_batch_size;
                detected_boxes.total_time = detected_boxes.pre_time + detected_boxes.infer_time +detected_boxes.host_time;
                model_info_["infer_time"] = detected_boxes.total_time;
                job.pro->set_value(detected_boxes);
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

        origin_img_c_ = image.channels();
        origin_img_h_ = image.rows;
        origin_img_w_ = image.cols;

        int     batch_num = 1;
        //先缩放，再转RGB
        cv::Mat resize_img;
        cv::resize(image, resize_img, cv::Size(input_width_, input_height_));
        cv::Mat canvas;
        //cv::cvtColor(resize_img, canvas, cv::COLOR_BGR2RGB);
        //RGB 先不转
        canvas = resize_img.clone();
       //归一化
        std::vector<float> mean{0.0,0.0,0.0};
        std::vector<float> scale{1.0,1.0,1.0};
        OCR::utility::normalize(&canvas, mean, scale, true);

        std::vector<cv::Mat> norm_img_batch;
        norm_img_batch.push_back(canvas);
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
        tensor->resize(batch_num, 3, input_height_, input_width_);
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
    int                             gpu_                         = 0;
    float                           confidence_threshold_        = 0;
    float                           nms_threshold_               = 0;
    int                             max_objects_                 = 1024;
    TRT::CUStream                   stream_                      = nullptr;
    bool                            use_multi_preprocess_stream_ = false;
    json                            model_info_;
    int                             origin_img_c_ = 0;
    int                             origin_img_h_ = 0;
    int                             origin_img_w_ = 0;
    int                      input_width_     = 1420;
    int                      input_height_    = 1000;
    int                      output_width_    = 0;
    int                      output_height_   = 0;
    float       mean_            = 0.5f;
    float       scale_           = 1/255.f;
    const unsigned int       max_nms_         = 30000;
};


shared_ptr<Algo::Infer> create_infer(const std::string& engine_file, int gpuid, float confidence_threshold, float nms_threshold, int max_objects)
{
    shared_ptr<InferImpl> instance(new InferImpl());
     if (!instance->startup(engine_file, gpuid, confidence_threshold, nms_threshold, max_objects)) {
        instance.reset();
    }
    return instance;
}

};  // namespace msae


#endif //USE_TRT