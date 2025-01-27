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
#    include "../../../include/private/trt/trt_app_deim/deim.hpp"
#    include "../../../include/private/trt/trt_app_ocr/ocr_utility.h"

namespace deim {

/*辅助函数前向声明*/
void resize_unscale(const cv::Mat& mat, cv::Mat& mat_rs, int target_height, int target_width);
void normalize_inplace(cv::Mat& mat_inplace, float mean, float scale);

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

    virtual bool startup(const string& file, int gpuid, float confidence_threshold, float nms_threshold, int max_objects, std::vector<std::vector<int>> dims)
    {
        confidence_threshold_ = confidence_threshold;
        nms_threshold_        = nms_threshold;
        max_objects_          = max_objects;
        dims_                 = dims;
        return ControllerImpl::startup(make_tuple(file, gpuid));
    }

    virtual bool set_param(const json& config) override
    {
        confidence_threshold_ = get_param<float>(config, "confidence_threshold", confidence_threshold_);
        nms_threshold_        = get_param<float>(config, "nms_threshold", nms_threshold_);
        max_objects_          = get_param<float>(config, "max_objects", max_objects_);
        return true;
    }

    virtual void worker(promise<bool>& result) override
    {
        string file  = get<0>(start_param_);
        int    gpuid = get<1>(start_param_);
        TRT::set_device(gpuid);
        auto engine = TRT::load_infer(file, dims_);
        if (engine == nullptr) {
            LOG_INFOE("Engine {} load failed", file.c_str());
            result.set_value(false);
            return;
        }

        engine->print();
        int max_batch_size = 0;
        if (dims_.size() > 0) {
            max_batch_size = dims_[0][0];
        }
        else {
            max_batch_size = engine->get_max_batch_size();
        }
        auto input_0  = engine->tensor(engine->get_input_name(0));
        auto input_1  = engine->tensor(engine->get_input_name(1));
        auto output_0 = engine->tensor(engine->get_output_name(0));
        auto output_1 = engine->tensor(engine->get_output_name(1));
        auto output_2 = engine->tensor(engine->get_output_name(2));

        input_height_              = input_0->size(2);
        input_width_               = input_0->size(3);
        model_info_["memory_size"] = engine->get_device_memory_size() >> 20;
        model_info_["dims"]        = { input_0->size(0), input_0->size(1), input_0->size(2), input_0->size(3) };

        if (dims_.size() > 0) {
            tensor_allocator_ = make_shared<MonopolyAllocator<TRT::Tensor>>(dims_[0][0]);
        }
        else {
            tensor_allocator_ = make_shared<MonopolyAllocator<TRT::Tensor>>(max_batch_size * 2);
        }

        stream_ = engine->get_stream();
        gpu_    = gpuid;
        result.set_value(true);
        input_0->resize_single_dim(0, max_batch_size).to_gpu();
        vector<Job> fetch_jobs;

        int intput_size_0 = input_0->size(1) * input_0->size(2) * input_0->size(3);
        int intput_size_1 = input_1->size(1);
        while (get_jobs_and_wait(fetch_jobs, max_batch_size)) {
            int infer_batch_size = fetch_jobs.size();
            input_0->resize_single_dim(0, infer_batch_size);
            input_1->resize_single_dim(0, infer_batch_size);
            for (int ibatch = 0; ibatch < infer_batch_size; ++ibatch) {
                auto& job  = fetch_jobs[ibatch];
                auto& mono = job.mono_tensor->data();
                if (mono->get_stream() != stream_) {
                    checkCudaRuntime(cudaStreamSynchronize(mono->get_stream()));
                }
                input_0->copy_from_gpu(input_0->offset(ibatch), (float*)mono->gpu<float>(), intput_size_0);
                input_1->copy_from_gpu(input_1->offset(ibatch), (__int32*)(mono->gpu<float>() + intput_size_0), intput_size_1);
                job.mono_tensor->release();
            }
            TRT::TimeCost infer_time_cost;

            infer_time_cost.start();
            engine->forward();
            infer_time_cost.stop();
            LOG_INFOE("deim forward {} ms", infer_time_cost.get_cost_time());

            TRT::TimeCost post_time_cost;
            post_time_cost.start();

            int output_size_0 = output_0->size(0) * output_0->size(1);
            int output_size_1 = output_1->size(0) * output_1->size(1);
            int output_size_2 = output_2->size(0) * output_2->size(1) * output_2->size(2);

            for (size_t ibatch = 0; ibatch < infer_batch_size; ibatch++) {
                std::vector<float>   scores(output_size_0);
                std::vector<__int32> labels(output_size_1);
                std::vector<float>   boxes(output_size_2);
                checkCudaRuntime(cudaMemcpyAsync(scores.data(), output_0->gpu<float>(ibatch), output_size_0 * sizeof(float), cudaMemcpyDeviceToHost, stream_));
                checkCudaRuntime(cudaMemcpyAsync(labels.data(), output_1->gpu<__int32>(ibatch), output_size_1 * sizeof(__int32), cudaMemcpyDeviceToHost, stream_));
                checkCudaRuntime(cudaMemcpyAsync(boxes.data(), output_2->gpu<float>(ibatch), output_size_2 * sizeof(float), cudaMemcpyDeviceToHost, stream_));
                auto& job               = fetch_jobs[ibatch];
                auto& image_based_boxes = job.output;
                for (int m = 0; m < scores.size(); m++) {
                    if (scores[m] >= confidence_threshold_) {
                        Box box;
                        box.left        = boxes[4 * m + 0];
                        box.top         = boxes[4 * m + 1];
                        box.right       = boxes[4 * m + 2];
                        box.bottom      = boxes[4 * m + 3];
                        box.confidence  = scores[m];
                        box.class_label = labels[m];
                        image_based_boxes.emplace_back(box);
                    }
                }
                image_based_boxes.pre_time   = job.preTime.get_cost_time() / infer_batch_size;
                image_based_boxes.infer_time = infer_time_cost.get_cost_time() / infer_batch_size;
                image_based_boxes.host_time  = post_time_cost.get_cost_time() / infer_batch_size;
                image_based_boxes.total_time = image_based_boxes.pre_time + image_based_boxes.infer_time + image_based_boxes.host_time;
                model_info_["infer_time"]    = image_based_boxes.total_time;
                job.pro->set_value(image_based_boxes);
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
        origin_img_c_     = image.channels();
        origin_img_h_     = image.rows;
        origin_img_w_     = image.cols;
        int     batch_num = 1;
        cv::Mat canvas;
        cv::cvtColor(image, canvas, cv::COLOR_BGR2RGB);
        cv::Mat mat_rs;
        resize_unscale(canvas, mat_rs, input_width_, input_height_);
        normalize_inplace(mat_rs, mean_, scale_);
        std::vector<cv::Mat> norm_img_batch;
        norm_img_batch.push_back(mat_rs);
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
        tensor->get_data()->gpu(data_size * sizeof(float) + 2 * sizeof(__int32));
        tensor->get_data()->cpu(data_size * sizeof(float) + 2 * sizeof(__int32));
        bool    set_ret            = tensor->set_bytes(data_size * sizeof(float) + 2 * sizeof(__int32));
        __int32 origin_img_size[2] = { __int32(origin_img_w_), __int32(origin_img_h_) };
        memcpy(tensor->cpu(), inBlob.get(), data_size * sizeof(float));
        memcpy((__int32*)(tensor->cpu<float>() + data_size), (__int32*)origin_img_size, 2 * sizeof(__int32));
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
    int                           gpu_                         = 0;
    float                         confidence_threshold_        = 0;
    float                         nms_threshold_               = 0;
    int                           max_objects_                 = 1024;
    TRT::CUStream                 stream_                      = nullptr;
    bool                          use_multi_preprocess_stream_ = false;
    json                          model_info_;
    int                           origin_img_c_  = 0;
    int                           origin_img_h_  = 0;
    int                           origin_img_w_  = 0;
    int                           input_width_   = 640;
    int                           input_height_  = 640;
    int                           output_width_  = 0;
    int                           output_height_ = 0;
    float                         mean_          = 0.0f;
    float                         scale_         = 1 / 255.f;
    const unsigned int            max_nms_       = 30000;
    std::vector<std::vector<int>> dims_          = {};
};

void resize_unscale(const cv::Mat& mat, cv::Mat& mat_rs, int target_height, int target_width)
{
    if (mat.empty())
        return;
    // auto img_height = mat.rows;
    // auto img_width  = mat.cols;
    // mat_rs = cv::Mat(target_height, target_width, CV_8UC3, cv::Scalar(0, 0, 0));
    // float w_r = (float)target_width / (float)img_width;
    // float h_r = (float)target_height / (float)img_height;
    // float r   = std::min(w_r, h_r);
    // auto new_unpad_w = static_cast<int>((float)img_width * r);   // floor
    // auto new_unpad_h = static_cast<int>((float)img_height * r);  // floor
    // cv::Mat new_unpad_mat;
    // cv::resize(mat, new_unpad_mat, cv::Size(new_unpad_w, new_unpad_h));
    // int  pad_w       = target_width - new_unpad_w;               // >=0
    // int  pad_h       = target_height - new_unpad_h;              // >=0
    // int dw = pad_w / 2;
    // int dh = pad_h / 2;
    // new_unpad_mat.copyTo(mat_rs(cv::Rect(dw, dh, new_unpad_w, new_unpad_h)));

    cv::resize(mat, mat_rs, cv::Size(target_width, target_height));
}
void normalize_inplace(cv::Mat& mat_inplace, float mean, float scale)
{
    if (mat_inplace.type() != CV_32FC3)
        mat_inplace.convertTo(mat_inplace, CV_32FC3);
    mat_inplace = (mat_inplace - mean) * scale;
}

shared_ptr<Algo::Infer> create_infer(const std::string& engine_file, int gpuid, float confidence_threshold,  int max_objects, std::vector<std::vector<int>> dims)
{
    shared_ptr<InferImpl> instance(new InferImpl());
    if (!instance->startup(engine_file, gpuid, confidence_threshold, 0, max_objects, dims)) {
        instance.reset();
    }
    return instance;
}
};  // namespace deim

#endif  // USE_TRT