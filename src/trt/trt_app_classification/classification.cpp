#if USE_TRT

#include "opencv2/opencv.hpp"
#include <string>

#include "../../../include/private/airuntime/logger.h"
#include "../../../include/public/AIRuntimeUtils.h"

#include "../../../include/private/trt/trt_common/cuda-tools.hpp"
#include "../../../include/private/trt/trt_common/monopoly_allocator.hpp"
#include "../../../include/private/trt/trt_common/time_cost.h"
#include "../../../include/private/trt/trt_common/trt_infer.hpp"
#include "../../../include/private/trt/trt_common/trt_infer_schedule.hpp"

#include "../../../include/private/trt/trt_app_classification/classification.hpp"

namespace Classification {

using namespace cv;
using BoxArray = Algo::BoxArray;
using Box      = Algo::Box;
using namespace std;
using namespace Algo;

using ControllerImpl =
    InferController<Mat, BoxArray, std::tuple<std::string, int>>;

class InferImpl : public Infer, public ControllerImpl
{
public:
    virtual ~InferImpl() { stop(); }

    virtual bool startup(const string& file, int gpuid, float confidence_threshold, std::vector<std::vector<int>> dims)
    {
        confidence_threshold_ = confidence_threshold;
        dims_                 = dims;
        return ControllerImpl::startup(make_tuple(file, gpuid));
    }

    virtual bool set_param(const json& config) override
    {
        confidence_threshold_ =
            get_param<float>(config, "confidence_threshold", confidence_threshold_);
        return true;
    }

    virtual void worker(promise<bool>& result) override
    {
        string file  = get<0>(start_param_);
        int    gpuid = get<1>(start_param_);
        TRT::set_device(gpuid);
        auto engine = TRT::load_infer(file, dims_);
        if (engine == nullptr) {
            LOG_INFOE("Engine {} load failed!", file.c_str());
            result.set_value(false);
            return;
        }

        engine->print();

        TRT::Tensor output_array_device(TRT::DataType::Float);

        //int         max_batch_size = engine->get_max_batch_size();
        int max_batch_size = 0;
        if (dims_.size() > 0) {
            max_batch_size = dims_[0][0];
        }
        else {
            // ´Ëº¯ÊýÓÀÔ¶·µ»Ø0
            max_batch_size = engine->get_max_batch_size();
        }



        auto        input          = engine->input();
        auto        output         = engine->output();
        int         num_classes    = output->size(1);
        input_width_               = input->size(3);
        input_height_              = input->size(2);
        model_info_["memory_size"] = engine->get_device_memory_size() >> 20;
        model_info_["dims"] = {input->size(0), input->size(1), input->size(2), input->size(3)};

        //tensor_allocator_ = make_shared<MonopolyAllocator<TRT::Tensor>>(max_batch_size * 2);
        if (dims_.size() > 0) {
            tensor_allocator_ = make_shared<MonopolyAllocator<TRT::Tensor>>(dims_[0][0]);
        }
        else {
            tensor_allocator_ = make_shared<MonopolyAllocator<TRT::Tensor>>(max_batch_size * 2);
        }


        stream_ = engine->get_stream();
        gpu_    = gpuid;
        result.set_value(true);

        input->resize_single_dim(0, max_batch_size).to_gpu();
        output_array_device.resize(max_batch_size, 1 + num_classes).to_gpu();
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

            output_array_device.to_gpu(false);
            TRT::TimeCost post_time_cost;
            post_time_cost.start();
            for (size_t ibatch = 0; ibatch < infer_batch_size; ibatch++) {
                auto&  job               = fetch_jobs[ibatch];
                auto&  image_based_boxes = job.output;
                float* prob              = output->cpu<float>(ibatch);
                int    predict_label     = std::max_element(prob, prob + num_classes) - prob;
                float  confidence        = 1 / (1 + exp(-prob[predict_label]));
                Box    box(0, 0, 0, 0, confidence, predict_label);
                image_based_boxes.push_back(box);

                image_based_boxes.pre_time =
                    job.preTime.get_cost_time() / infer_batch_size;
                image_based_boxes.infer_time =
                    infer_time_cost.get_cost_time() / infer_batch_size;
                image_based_boxes.host_time =
                    post_time_cost.get_cost_time() / infer_batch_size;
                image_based_boxes.total_time = image_based_boxes.pre_time +
                                               image_based_boxes.infer_time +
                                               image_based_boxes.host_time;
                model_info_["infer_time"] = image_based_boxes.total_time;
                job.pro->set_value(image_based_boxes);
            }
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

        if (image.empty()) {
            LOG_INFOE("Image is empty");
            return false;
        }
        cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

        job.mono_tensor = tensor_allocator_->query();
        if (job.mono_tensor == nullptr) {
            LOG_INFOE("Tensor allocator query failed.");
            return false;
        }

        CUDATools::AutoDevice auto_device(gpu_);
        auto&                 tensor = job.mono_tensor->data();

        TRT::CUStream preprocess_stream = nullptr;
        TRT::TimeCost pre_time_cost;
        pre_time_cost.start();
        if (tensor == nullptr) {
            // not init
            tensor = make_shared<TRT::Tensor>();
            tensor->set_workspace(make_shared<TRT::MixMemory>());
            preprocess_stream = stream_;
            // owner = false, tensor ignored the stream
            tensor->set_stream(preprocess_stream, false);
            tensor->resize(1, 3, input_height_, input_width_);
        }
        preprocess_stream = tensor->get_stream();
        tensor->resize(1, 3, input_height_, input_width_);
        // float mean[] = { 0.485, 0.456, 0.406 };
        // float std[] = { 0.229, 0.224, 0.225 };
        float mean[] = { 0., 0., 0. };
        float std[]  = { 1, 1, 1 };
        tensor->set_norm_mat(0, image, mean, std);
        tensor->to_gpu();
        job.preTime.stop();
        return true;
    }

    virtual json infer_info() override { return model_info_; }

    virtual vector<shared_future<BoxArray>>
    commits(const vector<Mat>& images) override
    {
        return ControllerImpl::commits(images);
    }

    virtual std::shared_future<BoxArray> commit(const Mat& image) override
    {
        return ControllerImpl::commit(image);
    }

private:
    float         confidence_threshold_;
    json          model_info_;
    int           input_width_;
    int           input_height_;
    int           gpu_{ 0 };
    TRT::CUStream stream_ = nullptr;
    std::vector<std::vector<int>> dims_   = {};
};

std::shared_ptr<Algo::Infer> create_infer(const std::string& engine_file, int gpuid, float confidence_threshold, std::vector<std::vector<int>> dims)
{
    shared_ptr<InferImpl> instance(new InferImpl());
    if (!instance->startup(engine_file, gpuid, confidence_threshold, dims)) {
        instance.reset();
    }
    return instance;
}

}  // end namespace Classification

#endif //USE_TRT