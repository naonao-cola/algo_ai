#if USE_TRT

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <queue>

#include "../../../include/private/airuntime/logger.h"
#include "../../../include/private/trt/trt_common/cuda-tools.hpp"
#include "../../../include/private/trt/trt_common/monopoly_allocator.hpp"
#include "../../../include/private/trt/trt_common/time_cost.h"
#include "../../../include/private/trt/trt_common/trt_infer.hpp"
#include "../../../include/private/trt/trt_common/trt_infer_schedule.hpp"
#include "../../../include/public/AIRuntimeUtils.h"

// #include "algo/algo_interface.h"
#include "../../../include/private/trt/trt_app_ocr/cls.hpp"
#include "../../../include/private/trt/trt_app_ocr/ocr_utility.h"

namespace OCR {
namespace cls {
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

    virtual bool startup(const string& file, int gpuid, float confidence_threshold)
    {
        // confidence_threshold_ = confidence_threshold;
        return ControllerImpl::startup(make_tuple(file, gpuid));
    }

    virtual bool set_param(const json& config) override
    {
        // confidence_threshold_ = get_param<float>(config, "confidence_threshold",
        // confidence_threshold_); nms_threshold_ = get_param<float>(config,
        // "nms_threshold", nms_threshold_); max_objects_ = get_param<float>(config,
        // "max_objects", max_objects_);
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
        // ע�������������
        auto input  = engine->tensor("x");
        auto output = engine->tensor("softmax_0.tmp_0");

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
            //输出结果是，1*2
            int output_size = output->size(0) * output->size(1);

            for (size_t ibatch = 0; ibatch < infer_batch_size; ibatch++) {
                std::shared_ptr<float> outBlob(new float[output_size], [](float* s) { delete[] s; });
                checkCudaRuntime(cudaMemcpyAsync(
                    outBlob.get(), output->gpu<float>(ibatch),
                    output_size * sizeof(float), cudaMemcpyDeviceToHost, stream_));
                int   label = int(OCR::utility::argmax(
                    &outBlob.get()[ibatch * output->size(1)],
                    &outBlob.get()[(ibatch + 1) * output->size(1)]));
                float score = float(
                    *std::max_element(&outBlob.get()[ibatch * output->size(1)], &outBlob.get()[(ibatch + 1) * output->size(1)]));
                auto& job         = fetch_jobs[ibatch];
                auto& ocr_rec_ret = job.output;
                // �����ֵ
                Box box;
                box.class_label = label;
                box.confidence  = score;

                ocr_rec_ret.emplace_back(box);
                ocr_rec_ret.pre_time = job.preTime.get_cost_time() / infer_batch_size;
                ocr_rec_ret.infer_time =
                    infer_time_cost.get_cost_time() / infer_batch_size;
                ocr_rec_ret.host_time =
                    post_time_cost.get_cost_time() / infer_batch_size;
                ocr_rec_ret.total_time = ocr_rec_ret.pre_time + ocr_rec_ret.infer_time +
                                         ocr_rec_ret.host_time;
                model_info_["infer_time"] = ocr_rec_ret.total_time;
                job.pro->set_value(ocr_rec_ret);
            }
            // �õ�gpu�Y����cpu
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

        int     batch_num = 1;
        cv::Mat resize_img;
        OCR::utility::cls_resize_img(image, resize_img, cls_image_shape_);
        OCR::utility::normalize(&resize_img, this->mean_, this->scale_, this->is_scale_);
        std::vector<cv::Mat> norm_img_batch;
        norm_img_batch.push_back(resize_img);

        // ���ݴ�СΪ 3 * 48 * 192
        int                    data_size = batch_num * 3 * input_width_ * input_height_;
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
    int           gpu_    = 0;
    TRT::CUStream stream_ = nullptr;
    json          model_info_;
    // ����
    std::vector<int> cls_image_shape_ = { 3, 48, 192 };
    bool             is_scale_        = true;
    int              input_width_     = 0;
    int              input_height_    = 0;

    std::vector<float> mean_  = { 0.5f, 0.5f, 0.5f };
    std::vector<float> scale_ = { 1 / 0.5f, 1 / 0.5f, 1 / 0.5f };
};

std::shared_ptr<Algo::Infer> create_infer(const std::string& engine_file, int gpuid, float confidence_threshold)
{
    std::shared_ptr<InferImpl> instance(new InferImpl());
    if (!instance->startup(engine_file, gpuid, confidence_threshold)) {
        instance.reset();
    }
    return instance;
}
}  // namespace cls
};  // namespace OCR


#endif //USE_TRT