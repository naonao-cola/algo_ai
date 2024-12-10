#if USE_TRT

#include "opencv2/opencv.hpp"

#include "../../../include/private/airuntime/logger.h"
#include "../../../include/public/AIRuntimeUtils.h"

#include "../../../include/private/trt/trt_app_segmentation/unet.hpp"
#include "../../../include/private/trt/trt_common/cuda-tools.hpp"
#include "../../../include/private/trt/trt_common/trt_infer.hpp"
#include "../../../include/private/trt/trt_common/trt_infer_schedule.hpp"
#include "../../../include/private/trt/trt_cuda/preprocess_kernel.cuh"

namespace UNet {

using namespace cv;
using namespace std;
using BoxArray = Algo::BoxArray;
using Box      = Algo::Box;
using Infer    = Algo::Infer;

struct AffineMatrix
{
    float    i2d[6];  // image to dst(network), 2x3 matrix
    float    d2i[6];  // dst to image, 2x3 matrix
    cv::Size isize;
    cv::Size dsize;

    void compute(const cv::Size& from, const cv::Size& to)
    {
        isize         = from;
        dsize         = to;
        float scale_x = to.width / (float)from.width;
        float scale_y = to.height / (float)from.height;
        float scale   = std::min(scale_x, scale_y);
        /*
                + scale * 0.5 - 0.5
           ����Ҫԭ����ʹ�����ĸ��Ӷ��룬�²��������ԣ������ϲ���ʱ�ͱȽ�����
            �ο���https://www.iteye.com/blog/handspeaker-1545126
        */
        i2d[0] = scale;
        i2d[1] = 0;
        i2d[2] = -scale * from.width * 0.5 + to.width * 0.5 + scale * 0.5 - 0.5;
        i2d[3] = 0;
        i2d[4] = scale;
        i2d[5] = -scale * from.height * 0.5 + to.height * 0.5 + scale * 0.5 - 0.5;

        cv::Mat m2x3_i2d(2, 3, CV_32F, i2d);
        cv::Mat m2x3_d2i(2, 3, CV_32F, d2i);
        cv::invertAffineTransform(m2x3_i2d, m2x3_d2i);
    }

    cv::Mat i2d_mat() { return cv::Mat(2, 3, CV_32F, i2d); }

    cv::Mat d2i_mat() { return cv::Mat(2, 3, CV_32F, d2i); }
};

AffineMatrix image_to_tensor_unet(const cv::Mat& image, shared_ptr<TRT::Tensor>& tensor, int ibatch)
{
    CUDAKernel::Norm normalize = CUDAKernel::Norm::alpha_beta(
        1 / 255.0f, 0.0f, CUDAKernel::ChannelType::Invert);
    Size         input_size(tensor->size(3), tensor->size(2));
    AffineMatrix affine;
    affine.compute(image.size(), input_size);

    size_t   size_image           = image.cols * image.rows * 3;
    size_t   size_matrix          = iLogger::upbound(sizeof(affine.d2i), 32);
    auto     workspace            = tensor->get_workspace();
    uint8_t* gpu_workspace        = (uint8_t*)workspace->gpu(size_matrix + size_image);
    float*   affine_matrix_device = (float*)gpu_workspace;
    uint8_t* image_device         = size_matrix + gpu_workspace;

    uint8_t* cpu_workspace      = (uint8_t*)workspace->cpu(size_matrix + size_image);
    float*   affine_matrix_host = (float*)cpu_workspace;
    uint8_t* image_host         = size_matrix + cpu_workspace;
    auto     stream             = tensor->get_stream();

    memcpy(image_host, image.data, size_image);
    memcpy(affine_matrix_host, affine.d2i, sizeof(affine.d2i));
    checkCudaRuntime(cudaMemcpyAsync(image_device, image_host, size_image, cudaMemcpyHostToDevice, stream));
    checkCudaRuntime(cudaMemcpyAsync(affine_matrix_device, affine_matrix_host, sizeof(affine.d2i), cudaMemcpyHostToDevice, stream));

    CUDAKernel::warp_affine_bilinear_and_normalize_plane(
        image_device, image.cols * 3, image.cols, image.rows,
        tensor->gpu<float>(ibatch), input_size.width, input_size.height,
        affine_matrix_device, 128, normalize, stream);
    // return affine.d2i_mat().clone();
    return affine;
}

using ControllerImpl =
    InferController<Mat,                 // input
                    BoxArray,            // output <output_prob, output_index>
                    tuple<string, int>,  // start param
                    AffineMatrix         // additional
                    >;
class InferImpl : public Infer, public ControllerImpl
{
public:
    virtual ~InferImpl() { stop(); }

    virtual bool startup(const string& file, int gpuid, float confidence_threshold)
    {
        confidence_threshold_ = confidence_threshold;
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
        auto engine = TRT::load_infer(file);
        if (engine == nullptr) {
            LOG_INFOE("Engine {} load failed!", file.c_str());
            result.set_value(false);
            return;
        }

        engine->print();

        TRT::Tensor output_array_device(TRT::DataType::Float);
        int         max_batch_size = engine->get_max_batch_size();
        auto        input          = engine->input();
        auto        output         = engine->output();
        int         num_classes    = output->size(1);
        input_width_               = input->size(3);
        input_height_              = input->size(2);
        model_info_["memory_size"] = engine->get_device_memory_size() >> 20;
        model_info_["dims"] = {input->size(0), input->size(1), input->size(2), input->size(3)};

        tensor_allocator_ =
            make_shared<MonopolyAllocator<TRT::Tensor>>(max_batch_size * 2);
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
                auto& job               = fetch_jobs[ibatch];
                auto& image_based_boxes = job.output;

                cv::Mat prob, iclass;
                tie(prob, iclass) = post_process(output, ibatch);
                cv::warpAffine(prob, prob, job.additional.d2i_mat(), job.additional.isize, cv::INTER_LINEAR);
                cv::warpAffine(iclass, iclass, job.additional.d2i_mat(), job.additional.isize, cv::INTER_NEAREST);
                image_based_boxes.index_mat = iclass;
                image_based_boxes.prob_mat  = prob;
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
        job.additional = image_to_tensor_unet(image, tensor, 0);
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
    int           gpu_;
    TRT::CUStream stream_ = nullptr;

private:
    tuple<cv::Mat, cv::Mat> post_process(shared_ptr<TRT::Tensor>& tensor, int ibatch)
    {
        cv::Mat output_prob(tensor->size(1), tensor->size(2), CV_32F);
        cv::Mat output_index(tensor->size(1), tensor->size(2), CV_8U);

        int      num_class = tensor->size(3);
        float*   pnet      = tensor->cpu<float>(ibatch);
        float*   prob      = output_prob.ptr<float>(0);
        uint8_t* pidx      = output_index.ptr<uint8_t>(0);

        for (int k = 0; k < output_prob.cols * output_prob.rows;
             ++k, pnet += num_class, ++prob, ++pidx) {
            int ic = std::max_element(pnet, pnet + num_class) - pnet;
            *prob  = pnet[ic];
            *pidx  = ic;
        }
        return make_tuple(output_prob, output_index);
    }
};

shared_ptr<Algo::Infer> create_infer(const string& engine_file, int gpuid, float confidence_threshold)
{
    shared_ptr<InferImpl> instance(new InferImpl());
    if (!instance->startup(engine_file, gpuid, confidence_threshold)) {
        instance.reset();
    }
    return instance;
}

}  // end namespace UNet


#endif //USE_TRT