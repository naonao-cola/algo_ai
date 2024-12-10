#if USE_TRT

#include "NvInferLegacyDims.h"
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
#include "../../../include/private/trt/trt_app_ocr/det.hpp"
#include "../../../include/private/trt/trt_app_ocr/ocr_utility.h"

namespace OCR {
namespace det {

using namespace cv;
using namespace std;
using BoxArray = Algo::BoxArray;
using Box      = Algo::Box;
using Infer    = Algo::Infer;
using ControllerImpl =
    InferController<Mat, /* input*/ BoxArray,
                    /*output*/ tuple<string, int> /*start param*/>;

class InferImpl : public Infer, public ControllerImpl
{
public:
    virtual ~InferImpl() { stop(); }

    virtual bool startup(const string& file, int gpuid, int kernel_size, bool use_dilation, bool enable_detmat, float det_db_box_thresh, float det_db_unclip_ratio, int max_side_len)
    {
        kernel_size_         = kernel_size;
        use_dilation_       = use_dilation;
        enable_detmat_       = enable_detmat;
        det_db_box_thresh_   = det_db_box_thresh;
        det_db_unclip_ratio_ = det_db_unclip_ratio;
        max_side_len_        = max_side_len;
        return ControllerImpl::startup(make_tuple(file, gpuid));
    }

    virtual bool set_param(const json& config) override
    {
        det_db_box_thresh_   = get_param<float>(config, "det_db_box_thresh", det_db_box_thresh_);
        det_db_unclip_ratio_ = get_param<float>(config, "det_db_unclip_ratio", det_db_unclip_ratio_);
        enable_detmat_       = get_param<bool>(config, "enableDetMat", enable_detmat_);
        use_dilation_        = get_param<float>(config, "useDilat", use_dilation_);
        max_side_len_        = get_param<int>(config, "max_side_len", max_side_len_);
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
        int  max_batch_size        = engine->get_max_batch_size();
        auto input                 = engine->tensor(engine->get_input_name(0)); //"x"
        auto output                = engine->tensor(engine->get_output_name(0)); //"sigmoid_0.tmp_0"

        // �������
        input_height_ = input->shape(2);
        input_width_  = input->shape(3);
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

            int output_size = output->size(1) * output->size(2) * output->size(3);
            int n2          = output->shape(2);
            int n3          = output->shape(3);
            int n           = n2 * n3;  // output_h * output_w

            for (int ibatch = 0; ibatch < infer_batch_size; ++ibatch) {
                // ÿһ���εĽ��
                std::shared_ptr<float> outBlob(new float[output_size], [](float* s) { delete[] s; });
                checkCudaRuntime(cudaMemcpyAsync(
                    outBlob.get(), output->gpu<float>(ibatch),
                    output_size * sizeof(float), cudaMemcpyDeviceToHost, stream_));

                auto& job               = fetch_jobs[ibatch];
                auto& image_based_boxes = job.output;

                std::vector<float>                         pred(n, 0.0);
                std::vector<unsigned char>                 cbuf(n, ' ');
                std::vector<std::vector<std::vector<int>>> boxes;

                for (int i = 0; i < n; i++) {
                    pred[i] = float(outBlob.get()[i]);
                    cbuf[i] = (unsigned char)((outBlob.get()[i]) * 255);
                }
                cv::Mat cbuf_map(n2, n3, CV_8UC1, (unsigned char*)cbuf.data());
                cv::Mat pred_map(n2, n3, CV_32F, (float*)pred.data());

                const double threshold = this->det_db_thresh_ * 255;
                const double maxvalue  = 255;
                cv::Mat      bit_map;
                cv::threshold(cbuf_map, bit_map, threshold, maxvalue, cv::THRESH_BINARY);
                if (this->use_dilation_) {

                    cv::Mat dila_ele =
                        cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernel_size_, kernel_size_));
                    cv::dilate(bit_map, bit_map, dila_ele);
                }
                boxes = OCR::utility::BoxesFromBitmap(
                    pred_map, bit_map, this->det_db_box_thresh_,
                    this->det_db_unclip_ratio_, this->use_polygon_score_);
                cv::Mat tmp_src =
                    cv::Mat::zeros(cv::Size(origin_img_w_, origin_img_h_), CV_8UC3);
                boxes = OCR::utility::FilterTagDetRes(
                    boxes, ratio_h_, ratio_w_,
                    tmp_src);  // ��resize_img�еõ���bbox ӳ���srcing�е�bbox

                // �����ֵ
                // ��ά�㣬��һά�������ٸ��㣬�ڶ�ά����ÿ�������4�����ꡣ
                // ����ά����ÿ���������������ֵ�� �ֱ��� ���ϣ����ϣ����£����¡�
                for (int i = 0; i < boxes.size(); i++) {
                    Box box;
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

                    image_based_boxes.emplace_back(box);
                }
                job.pro->set_value(image_based_boxes);
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
        if (image.empty()) {
            LOG_INFOE("Image is empty");
            return false;
        }
        job.mono_tensor = tensor_allocator_->query();
        if (job.mono_tensor == nullptr) {
            LOG_INFOE("Tensor allocator query failed.");
            return false;
        }
        job.input = image;
        // Ԥ����
        int     batch_num = 1;
        cv::Mat srcimg, resize_img;

        origin_img_c_ = image.channels();
        origin_img_h_ = image.rows;
        origin_img_w_ = image.cols;

        image.copyTo(srcimg);
        // ��һ��resize
        cv::resize(srcimg, srcimg, cv::Size(input_width_, input_height_));
        OCR::utility::resize_img_type0(srcimg, resize_img, this->max_side_len_, ratio_h_, ratio_w_);
        ratio_h_ = resize_img.rows * 1.0f / image.rows;
        ratio_w_ = resize_img.cols * 1.0f / image.cols;
        OCR::utility::normalize(&resize_img, this->mean_, this->scale_, true);
        std::shared_ptr<float> inBlob(
            new float[1 * 3 * resize_img.rows * resize_img.cols],
            [](float* s) { delete[] s; });
        OCR::utility::permute(&resize_img, inBlob.get());
        // ��ֵ
        resize_img_c  = resize_img.channels();
        resize_img_h_ = resize_img.rows;
        resize_img_w_ = resize_img.cols;
        int data_size = 1 * 3 * resize_img.rows * resize_img.cols;

        CUDATools::AutoDevice auto_device(gpu_);
        auto&                 tensor            = job.mono_tensor->data();
        TRT::CUStream         preprocess_stream = nullptr;
        if (tensor == nullptr) {
            tensor = make_shared<TRT::Tensor>();
            tensor->set_workspace(make_shared<TRT::MixMemory>());
            preprocess_stream = stream_;
            tensor->set_stream(preprocess_stream, false);
            tensor->resize(batch_num, 3, resize_img_h_, resize_img_w_);
        }
        // ���������Ĵ�С
        tensor->resize(batch_num, 3, resize_img_h_, resize_img_w_);
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
    int gpu_ = 0;

    TRT::CUStream stream_ = nullptr;
    json          model_info_;

    int input_width_  = 0;
    int input_height_ = 0;

    // ����
    double det_db_thresh_       = 0.3;
    double det_db_box_thresh_   = 0.5;
    double det_db_unclip_ratio_ = 2.0;
    int kernel_size_            = 2;
    bool   use_polygon_score_   = false;
    bool   use_dilation_        = false;
    bool   enable_detmat_       = false;


    int                resize_img_c  = 0;
    int                resize_img_h_ = 0;
    int                resize_img_w_ = 0;

    int                origin_img_c_  = 0;
    int                origin_img_h_  = 0;
    int                origin_img_w_  = 0;
    int                max_side_len_ = 640;
    float              ratio_h_      = 0.f;
    float              ratio_w_      = 0.f;
    std::vector<float> mean_         = { 0.485f, 0.456f, 0.406f };
    std::vector<float> scale_        = { 1 / 0.229f, 1 / 0.224f, 1 / 0.225f };
};

std::shared_ptr<Algo::Infer> create_infer(const std::string& engine_file, int gpuid, int kernel_size, bool use_dilation, bool enable_detmat, float det_db_box_thresh, float det_db_unclip_ratio, int max_side_len)
{
    std::shared_ptr<InferImpl> instance(new InferImpl());
    if (!instance->startup(engine_file, gpuid, kernel_size, use_dilation, enable_detmat, det_db_box_thresh, det_db_unclip_ratio, max_side_len)) {
        instance.reset();
    }
    return instance;
}
}  // namespace det
};  // namespace OCR


#endif //USE_TRT