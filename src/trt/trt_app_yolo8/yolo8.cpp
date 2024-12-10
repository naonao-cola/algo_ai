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
#include "../../../include/private/trt/trt_app_yolo8/yolo8.hpp"

namespace yolo8 {


using YoloV5ScaleParams = struct
{
    float r;
    int   dw;
    int   dh;
    int   new_unpad_w;
    int   new_unpad_h;
    bool  flag;
};


/*辅助函数前向声明*/
void resize_unscale(const cv::Mat& mat, cv::Mat& mat_rs, int target_height, int target_width, YoloV5ScaleParams& scale_params);
void generate_bboxes(const YoloV5ScaleParams& scale_params, BoxArray& bbox_collection, std::vector<float>& output_tensors, std::shared_ptr<TRT::Tensor> output, float score_threshold, int img_height, int img_width, int max_nms);
void nms(BoxArray& input, BoxArray& output, float iou_threshold, unsigned int topk);
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
        nms_threshold_ = get_param<float>(config, "nms_threshold", nms_threshold_);
        max_objects_   = get_param<float>(config, "max_objects", max_objects_);
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
        else{
            //此函数永远返回0
            max_batch_size = engine->get_max_batch_size();
        }
       
        
        auto input          = engine->tensor(engine->get_input_name(0));//images
        auto output         = engine->tensor(engine->get_output_name(0));//output0

        input_height_ = input->size(2);
        input_width_  = input->size(3);
        model_info_["memory_size"] = engine->get_device_memory_size() >> 20;
        model_info_["dims"] = {input->size(0), input->size(1), input->size(2), input->size(3)};

        if (dims_.size()>0){
            tensor_allocator_ = make_shared<MonopolyAllocator<TRT::Tensor>>(dims_[0][0]);
        }
        else{
            tensor_allocator_ = make_shared<MonopolyAllocator<TRT::Tensor>>(max_batch_size * 2);
        }
        
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
            LOG_INFOE("yolov8 forward {} ms", infer_time_cost.get_cost_time());

            TRT::TimeCost post_time_cost;
            post_time_cost.start();

            int output_size = output->size(1) * output->size(2);

            for (size_t ibatch = 0; ibatch < infer_batch_size; ibatch++) {
                std::vector<float> featureVector;
                featureVector.resize(output_size);
                checkCudaRuntime(cudaMemcpyAsync(featureVector.data(), output->gpu<float>(ibatch), output_size * sizeof(float), cudaMemcpyDeviceToHost, stream_));
               
                BoxArray bbox_collection;
                generate_bboxes(scale_params_, bbox_collection, featureVector, output, confidence_threshold_, origin_img_h_, origin_img_w_, max_nms_);
                BoxArray detected_boxes;
                nms(bbox_collection, detected_boxes, nms_threshold_, max_objects_);
                post_time_cost.stop();
                auto& job         = fetch_jobs[ibatch];
                
                detected_boxes.pre_time = job.preTime.get_cost_time() / infer_batch_size;
                detected_boxes.infer_time =infer_time_cost.get_cost_time() / infer_batch_size;
                detected_boxes.host_time = post_time_cost.get_cost_time() / infer_batch_size;
                detected_boxes.total_time = detected_boxes.pre_time + detected_boxes.infer_time +detected_boxes.host_time;
                model_info_["infer_time"] = detected_boxes.total_time;
                job.pro->set_value(detected_boxes);
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

        origin_img_c_ = image.channels();
        origin_img_h_ = image.rows;
        origin_img_w_ = image.cols;

        int     batch_num = 1;
        cv::Mat mat_rs;
        resize_unscale(image, mat_rs, input_width_, input_height_, scale_params_);
        cv::Mat canvas;
        cv::cvtColor(mat_rs, canvas, cv::COLOR_BGR2RGB);
        normalize_inplace(canvas,  mean_, scale_);
       
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
    YoloV5ScaleParams        scale_params_;
    int                      input_width_     = 3200;
    int                      input_height_    = 3200;
    int                      output_width_    = 0;
    int                      output_height_   = 0;
    float       mean_            = 0.5f;
    float       scale_           = 1/255.f;
    const unsigned int       max_nms_         = 30000;
    std::vector<std::vector<int>> dims_ = {};
};


void resize_unscale(const cv::Mat& mat, cv::Mat& mat_rs, int target_height, int target_width, YoloV5ScaleParams& scale_params)
{
    if (mat.empty())
        return;
    auto img_height = mat.rows;
    auto img_width  = mat.cols;

    mat_rs = cv::Mat(target_height, target_width, CV_8UC3, cv::Scalar(114, 114, 114));
    // scale ratio (new / old) new_shape(h,w)
    float w_r = (float)target_width / (float)img_width;
    float h_r = (float)target_height / (float)img_height;
    float r   = std::min(w_r, h_r);
    // compute padding
    auto new_unpad_w = static_cast<int>((float)img_width * r);   // floor
    auto new_unpad_h = static_cast<int>((float)img_height * r);  // floor
    int  pad_w       = target_width - new_unpad_w;               // >=0
    int  pad_h       = target_height - new_unpad_h;              // >=0

    int dw = pad_w / 2;
    int dh = pad_h / 2;

    // resize with unscaling
    cv::Mat new_unpad_mat;
    // cv::Mat new_unpad_mat = mat.clone(); // may not need clone.
    cv::resize(mat, new_unpad_mat, cv::Size(new_unpad_w, new_unpad_h));
    new_unpad_mat.copyTo(mat_rs(cv::Rect(dw, dh, new_unpad_w, new_unpad_h)));

    // record scale params.
    scale_params.r           = r;
    scale_params.dw          = dw;
    scale_params.dh          = dh;
    scale_params.new_unpad_w = new_unpad_w;
    scale_params.new_unpad_h = new_unpad_h;
    scale_params.flag        = true;
}
void   normalize_inplace(cv::Mat& mat_inplace, float mean, float scale)
{
    if (mat_inplace.type() != CV_32FC3)
        mat_inplace.convertTo(mat_inplace, CV_32FC3);
    mat_inplace = (mat_inplace - mean) * scale;
}

void generate_bboxes(const YoloV5ScaleParams& scale_params, BoxArray& bbox_collection, std::vector<float>& output_tensors, std::shared_ptr<TRT::Tensor> output, float score_threshold, int img_height, int img_width, int max_nms)
{

     int64_t one_output_length = output->size(0) * output->size(1) * output->size(2);
     int     net_width         = output->size(1);

     cv::Mat output0           = cv::Mat(cv::Size((int)output->size(2), (int)output->size(1)), CV_32F, output_tensors.data()).t();
     float*  pdata             = (float*)output0.data;
     int     rows              = output0.rows; 

     float r_  = scale_params.r;
     int   dw_ = scale_params.dw;
     int   dh_ = scale_params.dh;

      bbox_collection.clear();
     unsigned int count = 0;
     // 一张图片的预测框
     for (int i = 0; i < rows; i++) {
        cv::Mat   scores(1, output->size(1) - 4, CV_32F, pdata + 4);
        cv::Point class_id_point;
        double    max_class_socse;
        cv::minMaxLoc(scores, 0, &max_class_socse, 0, &class_id_point);
        max_class_socse = (float)max_class_socse;

        if (max_class_socse >= score_threshold) {
            // 映射到原图。
            float cx = pdata[0];
            float cy = pdata[1];
            float w  = pdata[2];
            float h  = pdata[3];
            float x1 = ((cx - w / 2.f) - (float)dw_) / r_;
            float y1 = ((cy - h / 2.f) - (float)dh_) / r_;
            float x2 = ((cx + w / 2.f) - (float)dw_) / r_;
            float y2 = ((cy + h / 2.f) - (float)dh_) / r_;
            Box   box;
            box.left        = std::max(0.f, x1);
            box.top         = std::max(0.f, y1);
            box.right       = std::min(x2, (float)img_width - 1.f);
            box.bottom      = std::min(y2, (float)img_height - 1.f);
            box.confidence  = max_class_socse;
            box.class_label = class_id_point.x;
            bbox_collection.push_back(box);
            count += 1;  // limit boxes for nms.
            if (count > max_nms)
                break;
        }
        pdata += net_width;  // 下一个预测框
     }
}

float iou_of(const Box& t_lsh, const Box& t_rsh)
{
     float inner_x1 = std::max(t_lsh.left, t_rsh.left);
     float inner_y1 = std::max(t_lsh.top, t_rsh.top);
     float inner_x2 = std::min(t_lsh.right, t_rsh.right);
     float inner_y2 = std::min(t_lsh.bottom, t_rsh.bottom);

     float inner_h = inner_y2 - inner_y1 + 1.0f;
     float inner_w = inner_x2 - inner_x1 + 1.0f;

     if (inner_h <= 0.f || inner_w <= 0.f)
        return std::numeric_limits<float>::min();
     float inner_area = inner_h * inner_w;
     float lsh_area   = std::max(0.0f, (t_lsh.right - t_lsh.left)) * std::max(0.0f, (t_lsh.bottom - t_lsh.top));
     float rsh_area   = std::max(0.0f, (t_rsh.right - t_rsh.left)) * std::max(0.0f, (t_rsh.bottom - t_rsh.top));
     return float(inner_area / (lsh_area + rsh_area - inner_area));
}

void  nms(BoxArray& input, BoxArray& output, float iou_threshold, unsigned int topk){
     if (input.empty())return;
     std::sort(input.begin(), input.end(), [](const Box& a, const Box& b) {return a.confidence > b.confidence; });
     
     const unsigned int box_num = input.size();
     std::vector<int>   merged(box_num, 0);
     unsigned int count = 0;
    
     for (unsigned int i = 0; i < box_num; ++i) {
        if (merged[i])continue;
        BoxArray buf;
        buf.push_back(input[i]);
        merged[i] = 1;

        for (unsigned int j = i + 1; j < box_num; ++j) {
            if (merged[j])continue;
            float iou = iou_of(input[i], input[j]);
            if (iou > iou_threshold) {
                merged[j] = 1;
                buf.push_back(input[j]);
            }
        }
        output.push_back(buf[0]);
        // keep top k
        count += 1;
        if (count >= topk)break;
     }
}
shared_ptr<Algo::Infer> create_infer(const std::string& engine_file, int gpuid, float confidence_threshold, float nms_threshold, int max_objects, std::vector<std::vector<int>> dims)
{
    shared_ptr<InferImpl> instance(new InferImpl());
     if (!instance->startup(engine_file, gpuid, confidence_threshold, nms_threshold, max_objects,dims)) {
        instance.reset();
    }
    return instance;
}

};  // namespace OCR


#endif //USE_TRT