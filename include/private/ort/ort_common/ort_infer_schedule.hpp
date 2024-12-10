#ifndef INFER_CONTROLLER_HPP
#define INFER_CONTROLLER_HPP
#if USE_ORT
#include "../../../include/private/airuntime/logger.h"
#include "../../../private/ort/ort_common/ort_infer.hpp"
#include "../../../public/AIRuntimeUtils.h"
#include "onnxruntime_cxx_api.h"
#include <condition_variable>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>

namespace ort {
template <
    class Input,
    class Output,
    class StartParam    = std::tuple<std::string, int>,
    class JobAdditional = int>
class InferController
{
public:
    struct Job
    {
        Input                                 input;
        Output                                output;
        JobAdditional                         additional;
        std::shared_ptr<float[]>              input_value;
        std::shared_ptr<std::promise<Output>> pro;
        TimeCost                              preTime;
        TimeCost                              inferTime;
        TimeCost                              hostTime;
        TimeCost                              totalime;
    };

    virtual ~InferController()
    {
        stop();
    }

    void stop()
    {
        _run = false;
        _cond.notify_all();

        ////////////////////////////////////////// cleanup jobs
        {
            std::unique_lock<std::mutex> l(_jobs_lock);
            while (!_jobs.empty()) {
                auto& item = _jobs.front();
                if (item.pro)
                    item.pro->set_value(Output());
                _jobs.pop();
            }
        };

        if (_worker) {
            _worker->join();
            _worker.reset();
        }
    }

    bool startup(const StartParam& param)
    {
        _run = true;

        std::promise<bool> pro;
        _start_param = param;
        _worker      = std::make_shared<std::thread>(&InferController::worker, this, std::ref(pro));
        return pro.get_future().get();
    }

    virtual std::shared_future<Output> commit(const Input& input)
    {
        Job job;
        job.pro = std::make_shared<std::promise<Output>>();
        if (!preprocess(job, input)) {
            job.pro->set_value(Output());
            return job.pro->get_future();
        }

        ///////////////////////////////////////////////////////////
        {
            std::unique_lock<std::mutex> l(_jobs_lock);
            _jobs.push(job);
        };
        _cond.notify_one();
        return job.pro->get_future();
    }

    virtual std::vector<std::shared_future<Output>> commits(const std::vector<Input>& inputs)
    {
        int                                     batch_size = std::min((int)inputs.size(), this->_max_batch_size);
        std::vector<Job>                        jobs(inputs.size());
        std::vector<std::shared_future<Output>> results(inputs.size());

        int nepoch = (inputs.size() + batch_size - 1) / batch_size;
        for (int epoch = 0; epoch < nepoch; ++epoch) {
            int begin = epoch * batch_size;
            int end   = std::min((int)inputs.size(), begin + batch_size);

            for (int i = begin; i < end; ++i) {
                Job& job = jobs[i];
                job.pro  = std::make_shared<std::promise<Output>>();
                if (!preprocess(job, inputs[i])) {
                    job.pro->set_value(Output());
                }
                results[i] = job.pro->get_future();
            }

            ///////////////////////////////////////////////////////////
            {
                std::unique_lock<std::mutex> l(_jobs_lock);
                for (int i = begin; i < end; ++i) {
                    _jobs.emplace(std::move(jobs[i]));
                };
            }
            _cond.notify_one();
        }
        return results;
    }

protected:
    virtual bool preprocess(Job& job, const Input& input)                       = 0;
    virtual bool postprocess(Job& job, std::vector<Ort::Value>& output_tensors) = 0;

    // 目前是单输入
    virtual void worker(std::promise<bool>& result)
    {
        std::string   onnx_file = std::get<0>(_start_param);
        BasicOrtInfer engine(onnx_file);
        _engine_ptr = &engine;
        if (_engine_ptr->get_session() == nullptr) {
            LOG_INFOE("Engine {} load failed", onnx_file.c_str());
            result.set_value(false);
            return;
        }
        _engine_ptr->print_debug_string();

        _input_shapes  = _engine_ptr->get_input_node_dims();
        _output_shapes = _engine_ptr->get_out_node_dims();
        _model_info["dims"] = {_input_shapes[0][0], _input_shapes[0][1], _input_shapes[0][2], _input_shapes[0][3]};

        auto input_names = _engine_ptr->get_input_node_names();
        auto output_name = _engine_ptr->get_output_node_names();

        // 第一个输入的的第一个维度为batch
        int max_batch_size = _input_shapes.at(0).at(0) <= 0 ? 1 : _input_shapes.at(0).at(0);

        std::vector<Job>        fetch_jobs;
        std::vector<Ort::Value> input_data;

        result.set_value(true);

        while (get_jobs_and_wait(fetch_jobs, max_batch_size)) {
            int infer_batch_size = fetch_jobs.size();

            for (int ibatch = 0; ibatch < infer_batch_size; ++ibatch) {
                auto&                job = fetch_jobs[ibatch];
                std::vector<int64_t> tensor_dims{ 1, _input_shapes.at(0).at(1), _input_shapes.at(0).at(2), _input_shapes.at(0).at(3) };
                input_data.emplace_back(Ort::Value::CreateTensor<float>(_engine_ptr->get_memory_info(), job.input_value.get(), _input_shapes.at(0).at(1) * _input_shapes.at(0).at(2) * _input_shapes.at(0).at(3), tensor_dims.data(), tensor_dims.size()));
            }

            TimeCost infer_time_cost;
            infer_time_cost.start();
            auto output_tensors = _engine_ptr->get_session()->Run(Ort::RunOptions{ nullptr }, input_names.data(), input_data.data(), infer_batch_size, output_name.data(), infer_batch_size);
            infer_time_cost.stop();

            for (int ibatch = 0; ibatch < infer_batch_size; ++ibatch) {
                auto& job     = fetch_jobs[ibatch];
                job.inferTime = infer_time_cost;
                postprocess(job, output_tensors);
            }
            input_data.clear();
            fetch_jobs.clear();
        }
        LOG_INFOE("Engine destroy.");
    }

    virtual bool get_jobs_and_wait(std::vector<Job>& fetch_jobs, int max_size)
    {
        std::unique_lock<std::mutex> l(_jobs_lock);
        _cond.wait(l, [&]() {
            return !_run || !_jobs.empty();
        });

        if (!_run)
            return false;

        fetch_jobs.clear();
        for (int i = 0; i < max_size && !_jobs.empty(); ++i) {
            fetch_jobs.emplace_back(std::move(_jobs.front()));
            _jobs.pop();
        }
        return true;
    }

    virtual bool get_job_and_wait(Job& fetch_job)
    {
        std::unique_lock<std::mutex> l(_jobs_lock);
        _cond.wait(l, [&]() {
            return !_run || !_jobs.empty();
        });

        if (!_run)
            return false;

        fetch_job = std::move(_jobs.front());
        _jobs.pop();
        return true;
    }

protected:
    StartParam                        _start_param;
    std::atomic<bool>                 _run;
    std::mutex                        _jobs_lock;
    std::queue<Job>                   _jobs;
    std::shared_ptr<std::thread>      _worker;
    std::condition_variable           _cond;
    int                               _max_batch_size = 1;
    BasicOrtInfer*                    _engine_ptr;
    std::vector<std::vector<int64_t>> _input_shapes;
    std::vector<std::vector<int64_t>> _output_shapes;
    nlohmann::json  _model_info;
};
}  // namespace ort
#endif  // INFER_CONTROLLER_HPP
#endif //USE_ORT