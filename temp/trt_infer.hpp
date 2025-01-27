#ifndef __TRT_INFER_H__
#define __TRT_INFER_H__
#if USE_TRT

#    include "./trt-tensor.hpp"
#    include "NvInferRuntime.h"
#    include <NvCaffeParser.h>
#    include <NvInfer.h>
#    include <NvInferPlugin.h>
#    include <algorithm>
#    include <cuda_fp16.h>
#    include <cuda_runtime.h>
#    include <map>
#    include <memory>
#    include <string>
#    include <vector>

namespace TRT
{
// 前向声明
class InferEngine;
// 多上下文
class MutiContext : public std::enable_shared_from_this<MutiContext>
{
public:
    bool                         create_context(InferEngine& input_engine);
    void                         forward(bool sync = true);
    int                          get_max_batch_size();
    CUStream                     get_stream();
    void                         set_stream(CUStream stream);
    void                         synchronize();
    size_t                       get_device_memory_size();
    std::shared_ptr<MixMemory>   get_workspace();
    std::shared_ptr<Tensor>      input(int index = 0);
    std::string                  get_input_name(int index = 0);
    std::shared_ptr<Tensor>      output(int index = 0);
    std::string                  get_output_name(int index = 0);
    std::shared_ptr<Tensor>      tensor(const std::string& name);
    bool                         is_output_name(const std::string& name);
    bool                         is_input_name(const std::string& name);
    void                         set_input(int index, std::shared_ptr<Tensor> tensor);
    void                         set_output(int index, std::shared_ptr<Tensor> tensor);
    nvinfer1::IExecutionContext* get_context();
    void                         print();
    int                          num_output();
    int                          num_input();
    int                          device();
    ~MutiContext();

public:
    cudaStream_t                                 stream_       = nullptr;
    bool                                         owner_stream_ = false;
    std::shared_ptr<nvinfer1::IExecutionContext> context_;
    std::vector<std::shared_ptr<Tensor>>         inputs_;
    std::vector<std::shared_ptr<Tensor>>         outputs_;
    std::vector<int>                             inputs_map_to_ordered_index_;
    std::vector<int>                             outputs_map_to_ordered_index_;
    std::vector<std::string>                     inputs_name_;
    std::vector<std::string>                     outputs_name_;
    std::vector<std::shared_ptr<Tensor>>         orderdBlobs_;
    std::map<std::string, int>                   blobsNameMapper_;
    std::vector<void*>                           bindingsPtr_;
    std::shared_ptr<MixMemory>                   workspace_;
    int                                          device_ = 0;
    std::vector<std::vector<int>>                dim_    = {};
    std::weak_ptr<nvinfer1::ICudaEngine>         engine_;
    int                                          index_ = 0;
    std::weak_ptr<nvinfer1::IRuntime>            runtime_;
};

// engine 文件
class InferEngine
{
public:
    ~InferEngine();
    bool build_model(const void* pdata, size_t size);
    void destroy();

public:
    std::vector<std::shared_ptr<MutiContext>> context_vec_;
    std::shared_ptr<nvinfer1::ICudaEngine>    engine_;
    std::shared_ptr<nvinfer1::IRuntime>       runtime_ = nullptr;
    int                                       device_  = 0;
};

class Infer
{
public:
    virtual int                                   device()                                               = 0;
    virtual std::shared_ptr<std::vector<uint8_t>> serial_engine()                                        = 0;
    virtual std::shared_ptr<MutiContext>          create_context(std::vector<std::vector<int>> dim = {}) = 0;
};

struct DeviceMemorySummary
{
    size_t total;
    size_t available;
};

DeviceMemorySummary    get_current_device_summary();
int                    get_device_count();
int                    get_device();
void                   set_device(int device_id);
std::shared_ptr<Infer> create_engine(const std::string& file);
bool                   init_nv_plugins();
};       // namespace TRT
#endif   // __TRT_INFER_H__
#endif   // USE_TRT