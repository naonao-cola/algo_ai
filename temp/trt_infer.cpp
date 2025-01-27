#if USE_TRT
#    include "../../../include/private/trt/trt_common/trt_infer.hpp"
#    include "../../../include/private/airuntime/AIRuntimeDefines.h"
#    include "../../../include/private/airuntime/logger.h"
#    include "../../../include/private/trt/trt_common/cuda-tools.hpp"
#    include "../../../include/private/trt/trt_common/ilogger.hpp"
#    include "../../../include/private/trt/trt_common/trt_utils.h"
#

using namespace nvinfer1;
using namespace std;

class Logger : public ILogger
{
public:
    virtual void log(Severity severity, const char* msg) noexcept override
    {
        if (severity == Severity::kINTERNAL_ERROR) {
            LOG_INFOE("NVInfer INTERNAL_ERROR: {}", msg);
            abort();
        }
        else if (severity == Severity::kERROR) {
            LOG_INFOD("NVInfer: {}", msg);
        }
        else if (severity == Severity::kWARNING) {
            LOG_INFOD("NVInfer: {}", msg);
        }
        else if (severity == Severity::kINFO) {
            LOG_INFOD("NVInfer: {}", msg);
        }
        else {
            LOG_INFOD("{}", msg);
        }
    }
};
static Logger gLogger;

namespace TRT
{
template<typename _T>
static void destroy_nvidia_pointer(_T* ptr)
{
    if (ptr)
        ptr->destroy();
}

static TRT::DataType convert_trt_datatype(nvinfer1::DataType dt)
{
    switch (dt) {
    case nvinfer1::DataType::kFLOAT:
        return TRT::DataType::Float;
    case nvinfer1::DataType::kHALF:
        return TRT::DataType::Float16;
    case nvinfer1::DataType::kINT32:
        return TRT::DataType::Int32;
    default:
        INFOE("Unsupport data type %d", dt);
        return TRT::DataType::Float;
    }
};

bool InferEngine::build_model(const void* pdata, size_t size)
{
    destroy();
    if (pdata == nullptr || size == 0)
        return false;
    runtime_ = std::shared_ptr<IRuntime>(createInferRuntime(gLogger), destroy_nvidia_pointer<IRuntime>);
    if (runtime_ == nullptr)
        return false;
    engine_ = std::shared_ptr<ICudaEngine>(runtime_->deserializeCudaEngine(pdata, size, nullptr), destroy_nvidia_pointer<ICudaEngine>);
    return engine_ != nullptr;
}
void InferEngine::destroy()
{
    for (auto& context : context_vec_) {
        context=nullptr;
    }
    context_vec_.clear();
    runtime_.reset();
    engine_.reset();
}
InferEngine::~InferEngine()
{
    INFO("InferEngine::destroy() ++");
    destroy();
    INFO("InferEngine::destroy() --");
}

// 推理模板
class InferImpl : public Infer
{
public:
    virtual ~InferImpl();
    virtual void                                  destroy();
    virtual std::shared_ptr<std::vector<uint8_t>> serial_engine() override;
    virtual bool                                  create_engine(const std::string& file);
    virtual std::shared_ptr<MutiContext>          create_context(std::vector<std::vector<int>> dim = {});
    virtual int                                   device() override;
private:
    void build_engine_input_and_outputs_mapper(std::shared_ptr<MutiContext>& input_context);
private:
    std::shared_ptr<InferEngine> inferEngine_;
    int                          device_ = 0;
};

////////////////////////////////////////////////////////////////////////////////////
InferImpl::~InferImpl()
{
    INFO("InferImpl::destroy() ++");
    destroy();
    INFO("InferImpl::destroy() --");
}

int InferImpl::device()
{
    return device_;
}
void InferImpl::destroy()
{
    int old_device = 0;
    checkCudaRuntime(cudaGetDevice(&old_device));
    checkCudaRuntime(cudaSetDevice(device_));
    if (this->inferEngine_)
        this->inferEngine_.reset();
    checkCudaRuntime(cudaSetDevice(old_device));
}

std::shared_ptr<std::vector<uint8_t>> InferImpl::serial_engine()
{
    auto memory = this->inferEngine_->engine_->serialize();
    auto output = make_shared<std::vector<uint8_t>>((uint8_t*)memory->data(), (uint8_t*)memory->data() + memory->size());
    memory->destroy();
    return output;
}

bool InferImpl::create_engine(const std::string& file)
{
    auto data = load_file(file);
    if (data.empty())
        return false;
    inferEngine_ = std::make_shared<InferEngine>();
    if (!inferEngine_->build_model(data.data(), data.size())) {
        inferEngine_.reset();
        return false;
    }
    cudaGetDevice(&device_);
    return true;
}

std::shared_ptr<MutiContext> InferImpl::create_context(std::vector<std::vector<int>> dim)
{
    std::shared_ptr<MutiContext> context = std::make_shared<MutiContext>();
    context->create_context(*inferEngine_);
    context->dim_ = dim;
    context->workspace_ = std::make_shared<MixMemory>();
    cudaGetDevice(&device_);
    build_engine_input_and_outputs_mapper(context);
    return context;
}

void InferImpl::build_engine_input_and_outputs_mapper(std::shared_ptr<MutiContext>& input_context)
{
    int nbBindings = input_context->engine_.lock()->getNbBindings();
    // 设置动态维度
    if (input_context->dim_.size() > 0) {
        for (int i = 0; i < input_context->dim_.size(); i++) {
            nvinfer1::Dims oriDims = input_context->context_->getBindingDimensions(i);
            for (int j = 0; j < input_context->dim_[i].size(); j++) {
                if (input_context->dim_[i][j] != -1) {
                    oriDims.d[j] = input_context->dim_[i][j];
                }
                input_context->context_->setBindingDimensions(i, oriDims);
            }
        }
    }
    input_context->inputs_.clear();
    input_context->inputs_name_.clear();
    input_context->outputs_.clear();
    input_context->outputs_name_.clear();
    input_context->orderdBlobs_.clear();
    input_context->bindingsPtr_.clear();
    input_context->blobsNameMapper_.clear();
   
    for (int i = 0; i < nbBindings; ++i) {
        // 获取运行时维度
        auto           type        = input_context->engine_.lock()->getBindingDataType(i);
        const char*    bindingName = input_context->engine_.lock()->getBindingName(i);
        nvinfer1::Dims dims = input_context->context_->getBindingDimensions(i);
       
        auto newTensor = std::make_shared<Tensor>(dims.nbDims, dims.d, convert_trt_datatype(type));
        newTensor->set_stream(input_context->stream_);
        newTensor->set_workspace(input_context->workspace_);

        if (input_context->engine_.lock()->bindingIsInput(i)) {
            // if is input
            input_context->inputs_.emplace_back(newTensor);
            input_context->inputs_name_.emplace_back(bindingName);
            input_context->inputs_map_to_ordered_index_.emplace_back(input_context->orderdBlobs_.size());
        }
        else {
            // if is output
            input_context->outputs_.emplace_back(newTensor);
            input_context->outputs_name_.emplace_back(bindingName);
            input_context->outputs_map_to_ordered_index_.emplace_back(input_context->orderdBlobs_.size());
        }
        input_context->blobsNameMapper_[bindingName] = i;
        input_context->orderdBlobs_.emplace_back(newTensor);
    }
    input_context->bindingsPtr_.resize(input_context->orderdBlobs_.size());
}


 MutiContext::~MutiContext(){
    int old_device = 0;
    INFO("MutiContext::destroy() ++");
    checkCudaRuntime(cudaGetDevice(&old_device));
    checkCudaRuntime(cudaSetDevice(device_));
    try {
        if (owner_stream_) {
            cudaStreamDestroy(stream_);
            owner_stream_ = false;
            stream_       = nullptr;
        }
        if (inputs_name_.size() > 0)
            inputs_name_.clear();
        if (outputs_name_.size() > 0)
            outputs_name_.clear();
        if (inputs_.size() > 0)
            inputs_.clear();
        if (outputs_.size() > 0)
            outputs_.clear();
        if (context_) {
            context_.reset();
        }
        if (blobsNameMapper_.size() > 0)
            blobsNameMapper_.clear();
        if (workspace_)
            workspace_.reset();
        
    }
    catch (const std::exception& e) {
        std::cout << e.what() << std::endl;
    }
    checkCudaRuntime(cudaSetDevice(old_device));
    INFO("MutiContext::destroy() --");
 }


bool MutiContext::create_context(InferEngine& input_engine)
{
   
    engine_ = input_engine.engine_;
    runtime_ = input_engine.runtime_;
   
    owner_stream_ = true;
    checkCudaRuntime(cudaStreamCreate(&stream_));
    if (stream_ == nullptr) {
        return false;
    }
    context_ = std::shared_ptr<IExecutionContext>(input_engine.engine_->createExecutionContext(), destroy_nvidia_pointer<IExecutionContext>);
    input_engine.context_vec_.emplace_back(shared_from_this());
    return true;
}

void MutiContext::print()
{
    if (!context_) {
        INFOW("MutiContext print, nullptr.");
        return;
    }
    INFO("Infer %p detail", this);
    INFO("\tBase device: %s", CUDATools::device_description().c_str());
    INFO("\tMax Batch Size: %d", this->get_max_batch_size());
    INFO("\tInputs: %d", inputs_.size());
    for (int i = 0; i < inputs_.size(); ++i) {
        auto& tensor = inputs_[i];
        auto& name   = inputs_name_[i];
        INFO("\t\t%d.%s : shape {%s}, %s", i, name.c_str(), tensor->shape_string(), data_type_string(tensor->type()));
    }
    INFO("\tOutputs: %d", outputs_.size());
    for (int i = 0; i < outputs_.size(); ++i) {
        auto& tensor = outputs_[i];
        auto& name   = outputs_name_[i];
        INFO("\t\t%d.%s : shape {%s}, %s", i, name.c_str(), tensor->shape_string(), data_type_string(tensor->type()));
    }
}


size_t MutiContext::get_device_memory_size()
{
    return context_->getEngine().getDeviceMemorySize() + inputs_[0]->bytes() + outputs_[0]->bytes();
}

void MutiContext::set_stream(CUStream stream)
{
    if (owner_stream_) {
        if (stream_) {
            cudaStreamDestroy(stream_);
        }
        owner_stream_ = false;
    }
    stream_ = stream;

    for (auto& t : orderdBlobs_)
        t->set_stream(stream);
}

CUStream MutiContext::get_stream()
{
    return stream_;
}

int MutiContext::device()
{
    return device_;
}

void MutiContext::synchronize()
{
    checkCudaRuntime(cudaStreamSynchronize(stream_));
}

bool MutiContext::is_output_name(const std::string& name)
{
    return std::find(outputs_name_.begin(), outputs_name_.end(), name) != outputs_name_.end();
}

bool MutiContext::is_input_name(const std::string& name)
{
    return std::find(inputs_name_.begin(), inputs_name_.end(), name) != inputs_name_.end();
}

nvinfer1::IExecutionContext* MutiContext::get_context()
{
    return context_.get();
}
void MutiContext::forward(bool sync)
{
    int inputBatchSize = inputs_[0]->size(0);
    for (int i = 0; i < engine_.lock()->getNbBindings(); ++i) {
        auto dims = engine_.lock()->getBindingDimensions(i);
        auto type = engine_.lock()->getBindingDataType(i);
        dims.d[0] = inputBatchSize;
        if (engine_.lock()->bindingIsInput(i)) {
            context_->setBindingDimensions(i, dims);
        }
       
    }
    for (int i = 0; i < outputs_.size(); ++i) {
        outputs_[i]->resize_single_dim(0, inputBatchSize);
        outputs_[i]->to_gpu(false);
    }

    for (int i = 0; i < orderdBlobs_.size(); ++i)
        bindingsPtr_[i] = orderdBlobs_[i]->gpu();

    void** bindingsptr = bindingsPtr_.data();

    bool execute_result = context_->enqueueV2(bindingsptr, stream_, nullptr);

    if (!execute_result) {
        auto code = cudaGetLastError();
        INFOF("execute fail, code %d[%s], message %s", code, cudaGetErrorName(code), cudaGetErrorString(code));
    }

    if (sync) {
        synchronize();
    }
}

std::shared_ptr<MixMemory> MutiContext::get_workspace()
{
    return workspace_;
}

int MutiContext::num_input()
{
    return static_cast<int>(this->inputs_.size());
}

int MutiContext::num_output()
{
    return static_cast<int>(this->outputs_.size());
}

void MutiContext::set_input(int index, std::shared_ptr<Tensor> tensor)
{
    if (index < 0 || index >= inputs_.size()) {
        INFOF("Input index[%d] out of range [size=%d]", index, inputs_.size());
    }

    this->inputs_[index]            = tensor;
    int order_index                 = inputs_map_to_ordered_index_[index];
    this->orderdBlobs_[order_index] = tensor;
}

void MutiContext::set_output(int index, std::shared_ptr<Tensor> tensor)
{
    if (index < 0 || index >= outputs_.size()) {
        INFOF("Output index[%d] out of range [size=%d]", index, outputs_.size());
    }
    this->outputs_[index]           = tensor;
    int order_index                 = outputs_map_to_ordered_index_[index];
    this->orderdBlobs_[order_index] = tensor;
}

std::shared_ptr<Tensor> MutiContext::input(int index)
{
    if (index < 0 || index >= inputs_.size()) {
        INFOF("Input index[%d] out of range [size=%d]", index, inputs_.size());
    }
    return this->inputs_[index];
}

std::string MutiContext::get_input_name(int index)
{
    if (index < 0 || index >= inputs_name_.size()) {
        INFOF("Input index[%d] out of range [size=%d]", index, inputs_name_.size());
    }
    return inputs_name_[index];
}

std::shared_ptr<Tensor> MutiContext::output(int index)
{
    if (index < 0 || index >= outputs_.size()) {
        INFOF("Output index[%d] out of range [size=%d]", index, outputs_.size());
    }
    return outputs_[index];
}

std::string MutiContext::get_output_name(int index)
{
    if (index < 0 || index >= outputs_name_.size()) {
        INFOF("Output index[%d] out of range [size=%d]", index, outputs_name_.size());
    }
    return outputs_name_[index];
}

int MutiContext::get_max_batch_size()
{
    return engine_.lock()->getMaxBatchSize();
}

std::shared_ptr<Tensor> MutiContext::tensor(const std::string& name)
{
    auto node = this->blobsNameMapper_.find(name);
    if (node == this->blobsNameMapper_.end()) {
        INFOF("Could not found the input/output node '%s', please makesure your "
              "model",
              name.c_str());
    }
    return orderdBlobs_[node->second];
}


std::shared_ptr<Infer> create_engine(const std::string& file)
{
    std::shared_ptr<InferImpl> Infer = std::make_shared<InferImpl>();
    if (!Infer->create_engine(file)) {
        Infer.reset();
    }
    return Infer;
}


DeviceMemorySummary get_current_device_summary()
{
    DeviceMemorySummary info;
    checkCudaRuntime(cudaMemGetInfo(&info.available, &info.total));
    return info;
}

int get_device_count()
{
    int count = 0;
    checkCudaRuntime(cudaGetDeviceCount(&count));
    return count;
}

int get_device()
{
    int device = 0;
    checkCudaRuntime(cudaGetDevice(&device));
    return device;
}

void set_device(int device_id)
{
    if (device_id == -1)
        return;
    checkCudaRuntime(cudaSetDevice(device_id));
}

};   // namespace TRT

#endif   // USE_TRT
